"""
Batch evaluation script for the Themis pipeline.

Runs all labeled petitions through the pipeline using local prompt files
and the HarnessConfig, saves per-petition execution traces, and outputs
aggregate metrics.

Usage:
    python -m scripts.evaluate

Expects a manifest file at scripts/meta_harness/dataset.json:
[
    {"pdf": "path/to/petition.pdf", "expected_id": "precedent-id-123"},
    ...
]

Traces are written to scripts/meta_harness/traces/.
Aggregate metrics are written to scripts/meta_harness/metrics.json.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.meta_harness.config import HarnessConfig

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

_langfuse = Langfuse()
from themis.infra.embeddings import resolve_embedding_stack
from themis.infra.prompts import LocalPromptProvider
from themis.infra.providers import resolve_providers
from themis.infra.settings import PROVIDER
from themis.interfaces.prompts import PromptProvider
from themis.models.domain import RankedPrecedent, RetrievedPrecedent
from themis.services.judge import (
    _build_candidates_text,
    _parse_json_response,
    _parse_text_response,
    _rank_candidates,
)
from themis.services.pdf import extract_text
from themis.services.retrieval import _dedup, extract_legal_queries

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_HARNESS_DIR = Path(__file__).resolve().parent / "meta_harness"
_TRACES_DIR = _HARNESS_DIR / "traces"
_DATASET_PATH = _HARNESS_DIR / "dataset.json"
_METRICS_PATH = _HARNESS_DIR / "metrics.json"


# ── Trace data structures ──────────────────────────────────────────────


@dataclass
class PetitionTrace:
    """Complete execution trace for a single petition evaluation."""

    petition_file: str
    expected_id: str
    petition_length: int

    # Config snapshot
    config: dict

    # Retrieval stage
    extracted_queries: list[str]
    retrieved_count: int
    retrieved_ids: list[str]
    retrieved_scores: list[float]

    # Judge stage
    candidates_text_sent: str
    judge_raw_response: str
    ranked_results: list[dict]

    # Outcome
    retrieved: bool
    retrieval_rank: int | None
    similarity_score: float | None
    pipeline_rank: int | None
    classification: str | None
    judge_score: int | None
    hit_at_k: dict[str, bool]
    reciprocal_rank: float

    elapsed_seconds: float


@dataclass
class AggregateMetrics:
    total: int
    config: dict
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    hit_at_25: float
    mrr: float
    retrieval_recall: float
    per_petition: list[dict]


# ── Pipeline steps (decomposed for tracing) ────────────────────────────


def _retrieve(
    petition_text: str,
    query_provider,
    repository,
    embedding_provider,
    prompt_provider: PromptProvider,
    cfg: HarnessConfig,
) -> tuple[list[str], list[RetrievedPrecedent]]:
    """Run retrieval and return (extracted_queries, retrieved_precedents)."""
    if cfg.use_queries:
        queries = extract_legal_queries(
            petition_text, query_provider, prompt_provider,
            prompt_name=cfg.query_prompt,
            temperature=cfg.query_extraction_temperature,
        )
    else:
        queries = []
    embeddings = embedding_provider.embed([petition_text] + queries)

    limit = cfg.candidates * cfg.results_per_embedding_multiplier
    with ThreadPoolExecutor(max_workers=len(embeddings)) as pool:
        futures = [pool.submit(repository.search, emb, limit) for emb in embeddings]
        hits: list[RetrievedPrecedent] = []
        for future in as_completed(futures):
            hits.extend(future.result())

    return queries, [
        hit for hit in _dedup(hits)
        if hit.cosine_similarity >= cfg.vector_score_threshold
    ][:cfg.candidates]


def _judge(
    petition_text: str,
    candidates: list[RetrievedPrecedent],
    judge_provider,
    prompt_provider: PromptProvider,
    cfg: HarnessConfig,
) -> tuple[str, list[RankedPrecedent]]:
    """Run judge and return (raw_llm_response, ranked_precedents)."""
    if not candidates:
        return "", []

    truncated_petition = petition_text[:cfg.petition_text_limit]
    prompt_content = prompt_provider.compile(
        cfg.judge_prompt,
        petition_text=truncated_petition,
        candidates_text=_build_candidates_text(candidates, cfg.field_limits),
    )
    messages = [{"role": "user", "content": prompt_content}]

    if cfg.use_json:
        from themis.services.judge import _JUDGE_SCHEMA
        data = judge_provider.complete_json(messages, temperature=cfg.judge_temperature, response_schema=_JUDGE_SCHEMA)
        raw_response = json.dumps(data, ensure_ascii=False)
        judgments = _parse_json_response(data, len(candidates))
    else:
        raw_response = judge_provider.complete(messages, temperature=cfg.judge_temperature)
        judgments = _parse_text_response(
            raw_response, cfg.use_score, judge_provider.name, len(candidates),
        )

    return raw_response, _rank_candidates(candidates, judgments)


# ── Evaluation logic ───────────────────────────────────────────────────


def _find_rank(items: list, expected_id: str) -> int | None:
    """Return 1-based rank of expected_id in items, or None if absent."""
    return next(
        (i + 1 for i, item in enumerate(items) if item.id == expected_id),
        None,
    )


def evaluate_petition(
    pdf_path: Path,
    expected_id: str,
    query_provider,
    judge_provider,
    repository,
    embedding_provider,
    prompt_provider: PromptProvider,
    cfg: HarnessConfig,
) -> PetitionTrace:
    """Run the full pipeline on a single petition and return a detailed trace."""

    start = time.monotonic()

    petition_text = extract_text(pdf_path.read_bytes())

    queries, retrieved = _retrieve(
        petition_text, query_provider, repository,
        embedding_provider, prompt_provider, cfg,
    )

    candidates_text = _build_candidates_text(retrieved, cfg.field_limits)
    judge_raw, ranked = _judge(
        petition_text, retrieved, judge_provider, prompt_provider, cfg,
    )

    # ── Compute metrics ────────────────────────────────────────────
    sorted_by_sim = sorted(retrieved, key=lambda p: p.cosine_similarity, reverse=True)
    retrieval_rank = _find_rank(sorted_by_sim, expected_id)
    pipeline_rank = _find_rank(ranked, expected_id)
    is_retrieved = retrieval_rank is not None

    sim_score = next(
        (p.cosine_similarity for p in sorted_by_sim if p.id == expected_id), None,
    )
    classification = next((p.relevance_label for p in ranked if p.id == expected_id), None)
    judge_score = next((p.relevance_score for p in ranked if p.id == expected_id), None)

    hit_at_k = {
        str(k): any(p.id == expected_id for p in sorted_by_sim[:k])
        for k in [1, 5, 10, 25]
    }
    rr = (1.0 / pipeline_rank) if pipeline_rank else 0.0

    return PetitionTrace(
        petition_file=str(pdf_path),
        expected_id=expected_id,
        petition_length=len(petition_text),
        config=asdict(cfg),
        extracted_queries=queries,
        retrieved_count=len(retrieved),
        retrieved_ids=[p.id for p in sorted_by_sim],
        retrieved_scores=[round(p.cosine_similarity, 4) for p in sorted_by_sim],
        candidates_text_sent=candidates_text,
        judge_raw_response=judge_raw,
        ranked_results=[
            {
                "id": p.id, "rank": i + 1,
                "relevance_label": p.relevance_label,
                "relevance_score": p.relevance_score,
                "cosine_similarity": round(p.cosine_similarity, 4),
                "explanation": p.explanation,
            }
            for i, p in enumerate(ranked)
        ],
        retrieved=is_retrieved,
        retrieval_rank=retrieval_rank,
        similarity_score=round(sim_score, 4) if sim_score is not None else None,
        pipeline_rank=pipeline_rank,
        classification=classification,
        judge_score=judge_score,
        hit_at_k=hit_at_k,
        reciprocal_rank=round(rr, 4),
        elapsed_seconds=round(time.monotonic() - start, 2),
    )


def compute_aggregate(traces: list[PetitionTrace], cfg: HarnessConfig) -> AggregateMetrics:
    """Compute aggregate metrics across all petition traces."""
    n = len(traces)
    if n == 0:
        return AggregateMetrics(0, asdict(cfg), 0, 0, 0, 0, 0, 0, [])

    return AggregateMetrics(
        total=n,
        config=asdict(cfg),
        hit_at_1=round(sum(1 for t in traces if t.hit_at_k.get("1")) / n, 4),
        hit_at_5=round(sum(1 for t in traces if t.hit_at_k.get("5")) / n, 4),
        hit_at_10=round(sum(1 for t in traces if t.hit_at_k.get("10")) / n, 4),
        hit_at_25=round(sum(1 for t in traces if t.hit_at_k.get("25")) / n, 4),
        mrr=round(sum(t.reciprocal_rank for t in traces) / n, 4),
        retrieval_recall=round(sum(1 for t in traces if t.retrieved) / n, 4),
        per_petition=[
            {
                "file": t.petition_file,
                "expected_id": t.expected_id,
                "retrieved": t.retrieved,
                "retrieval_rank": t.retrieval_rank,
                "pipeline_rank": t.pipeline_rank,
                "classification": t.classification,
                "judge_score": t.judge_score,
                "hit_at_1": t.hit_at_k.get("1", False),
                "hit_at_5": t.hit_at_k.get("5", False),
                "reciprocal_rank": t.reciprocal_rank,
                "elapsed_seconds": t.elapsed_seconds,
            }
            for t in traces
        ],
    )


# ── Langfuse logging ───────────────────────────────────────────────────

def _langfuse_petition_context(pdf_path: Path, expected_id: str, run_metadata: dict):
    """Context manager that creates the top-level Langfuse trace for one petition."""
    from langfuse.types import TraceContext
    trace_id = _langfuse.create_trace_id()
    return _langfuse.start_as_current_observation(
        trace_context=TraceContext(trace_id=trace_id),
        name=f"harness/{pdf_path.stem}",
        input={"petition_file": str(pdf_path), "expected_id": expected_id},
        metadata=run_metadata,
    )


def _score_langfuse_trace(trace: PetitionTrace) -> None:
    """Update the current observation with results and attach scores."""
    _langfuse.update_current_span(
        output={
            "pipeline_rank": trace.pipeline_rank,
            "judge_score": trace.judge_score,
            "classification": trace.classification,
            "ranked_results": trace.ranked_results[:5],
        },
        metadata={
            "extracted_queries": trace.extracted_queries,
            "retrieved_ids": trace.retrieved_ids,
        },
    )
    scores = [
        ("hit_at_1", 1.0 if trace.pipeline_rank == 1 else 0.0),
        ("reciprocal_rank", trace.reciprocal_rank),
        ("retrieved", 1.0 if trace.retrieved else 0.0),
    ]
    if trace.judge_score is not None:
        scores.append(("judge_score", float(trace.judge_score)))
    for name, value in scores:
        _langfuse.score_current_trace(name=name, value=value)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    if not _DATASET_PATH.exists():
        logger.error("Dataset not found at %s", _DATASET_PATH)
        logger.error('Expected format: [{"pdf": "path/to/file.pdf", "expected_id": "..."}]')
        sys.exit(1)

    cfg = HarnessConfig()
    run_name = f"harness-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    logger.info("Harness run: %s", run_name)
    logger.info("Harness config: %s", asdict(cfg))

    dataset = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d labeled petitions from %s", len(dataset), _DATASET_PATH)

    _TRACES_DIR.mkdir(parents=True, exist_ok=True)

    embedding_provider_name = os.getenv("EMBEDDING_PROVIDER", "openai")
    query_provider, judge_provider = resolve_providers(PROVIDER)
    embedding_provider, repository = resolve_embedding_stack(embedding_provider_name)
    prompt_provider = LocalPromptProvider()

    run_metadata = {
        "run_name": run_name,
        "config": asdict(cfg),
        "providers": {
            "llm": PROVIDER,
            "query_model": getattr(query_provider, "name", PROVIDER),
            "judge_model": getattr(judge_provider, "name", PROVIDER),
            "embedding": embedding_provider_name,
            "query_model_id": getattr(query_provider, "_model", None),
            "judge_model_id": getattr(judge_provider, "_model", None),
            "embedding_model_id": getattr(embedding_provider, "_model", None) or getattr(embedding_provider, "_MODEL", None),
            "prompts": "local",
        },
    }
    logger.info("Providers: %s", run_metadata["providers"])

    traces: list[PetitionTrace] = []

    for i, entry in enumerate(dataset, start=1):
        pdf_path = Path(entry["pdf"])
        if not pdf_path.is_absolute():
            pdf_path = _HARNESS_DIR / pdf_path

        expected_id = entry["expected_id"]
        logger.info("[%d/%d] Evaluating: %s (expected: %s)", i, len(dataset), pdf_path.name, expected_id)

        try:
            with _langfuse_petition_context(pdf_path, expected_id, run_metadata):
                trace = evaluate_petition(
                    pdf_path, expected_id,
                    query_provider, judge_provider,
                    repository, embedding_provider,
                    prompt_provider, cfg,
                )
                _score_langfuse_trace(trace)

            traces.append(trace)
            trace_path = _TRACES_DIR / f"petition_{i:02d}.json"
            trace_path.write_text(
                json.dumps(asdict(trace), ensure_ascii=False, indent=2), encoding="utf-8",
            )
            logger.info(
                "  → retrieved=%s  retrieval_rank=%s  pipeline_rank=%s  (%.1fs)",
                trace.retrieved, trace.retrieval_rank, trace.pipeline_rank, trace.elapsed_seconds,
            )
        except Exception:
            logger.exception("  → FAILED: %s", pdf_path.name)

        if i < len(dataset) and cfg.delay_between_petitions > 0:
            sleep(cfg.delay_between_petitions)

    metrics = compute_aggregate(traces, cfg)
    _METRICS_PATH.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8",
    )

    logger.info("─" * 50)
    logger.info("RESULTS (%d/%d succeeded)", len(traces), len(dataset))
    logger.info("  hit@1:  %.1f%%", metrics.hit_at_1 * 100)
    logger.info("  hit@5:  %.1f%%", metrics.hit_at_5 * 100)
    logger.info("  hit@10: %.1f%%", metrics.hit_at_10 * 100)
    logger.info("  hit@25: %.1f%%", metrics.hit_at_25 * 100)
    logger.info("  MRR:    %.4f", metrics.mrr)
    logger.info("  Retrieval recall: %.1f%%", metrics.retrieval_recall * 100)
    logger.info("Traces saved to %s", _TRACES_DIR)
    logger.info("Metrics saved to %s", _METRICS_PATH)
    _langfuse.flush()
    logger.info("Langfuse traces flushed (run: %s)", run_name)


if __name__ == "__main__":
    main()
