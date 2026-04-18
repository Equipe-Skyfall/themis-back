# Themis — Legal Precedent Retrieval System

## What This Project Does

Themis helps Brazilian judges and lawyers find the most relevant legal precedents for a given petition. A PDF petition is uploaded, processed through a retrieval + LLM ranking pipeline, and the most relevant precedents are returned ranked by applicability.

## Architecture

```
PDF → extract text → LLM extracts 3-5 legal queries → embed (petition + queries)
    → parallel vector search (MongoDB Atlas) → dedup & filter → LLM judge scores 0-10
    → sorted results returned to user
```

**Stack:** FastAPI, MongoDB Atlas (vector search), OpenAI embeddings, Groq/OpenAI LLMs, Langfuse (observability), PyMUPDF (PDF extraction).

## Key Files

| File | Purpose |
|------|---------|
| `themis/services/retrieval.py` | Query extraction + vector search pipeline |
| `themis/services/judge.py` | LLM judge — classifies and scores candidates |
| `themis/use_cases/analyze.py` | `PetitionAnalyzer` — orchestrates the full pipeline |
| `themis/infra/providers.py` | LLM provider implementations (OpenAI, Groq) with retry |
| `themis/infra/prompts.py` | `LangfusePromptProvider` (production) and `LocalPromptProvider` (harness) |
| `themis/infra/repositories.py` | MongoDB Atlas vector search repository |
| `themis/infra/settings.py` | Environment-based configuration |
| `themis/models/domain.py` | Core entities: `Precedent`, `RetrievedPrecedent`, `RankedPrecedent`, `Judgment` |
| `themis/interfaces/` | Protocol definitions (`ChatProvider`, `EmbeddingProvider`, `PrecedentRepository`, `PromptProvider`) |
| `themis/api/routes.py` | FastAPI endpoints: `/petition/analyze`, `/petition/evaluate` |
| `themis/prompts/` | Local prompt templates for the harness (uses `{{variable}}` placeholders) |

## Design Patterns

- **Protocol-based DI** — all dependencies are typed as Protocols, implementations are injected
- **Strategy pattern** — swappable LLM providers, embedding providers, prompt sources
- **`@observe` decorators** — Langfuse tracing on all pipeline steps
- **Keyword-only configurable params** — pipeline functions accept optional overrides with production defaults

## Dual Prompt System

- **Production** (`/petition/analyze`): prompts are fetched from Langfuse via `LangfusePromptProvider`
- **Harness** (`scripts/evaluate.py`): prompts are read from local files in `themis/prompts/` via `LocalPromptProvider`

When optimizing, edit the local prompt files freely. Once satisfied with results, the user manually copies them to Langfuse for production.

## Running the Project

```bash
pip install -r requirements.txt
# Set env vars (see .env.example)
python run.py  # starts FastAPI on port 8000
```

## Code Quality

- Prioritize readability, maintainability, and elegant code
- Follow existing patterns — do not introduce new abstractions unless justified
- Keep functions focused and names descriptive
- Do not add hybrid search (BM25 + vector) — it was tested and only added noise

---

# Meta-Harness Optimization

## What Is It

An iterative optimization loop where you (the agent) analyze pipeline execution traces, diagnose why specific petitions failed to retrieve the correct precedent, and propose targeted changes to config/prompts/code.

## Current Metrics (baseline)

- **hit@1: 60%** — correct precedent ranked #1
- **hit@5: 84%** — correct precedent in top 5
- 15 labeled petitions available for evaluation

## The Optimization Loop

```
1. User runs:  python -m scripts.evaluate
2. User asks:  "analyze the traces and improve the pipeline"
3. You do:     read scripts/meta_harness/traces/*.json and scripts/meta_harness/metrics.json
4. You do:     diagnose failure patterns across petitions
5. You do:     edit config.py, prompts, or pipeline code
6. User runs:  python -m scripts.evaluate  (re-evaluate)
7. Repeat
```

## Key Harness Files

| File | Purpose |
|------|---------|
| `scripts/meta_harness/config.py` | `HarnessConfig` — ALL tunable parameters in one place |
| `scripts/meta_harness/dataset.json` | Labeled test set: `[{"pdf": "petitions/x.pdf", "expected_id": "..."}]` |
| `scripts/meta_harness/traces/` | Per-petition JSON traces (written by evaluate.py) |
| `scripts/meta_harness/metrics.json` | Aggregate metrics (written by evaluate.py) |
| `scripts/evaluate.py` | Batch evaluation script |
| `themis/prompts/query-extraction.txt` | Query extraction prompt template |
| `themis/prompts/judge-v2.txt` | Judge prompt template (score mode) |
| `themis/prompts/judge.txt` | Judge prompt template (label mode) |

## What You Can Tune

Edit `scripts/meta_harness/config.py`:

```python
@dataclass
class HarnessConfig:
    candidates: int = 25                        # how many candidates reach the judge
    vector_score_threshold: float = 0.70        # minimum cosine similarity
    query_extraction_temperature: float = 0.0   # LLM temperature for query extraction
    results_per_embedding_multiplier: int = 4   # ANN results per embedding = candidates * this
    use_json: bool = False                      # JSON mode for judge (eliminates parse failures)
    use_score: bool = True                      # score mode (0-10) vs label mode (3-tier)
    judge_temperature: float = 1.0              # LLM temperature for judge
    field_limits: dict = {                      # char limits per field sent to judge
        "tese": 1000, "questao": 400, "textoEmenta": 1000,
    }
    query_prompt: str = "query-extraction"      # prompt file name
    judge_prompt: str = "judge-v2"              # prompt file name
```

Edit prompt templates in `themis/prompts/`:
- `{{petition_text}}` — the extracted petition text
- `{{candidates_text}}` — formatted candidate precedents

## Trace Format

Each `scripts/meta_harness/traces/petition_XX.json` contains:

```json
{
    "petition_file": "path/to/file.pdf",
    "expected_id": "the-correct-precedent-id",
    "config": { /* HarnessConfig snapshot */ },
    "extracted_queries": ["query 1", "query 2", ...],
    "retrieved_ids": ["id1", "id2", ...],
    "retrieved_scores": [0.89, 0.85, ...],
    "candidates_text_sent": "formatted text sent to judge",
    "judge_raw_response": "raw LLM output",
    "ranked_results": [
        {"id": "...", "rank": 1, "relevance_score": 9, "relevance_label": "aplicavel", "explanation": "..."},
        ...
    ],
    "retrieved": true,
    "retrieval_rank": 3,
    "pipeline_rank": 5,
    "classification": "possivelmente aplicavel",
    "judge_score": 6,
    "hit_at_k": {"1": false, "5": true, "10": true, "25": true},
    "reciprocal_rank": 0.2
}
```

## How to Diagnose Failures

When analyzing traces, categorize each failure:

1. **Retrieval miss** (`retrieved: false`) — the correct precedent never made it past vector search. Look at `vector_score_threshold` and `extracted_queries` to understand why.

2. **Judge ranking failure** (`retrieved: true` but `pipeline_rank` is high or `judge_score` is low) — the precedent was retrieved but the judge scored it poorly. Read `candidates_text_sent` to see if the truncated fields lost important context, and read `judge_raw_response` to understand the judge's reasoning.

3. **Borderline cases** — `retrieval_rank` is close to the `candidates` cutoff. Could be fixed by raising `candidates` or lowering `vector_score_threshold`.

Always look for **patterns across multiple petitions** — a single fix that helps several cases is more valuable than one-off tweaks.
