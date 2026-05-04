"""Quick script to summarize harness runs stored in Langfuse."""
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from collections import defaultdict
from datetime import datetime, timezone, timedelta

lf = Langfuse()

# Only fetch traces from the last 24 hours to avoid timeout
since = datetime.now(timezone.utc) - timedelta(hours=24)
page = lf.api.trace.list(limit=100, page=1, from_timestamp=since)
harness = [t for t in page.data if t.name and t.name.startswith("harness/")]
print(f"Harness traces (last 24h): {len(harness)}")

runs = defaultdict(list)
for t in harness:
    full = lf.api.trace.get(t.id)
    meta = full.metadata or {}
    run = meta.get("run_name", "unknown")
    providers = meta.get("providers", {})
    scores = {s.name: s.value for s in (full.scores or [])}
    expected_id = full.input.get("expected_id") if full.input else "?"
    runs[run].append({
        "expected_id": expected_id,
        "hit_at_1": scores.get("hit_at_1"),
        "reciprocal_rank": scores.get("reciprocal_rank"),
        "retrieved": scores.get("retrieved"),
        "judge_score": scores.get("judge_score"),
        "providers": providers,
    })

for run, petitions in sorted(runs.items()):
    n = len(petitions)
    if n < 10:
        continue
    p = petitions[0]["providers"]
    hit1 = sum(x["hit_at_1"] or 0 for x in petitions) / n
    mrr = sum(x["reciprocal_rank"] or 0 for x in petitions) / n
    recall = sum(x["retrieved"] or 0 for x in petitions) / n
    print(f"\n=== {run} ({n} petitions) ===")
    print(f"  llm={p.get('llm')}  query={p.get('query_model_id')}  judge={p.get('judge_model_id')}  emb={p.get('embedding')}({p.get('embedding_model_id')})")
    print(f"  hit@1={hit1:.1%}  MRR={mrr:.4f}  recall={recall:.1%}")
    for x in sorted(petitions, key=lambda x: x["expected_id"]):
        print(f"    {x['expected_id']:<22} hit@1={int(x['hit_at_1'] or 0)}  rr={x['reciprocal_rank']:.2f}  score={x['judge_score']}")
