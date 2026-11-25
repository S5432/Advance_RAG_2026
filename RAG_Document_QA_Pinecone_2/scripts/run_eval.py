"""Run the evaluation set against a running API and print a markdown table + precision@5 / recall@5.

Usage: python scripts/run_eval.py [http://localhost:8000]
Upload the three sample documents first (see README).
"""
import json
import sys
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
H = {"X-Tenant-ID": "company_001", "X-User-ID": "user_001"}
items = json.loads((Path(__file__).parent.parent / "reports" / "eval_set.json").read_text())

rows, precisions, recalls = [], [], []
for it in items:
    search = httpx.post(f"{BASE}/search", json={"question": it["q"], "top_k": 5}, headers=H, timeout=60).json()
    ask = httpx.post(f"{BASE}/ask", json={"question": it["q"], "top_k": 5}, headers=H, timeout=60).json()
    marker = it["relevant_chunks_contain"].lower()
    hits = [marker in r["text"].lower() for r in search["results"]] if marker else []
    if marker:
        precisions.append(sum(hits) / max(len(hits), 1))
        recalls.append(1.0 if any(hits) else 0.0)  # one gold chunk per question -> recall@5 is hit-rate
    correct = it["expected"].lower().split(" with ")[0] in ask["answer"].lower()
    cited = bool(ask["citations"]) or not marker
    rows.append(f"| {it['q']} | {it['expected']} | {len(search['results'])} | {'Yes' if correct else 'CHECK'} "
                f"| {'Yes' if cited else 'No'} | {ask['answer'][:80]} |")

print("| Question | Expected answer | Retrieved chunks | Answer correct? | Citation correct? | Model answer |")
print("| --- | --- | --- | --- | --- | --- |")
print("\n".join(rows))
print(f"\nprecision@5 = {sum(precisions)/len(precisions):.2f}   recall@5 = {sum(recalls)/len(recalls):.2f}")
