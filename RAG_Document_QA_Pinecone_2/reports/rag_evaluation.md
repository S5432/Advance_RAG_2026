# RAG Evaluation Report

> Status: the evaluation set and scoring script are ready (`reports/eval_set.json`, `scripts/run_eval.py`).
> The "Answer / Citation correct?" columns must be filled from a real run
> (`python scripts/run_eval.py`) because they need live Pinecone + Gemini keys.
> Expected answers below come from the sample documents in `data/`.

## Evaluation table (documents: sample_policy.pdf, sample_faq.csv, sample_notes.md)

| # | Question | Expected answer | Retrieved chunks | Answer correct? | Citation correct? | Issue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | How many annual leave days are allowed? | 18 days | 5 | _run_ | _run_ | |
| 2 | How many annual leave days can be carried forward? | Up to 5 days | 5 | _run_ | _run_ | |
| 3 | How many days of sick leave per year? | 12 days | 5 | _run_ | _run_ | |
| 4 | How many days can employees work remotely? | Up to 2 days/week with approval | 5 | _run_ | _run_ | |
| 5 | What are the core hours for remote employees? | 11:00 AM – 4:00 PM | 5 | _run_ | _run_ | |
| 6 | Within how many days must expense claims be submitted? | 30 days | 5 | _run_ | _run_ | |
| 7 | What is the notice period for resignation? | 60 days (CSV FAQ) | 5 | _run_ | _run_ | |
| 8 | How long is the probation period? | 3 months (Markdown) | 5 | _run_ | _run_ | |
| 9 | What is the annual learning budget? | 25000 INR | 5 | _run_ | _run_ | |
| 10 | What is the CEO's salary? (not in documents) | "I could not find this information…" | 5 | _run_ | n/a | Tests hallucination refusal |

Likely weak spots to check: Q2 (carry-forward sits in the same chunk as Q1 — verify the right sentence
is cited), Q7 (CSV rows are tiny chunks and compete with policy text), Q10 (the model must refuse even
though 5 chunks are always returned).

## Analysis questions

1. **Which questions produced incorrect answers?** Read from the run; typically multi-fact questions (Q2) and
   questions whose answer lives in a short CSV row (Q7). Log each failure with the retrieved chunk IDs.
2. **Irrelevant chunks?** Top-k always returns 5 results, even for Q10, so some will be irrelevant. Inspect `/search`
   scores; add a minimum-score cut-off (e.g. 0.3 for MiniLM cosine) once you see the score distribution.
3. **Hallucination on insufficient context?** Q10 checks this. Mitigations already present: "answer only from context"
   rule, exact refusal sentence, temperature 0.1, no-chunks short-circuit.
4. **Chunk size/overlap improvements.** Tune 500–1000 chars and 10–20 % overlap against precision@5; keep headings with
   their body; use per-type sizes (small for FAQ rows, larger for prose); try parent-document retrieval.
5. **Metadata filtering for multi-tenant security.** Every vector carries `tenant_id`; every query includes
   `filter={"tenant_id": ...}`, so another tenant's vectors are never candidates.
6. **Preventing cross-tenant access.** Derive tenant from a verified JWT (not a client header); inject the filter
   server-side and set it after user-supplied filters; tenant-scoped document IDs; return 404 not 403; for strict
   isolation use one Pinecone namespace per tenant; add an automated test (included: `test_tenant_isolation`).
7. **Confidential documents.** Encrypt in transit/at rest, store secrets in a secret manager, add per-document ACL
   metadata (`allowed_roles`) filtered at query time, redact PII before embedding, avoid logging chunk text, keep a
   delete/retention path (delete vectors by ID), and check the LLM provider's data-use terms.
8. **OCR quality checks.** Already stored: mean Tesseract confidence per page and `low_ocr_quality` (< 60). Add: reject or
   flag documents with many low pages, deskew/denoise before OCR, language packs, a word-validity ratio check, and
   manual review queue for flagged pages.
9. **precision@5 / recall@5.** Label the relevant chunks per question. precision@5 = relevant chunks in top 5 ÷ 5;
   recall@5 = relevant chunks in top 5 ÷ total relevant chunks. `scripts/run_eval.py` computes both using a
   substring marker for the gold chunk (with one gold chunk per question, recall@5 is the hit rate). Also track MRR.
10. **Docker Compose deployment.** `Dockerfile` (python-slim + tesseract + poppler, model baked in) and
    `docker-compose.yml` (env_file, restart policy, healthcheck on `/health`). Put it behind a reverse proxy with TLS,
    run multiple workers only after moving the status registry to Redis.
