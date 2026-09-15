# dotnet-docs-rag

A hybrid retrieval-augmented generation service over the official .NET documentation,
with an evaluation harness as a first-class component. Runs entirely on local models:
no API keys, no per-query cost, no data leaving the machine.

Built to answer a narrow question honestly: *when a developer asks something about .NET,
does the system find the right passage and answer from it, or does it make something up?*

## Measured results

310 documents, 2,657 chunks. Golden set of 22 cases: 18 answerable, 4 deliberately not.

| Metric | Result | What it means |
|---|---|---|
| `recall@5` | **100%** | The correct document reached the model on every answerable question |
| `grounding_rate` | **100%** | Every answer cited passages that were actually supplied |
| `refusal_rate` | **100%** | All 4 unanswerable questions returned `NOT_IN_CONTEXT` instead of an invention |

Retrieval ablation, same questions, retrieval only:

| Strategy | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|
| vector only | **100%** | 100% | 100% | 100% |
| keyword only | 28% | 39% | 39% | 39% |
| hybrid (weighted RRF) | 94% | 100% | 100% | 100% |
| hybrid + LLM rerank | **100%** | 100% | - | - |

### What the ablation actually showed, including the part that went against the design

The interesting result is a negative one. **Textbook unweighted RRF made retrieval worse.**
Vector search alone reaches 100% recall@1 on this corpus. Fusing it with the lexical arm at
equal weight dropped that to 89%: the lexical retriever only manages 28% recall@1 on its own,
and its mediocre candidates were displacing the correct passage from first place.

Sweeping the fusion weight recovered precision to 94% at 0.2 and to 100% only at 0, which is
vector-only by another name. The LLM re-ranking pass independently recovers 100% regardless of
the weight, because reordering a good candidate set is an easier problem than ranking it.

Three things follow, and they are the reason the harness exists:

1. **On this corpus, hybrid search does not earn its complexity on recall.** The embedding
   model handles .NET documentation vocabulary well enough that the lexical arm adds nothing.
2. **RRF is not free.** Giving a weak retriever an equal vote is a measurable regression, not a
   neutral safety net. The weight is a parameter, not a detail.
3. **The lexical arm is kept at a low weight anyway**, because the failure mode it protects
   against (exact tokens: SKUs, error codes, CLI flags, version strings) is one this particular
   golden set does not stress. That is a judgement call about unseen queries, stated as such,
   not a result.

**Limitations, stated plainly.** The golden set is 18 answerable cases, small enough that the
0.2 weight is at real risk of overfitting. The questions were written close to the source
document titles, which favours semantic retrieval and is very likely why vector-only saturates.
A larger, more adversarial set, and a corpus with heavier exact-token traffic, would probably
move these numbers. The weight should be re-swept against any change of embedding model.

Reproduce with `python eval/run_eval.py --ablate` and `python eval/sweep.py`.

## Design

**Hybrid retrieval.** Vector search is strong on meaning and weak on exact tokens. Full-text
search is the mirror image. Both indexes live in the same PostgreSQL instance, which keeps the
operational surface to one database: HNSW over `pgvector` for semantics, GIN over `tsvector` for
exact terms, fused with weighted Reciprocal Rank Fusion. RRF combines *rankings* rather than
scores, which matters because a cosine distance and a `ts_rank_cd` score live on incomparable
scales.

**Re-ranking.** Retrieval optimises recall, getting the answer somewhere in the top 30.
Re-ranking optimises precision, putting it in the top 5 that reach the prompt. A malformed
re-rank response degrades to the fused order rather than failing the request.

**Enforced citations.** The model may answer only from the supplied passages, must cite a
passage id per claim, and must return `NOT_IN_CONTEXT` when the corpus cannot answer. Cited ids
are validated against what was actually supplied, so a fabricated citation is caught
mechanically rather than trusted.

**Heading-aware chunking.** Splits on heading boundaries, keeps fenced code blocks intact, and
stamps each chunk with its heading path before embedding, because a chunk reading "Use
AddScoped" is ambiguous alone and unambiguous under "Dependency injection > Service lifetimes".
Falls back to line and then character boundaries so a large markdown table or a long code fence
cannot blow past the context budget, with overlap snapped to word boundaries.

```
 .NET docs (markdown)
        |
 heading-aware chunking
        |
 embeddings
        |
 PostgreSQL + pgvector
   |-- HNSW on vector(768)   -> semantic recall
   +-- GIN on tsvector       -> exact-token recall
        |
 weighted Reciprocal Rank Fusion
        |
 LLM re-ranking (top 8 -> top 5)
        |
 answer with validated inline citations
        |
 evaluation harness -> recall@k, grounding, refusal
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | PostgreSQL 17 + pgvector | One database for both halves of the hybrid search |
| Embeddings | `nomic-embed-text` via Ollama | 768 dimensions, local, free |
| Generation | `qwen2.5:14b` via Ollama | Strong instruction following, runs locally |
| API | FastAPI | Async throughout; retrieval is IO-bound |
| Language | Python 3.12 | |

The provider lives behind `app/llm.py`. Moving to OpenAI, Anthropic or Bedrock means rewriting
that one file; nothing above it knows which provider is in use.

## Running it

```bash
docker compose up -d

py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull qwen2.5:14b

.venv/Scripts/python scripts/fetch_docs.py --limit 320
.venv/Scripts/python scripts/ingest.py --reset

.venv/Scripts/python -m uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/health

curl -s localhost:8000/ask -H "content-type: application/json" \
  -d '{"question":"What does the dotnet clean command do?"}'
```

`/search` returns retrieval results without generating an answer, which is how you tell a bad
answer caused by bad retrieval from one caused by the model.

## Two bugs worth recording

**A chicken-and-egg deadlock on first run.** The connection pool registered the `pgvector` type
on every new connection, but `register_vector` cannot look the type up until the extension
exists, and on a fresh database it does not. No connection could open, so the schema that
creates the extension could never run. Fixed by creating the extension once on a standalone
connection before the pool opens.

**A race that the obvious fix introduced.** Moving `CREATE EXTENSION IF NOT EXISTS` into the
pool's configure hook made several connections run it simultaneously and collide on
`pg_extension_name_index`. `IF NOT EXISTS` is not a concurrency guarantee.

## Layout

```
app/
  config.py      settings; every retrieval knob in one place
  db.py          connection pool and schema
  llm.py         Ollama client, the only provider-aware file
  chunking.py    heading-aware markdown splitting
  retrieval.py   hybrid search, weighted RRF, LLM re-ranking
  generate.py    answer generation with citation validation
  main.py        FastAPI
scripts/
  fetch_docs.py  corpus download
  ingest.py      ingestion pipeline
eval/
  golden.yaml    the golden set
  run_eval.py    full-pipeline metrics and ablation
  sweep.py       retrieval-only recall across strategies and k
```

## Licence

Code under MIT. The ingested documentation belongs to Microsoft under CC BY 4.0.
