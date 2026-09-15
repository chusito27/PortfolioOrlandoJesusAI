# Orlando Jesús Calvo Vargas

AI Engineer. Ten years of production software engineering, currently Full Stack AI Engineer on a
direct-to-consumer commerce platform running across 50+ countries.

This repository is my portfolio: the site that presents the work, and the source of every project
it claims. **Every number on the site is reproducible from the code in this repo.**

Costa Rica, remote. UTC-6, full US workday overlap. English C1 certified.
[LinkedIn](https://www.linkedin.com/in/orlando-calvo-vargas-207570015/) ·
[GitHub](https://github.com/chusito27)

---

## Projects

### [Hybrid RAG over technical documentation](projects/dotnet-docs-rag)

A retrieval-augmented generation service over the official .NET documentation whose evaluation
harness is a first-class component rather than an afterthought. Hybrid retrieval over pgvector and
Postgres full-text, fused with weighted Reciprocal Rank Fusion and re-ranked by a cross-encoder.
Answers only from passages it can cite, and refuses when the corpus cannot answer.

| Metric | Result |
|---|---|
| Retrieval recall@5 | 100% |
| Citation grounding | 100% |
| Refusal rate on unanswerable questions | 100% |
| Corpus | 2,657 chunks across 310 documents |

The finding I am most pleased with is a negative one: textbook unweighted RRF regressed precision
from 100% to 89% recall@1, because the lexical arm reaches only 28% alone and its weak candidates
were displacing the correct passage from first place. That is documented rather than hidden, along
with the limitations of tuning on an 18-case golden set.

`Python` `FastAPI` `PostgreSQL` `pgvector` `RAG` `Embeddings` `Hugging Face` `Docker` `LLM evaluation`

---

## Layout

```
projects/          one directory per project, each with its own README and results
  dotnet-docs-rag/
site/              the portfolio site: React, Vite, deployed on Vercel
```

## Running the site

```bash
cd site
npm install
npm run dev
```
