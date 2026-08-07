# PulseAI

An AI-powered feedback intelligence system for e-commerce reviews. Given raw
customer reviews, PulseAI classifies feedback into categories, scores
sentiment and urgency, aggregates recurring themes, and answers natural
language questions - through counts, breakdowns, generated charts, and
narrative summaries - via a conversational interface.

All classification and query-parsing logic lives in `classifier.py`. The
Dash app, the Celery tasks, and the ingestion scripts are all thin callers
of that one module. None of them re-implement prompt construction or LLM
calls independently.

## Table of contents

- [Deliverables](#deliverables)
- [What it does](#what-it-does)
- [Quick start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Architecture](#architecture)
  - [Why few-shot, not RAG](#why-few-shot-not-rag)
  - [Why a two-table schema, not one](#why-a-two-table-schema-not-one)
  - [Why structured filter-parsing, not raw LLM-generated SQL](#why-structured-filter-parsing-not-raw-llm-generated-sql)
- [Known limitations](#known-limitations)
- [A real infrastructure bug, found and fixed](#a-real-infrastructure-bug-found-and-fixed)
- [Project structure](#project-structure)

## Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | Design prompts that produce structured outputs (JSON) | Done. Tool-use schema plus Pydantic validation in `classifier.py` |
| 2 | Build a classifier with few-shot examples | Done. 7 deliberately chosen examples, multi-turn conversational format |
| 3 | Score sentiment reliably | Done. Same schema-enforced call as classification, tested at 1,000-review scale |
| 4 | Aggregate themes across multiple inputs | Done. `aggregation.py`, breakdown queries grouped by any real column |
| 5 | Generate a narrative summary using AI | Done. `generate_summary()`, with rolling per-window storage and reuse |

## What it does

A customer review comes in as plain text. PulseAI:

1. Classifies it via a few-shot LLM call (`classify_review`): category,
   sentiment, urgency, and a one-sentence reasoning citing the specific
   part of the review that drove the decision.
2. Persists it to Postgres in a normalized two-table schema (`reviews` for
   raw source data, `classifications` for AI output, joined by
   `review_id`).
3. Processes reviews asynchronously via Celery and Redis, so classification
   doesn't block the triggering request.
4. Powers a conversational query interface: ask a question in plain
   English, get back a count, a lookup, a grouped breakdown, a generated
   chart, or a narrative summary, grounded in the real database schema and
   never guessing at filter values that don't exist.

## Quick start

### Prerequisites

- Python 3.12+
- PostgreSQL installed locally
- Redis installed locally
- An OpenAI API key

### Setup

```bash
git clone https://github.com/Amith0707/PulseAI.git
cd PulseAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pulseai
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password
```

Create the database and initialize tables:

```bash
createdb pulseai
python3 -c "from db import init_db; init_db()"
```

Build the demo dataset. This samples real Amazon reviews from a real,
density-verified 6-month window, deliberately mixing rating levels for
complaint variety:

```bash
python3 -m sample_builder
python3 -m ingest_reviews
```

Start the Celery worker in its own terminal. This must use the `threads`
pool, not the default `prefork`, to avoid a known psycopg2/fork crash on
macOS (see "A real infrastructure bug" below):

```bash
celery -A tasks worker --loglevel=info --pool=threads --concurrency=4
```

In a second terminal, classify the demo sample:

```python
import pandas as pd
from tasks import classify_review_task

sample = pd.read_parquet("data/demo_sample.parquet")
for _, row in sample.iterrows():
    classify_review_task.delay(str(row["review_id"]), row["text"])
```

Start the app:

```bash
python3 -m dashboard
```

Visit `http://127.0.0.1:8050`.

## Architecture

```
Raw reviews (Parquet, sampled from a real Amazon dataset)
        |
        v
Ingested into Postgres "reviews" table (raw data, no AI)
        |
        v
Celery task queue (Redis broker) -> few-shot LLM classification
        |
        v
Postgres "classifications" table (AI output, foreign-keyed to reviews)
        |
        v
Conversational query layer: natural language -> schema-grounded intent
parsing -> safe parameterized SQL -> count / breakdown / chart / summary
        |
        v
Dash UI: analytics dashboard plus chat interface with multi-turn memory
```

### Why few-shot, not RAG

Unlike an earlier project (ClaimPilot) which used RAG for a larger,
frequently changing policy manual, PulseAI's taxonomy is small and stable
enough to include directly in the prompt. There is no real retrieval
problem to solve here, so reaching for RAG would be unnecessary complexity
for the same reason it was worth using on the earlier project.

### Why a two-table schema, not one

`reviews` and `classifications` are separate tables, joined by
`review_id`, rather than one combined table. This means raw source data
can be ingested independently of classification. A review can exist in
`reviews` without yet having a row in `classifications`, reflecting the
real state of an async pipeline where ingestion and processing happen at
different times.

### Why structured filter-parsing, not raw LLM-generated SQL

The conversational query layer never lets the LLM write or execute raw SQL
directly. Instead, the LLM converts a question into structured parameters
(category, sentiment, date range, chart type), and the application code
builds parameterized SQL from those parameters. This is the same defense
against injection risk as a hand-written form, applied to a
natural-language interface instead.

## Known limitations

- The demo dataset is a fixed, deliberately sampled window (January to
  June 2022), not continuously arriving data. Date-range and "rolling"
  queries are validated live against the database's actual coverage,
  never hardcoded, so asking about a period outside this window correctly
  returns "no data available" rather than a hallucinated answer.
- The query parser handles one filter value per field per question. Asking
  for a breakdown across multiple values of the same dimension in one
  question (for example, "compare positive, negative, and neutral
  counts") is not yet supported. Ask as separate questions instead.
- Full Docker containerization was scoped out due to time. The local setup
  above is complete and tested. Containerizing would follow the same
  pattern as a prior project (ClaimPilot), with the added complexity of a
  correctly configured Celery worker container that must use
  `--pool=threads`.
- Production-scale infrastructure (connection pooling, sharding, load
  balancing, full monitoring) was deliberately not built. At this
  project's actual data volume (about 1,000 reviews), none of it would be
  meaningfully exercised. The architecture is stateless and horizontally
  scalable in principle, since each classification task is independent,
  so scaling would mean adding more Celery workers and a proper connection
  pool, not redesigning the pipeline.

## A real infrastructure bug, found and fixed

Celery's default `prefork` worker pool crashes with a segmentation fault
(SIGSEGV) on macOS when a task writes to Postgres via `psycopg2`. Forking
a process midway through a live database connection corrupts that
connection's internal state. This was diagnosed by reading the exact
sequence of worker logs (the crash occurred immediately after a
successful classification, right before the database write) and fixed by
switching to `--pool=threads`, which avoids forking entirely while
preserving real concurrency. Full write-up in `documentation.md`.

## Project structure

```
PulseAI/
+-- app files
|   +-- dashboard.py         Dash app: analytics dashboard and chat UI
|   +-- classifier.py        Core AI logic: classification, query parsing, summaries
|   +-- tasks.py             Celery task definitions
|   +-- celery_app.py        Celery app configuration
|   +-- db.py                Postgres connection and persistence
|   +-- aggregation.py       Aggregation logic and rolling summaries
|   +-- schema_info.py       Live schema description and date-range validation
|   +-- sample_builder.py    Builds the deliberate demo sample from raw data
|   +-- ingest_reviews.py    Loads the demo sample into Postgres
+-- utils/
|   +-- logger.py            Daily-rotated logging
|   +-- const.py             Prompts, few-shot examples, tool schemas
+-- data/                    Local Parquet files (not tracked in git)
+-- assets/
|   +-- style.css            UI styling
+-- requirements.txt
```
