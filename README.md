# Email Generation Assistant

An LLM-powered assistant that turns an **intent**, a few **key facts**, and a
desired **tone** into a polished, ready-to-send professional email — plus a
custom evaluation harness that scores the output on three bespoke metrics and
compares two models head-to-head.

Built for the SG Services AI Engineer assessment.

---

## What it does

1. **Generates email** with an advanced prompt (role-play + few-shot +
   chain-of-thought with delimited output — see `src/prompt.py`).
2. **Evaluates** every email on **three custom metrics** (`src/metrics.py`):
   | Metric | Technique | What it measures |
   |---|---|---|
   | Intent & Fact Coverage | LLM-as-judge | Are the intent and *all* key facts present, with no hallucinations? |
   | Tone Accuracy | LLM-as-judge | Does the register match the requested tone (1–5 rubric)? |
   | Conciseness & Format | Pure Python | Length vs. the human reference, structure (subject/greeting/sign-off), readability |
3. **Compares two models** on the same 10 scenarios and writes a full report
   with the comparative analysis.

## Models (via OpenRouter)

One OpenRouter key calls every model. Defaults (override in `.env`):

| Role | Default | Why |
|---|---|---|
| Model A | `openai/gpt-5.4` | Frontier model |
| Model B | `openai/gpt-5.4-mini` | Efficient sibling — same family, so the comparison isolates capability |
| Judge | `anthropic/claude-sonnet-4.6` | A *different* family, so neither model under test grades its own work |

Swap any of them to compare a Claude pair, cross-provider (`anthropic/claude-opus-4.8` vs `openai/gpt-5.4`), etc. Browse slugs at <https://openrouter.ai/models>.

---

## Setup

```bash
cd email-generation-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your OpenRouter key (https://openrouter.ai/keys)
```

## Run

```bash
python -m src.evaluate            # generate + score 10 scenarios × 2 models
python -m src.evaluate --fresh    # ignore cache, regenerate every email
```

Generated emails are cached in `results/generations_cache.json`, so re-runs only
re-score (cheap) unless you pass `--fresh`.

## Outputs

| File | Contents |
|---|---|
| `report/REPORT.md` | **The deliverable report** — prompt template, metric definitions + logic, raw data tables, and the comparative analysis. Export to PDF/Google Doc. |
| `results/results.json` | Structured: config, metric defs, every raw score + per-fact detail, per-model averages |
| `results/results.csv` | Flat scores table (scenario × model × metric) |
| `results/emails.md` | Every generated email, for human inspection |

## Project layout

```
src/
  config.py     env + paths + model selection
  llm.py        OpenRouter (OpenAI-compatible) client wrapper, with retries
  prompt.py     the advanced prompt template + email parsing
  generate.py   scenario -> email
  metrics.py    the 3 custom metrics + registry
  evaluate.py   run everything, write all outputs + report
data/
  scenarios.json   10 scenarios, each with a human reference email
```

## Test data

`data/scenarios.json` holds 10 diverse scenarios (follow-ups, apologies, urgent
escalations, cold outreach, price-increase announcements, …), each with an
`intent`, `key_facts`, `tone`, and a hand-written **human reference email** used
as the ideal for the Conciseness & Format metric.
