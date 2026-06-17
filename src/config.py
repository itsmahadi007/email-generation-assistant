"""Central configuration: paths, models, and the OpenRouter API key.

Everything that varies between runs (which models, where files live) is read
from the environment here so the rest of the code stays declarative.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
CACHE_PATH = RESULTS_DIR / "generations_cache.json"

for _d in (RESULTS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- API -------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Email Generation Assistant Eval")

# --- Models under test (Section 3 comparison) ------------------------------
MODEL_A = os.getenv("MODEL_A", "openai/gpt-5.4")
MODEL_B = os.getenv("MODEL_B", "openai/gpt-5.4-mini")
MODELS = [MODEL_A, MODEL_B]

# Independent judge for the LLM-as-judge metrics. A different family than the
# models under test, so neither model is grading its own homework.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4.6")

# --- Generation defaults ---------------------------------------------------
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.4"))
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "1200"))
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "800"))
