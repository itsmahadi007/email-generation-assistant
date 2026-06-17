"""The three custom evaluation metrics.

Each metric returns a dict ``{"score": float in [0,1], "detail": {...}}`` and is
registered in ``METRICS`` with a human-readable definition + logic so the report
can document them automatically.

Technique mix (the brief asks for a *combination* of automated techniques):
  - Metric 1 (Intent & Fact Coverage) ... LLM-as-judge
  - Metric 2 (Tone Accuracy) ............. LLM-as-judge
  - Metric 3 (Conciseness & Format) ...... deterministic Python (no LLM)
"""
from __future__ import annotations

import json
import re

import textstat

from . import config, llm


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _judge_json(instruction: str) -> dict:
    """Call the judge model and parse the first JSON object from its reply."""
    raw = llm.chat(
        model=config.JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a meticulous, impartial evaluator of professional "
                    "email. Respond with a single JSON object and nothing else."
                ),
            },
            {"role": "user", "content": instruction},
        ],
        temperature=0.0,
        max_tokens=config.JUDGE_MAX_TOKENS,
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise llm.LLMError(f"Judge did not return JSON: {raw[:200]!r}")
    return json.loads(match.group(0))


# --- Metric 1: Intent & Fact Coverage (LLM-as-judge) -----------------------
def metric_fact_intent_coverage(email: str, scenario: dict) -> dict:
    facts = scenario["key_facts"]
    facts_block = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    instruction = f"""\
Evaluate whether the EMAIL below accomplishes its INTENT and includes every KEY FACT.

INTENT: {scenario['intent']}

KEY FACTS:
{facts_block}

EMAIL:
\"\"\"
{email}
\"\"\"

Return JSON exactly in this shape:
{{
  "facts": [{{"fact": "<the fact>", "included": true|false}}, ...one per key fact...],
  "intent_accomplished": true|false,
  "hallucinated_claims": <integer count of material facts/names/dates/numbers
                          asserted in the email that were NOT in the intent or key facts>
}}
A fact counts as "included" if its meaning is clearly conveyed, even if paraphrased."""
    data = _judge_json(instruction)

    fact_flags = [bool(f.get("included")) for f in data.get("facts", [])]
    fact_recall = sum(fact_flags) / len(facts) if facts else 1.0
    intent = 1.0 if data.get("intent_accomplished") else 0.0
    halluc = int(data.get("hallucinated_claims", 0) or 0)
    penalty = min(0.30, 0.10 * halluc)
    score = _clamp(0.70 * fact_recall + 0.30 * intent - penalty)
    return {
        "score": round(score, 4),
        "detail": {
            "fact_recall": round(fact_recall, 4),
            "facts_included": sum(fact_flags),
            "facts_total": len(facts),
            "intent_accomplished": bool(intent),
            "hallucinated_claims": halluc,
        },
    }


# --- Metric 2: Tone Accuracy (LLM-as-judge) --------------------------------
def metric_tone_accuracy(email: str, scenario: dict) -> dict:
    instruction = f"""\
Rate how well the EMAIL's tone matches the REQUESTED TONE on a 1-5 scale.

REQUESTED TONE: {scenario['tone']}

EMAIL:
\"\"\"
{email}
\"\"\"

Scale: 5 = tone matches precisely; 4 = matches with minor slips;
3 = partially matches; 2 = largely the wrong register; 1 = wrong tone entirely.

Return JSON: {{"rating": <1-5 integer>, "rationale": "<one sentence>"}}"""
    data = _judge_json(instruction)
    rating = int(data.get("rating", 1) or 1)
    rating = max(1, min(5, rating))
    return {
        "score": round((rating - 1) / 4.0, 4),
        "detail": {"rating": rating, "rationale": data.get("rationale", "")},
    }


# --- Metric 3: Conciseness & Format (deterministic) ------------------------
_GREETING_RE = re.compile(r"^\s*(hi|hello|dear|hey|greetings|good (morning|afternoon|evening))\b", re.I | re.M)
_SIGNOFF_RE = re.compile(r"\b(best|regards|sincerely|thanks|thank you|cheers|warm regards|kind regards|best regards)\b", re.I)
_SUBJECT_RE = re.compile(r"^\s*subject\s*:", re.I | re.M)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def metric_conciseness_format(email: str, scenario: dict) -> dict:
    ref = scenario["reference_email"]
    gen_words = _word_count(email)
    ref_words = _word_count(ref) or 1
    ratio = gen_words / ref_words
    # 1.0 at length parity, falling to 0 by the time the email is 2x (or 0x) the
    # reference length — penalises both bloat and under-writing.
    length_score = _clamp(1.0 - abs(ratio - 1.0))

    has_subject = bool(_SUBJECT_RE.search(email))
    has_greeting = bool(_GREETING_RE.search(email))
    has_signoff = bool(_SIGNOFF_RE.search(email))
    structure_score = (has_subject + has_greeting + has_signoff) / 3.0

    try:
        flesch = textstat.flesch_reading_ease(email)
    except Exception:  # noqa: BLE001 - textstat can choke on tiny inputs
        flesch = 60.0
    # Reward standard business readability (~60 Flesch); fall off either side.
    readability_score = _clamp(1.0 - abs(flesch - 60.0) / 60.0)

    score = 0.5 * length_score + 0.3 * structure_score + 0.2 * readability_score
    return {
        "score": round(score, 4),
        "detail": {
            "gen_words": gen_words,
            "ref_words": ref_words,
            "length_ratio": round(ratio, 3),
            "length_score": round(length_score, 4),
            "has_subject": has_subject,
            "has_greeting": has_greeting,
            "has_signoff": has_signoff,
            "structure_score": round(structure_score, 4),
            "flesch_reading_ease": round(flesch, 1),
            "readability_score": round(readability_score, 4),
        },
    }


# --- Registry (drives both scoring and the auto-generated report) ----------
METRICS = [
    {
        "key": "fact_intent_coverage",
        "name": "Intent & Fact Coverage",
        "focus": "Fact Recall",
        "technique": "LLM-as-judge",
        "fn": metric_fact_intent_coverage,
        "definition": (
            "Does the email accomplish the stated intent and include every key "
            "fact, without inventing new ones?"
        ),
        "logic": (
            "An impartial judge model marks each key fact present/absent "
            "(semantic match, paraphrase allowed), judges whether the intent is "
            "accomplished, and counts hallucinated claims. "
            "score = 0.70*fact_recall + 0.30*intent_accomplished "
            "- min(0.30, 0.10*hallucinated_claims), clamped to [0,1]."
        ),
    },
    {
        "key": "tone_accuracy",
        "name": "Tone Accuracy",
        "focus": "Tone Accuracy",
        "technique": "LLM-as-judge",
        "fn": metric_tone_accuracy,
        "definition": "How precisely does the email's register match the requested tone?",
        "logic": (
            "The judge rates tone match on a 1-5 rubric (5 = precise match, "
            "1 = wrong tone). score = (rating - 1) / 4, mapping to [0,1]."
        ),
    },
    {
        "key": "conciseness_format",
        "name": "Conciseness & Format",
        "focus": "Conciseness / Format Adherence",
        "technique": "Deterministic Python",
        "fn": metric_conciseness_format,
        "definition": (
            "Is the email appropriately concise (vs the human reference) and "
            "well-formed (subject, greeting, sign-off, readable prose)?"
        ),
        "logic": (
            "Pure Python, no LLM. length_score = 1 - |gen_words/ref_words - 1| "
            "(penalises bloat and under-writing). structure_score = fraction of "
            "{subject line, greeting, sign-off} present (regex). readability_score "
            "= 1 - |Flesch - 60|/60 (textstat). "
            "score = 0.5*length + 0.3*structure + 0.2*readability."
        ),
    },
]

METRIC_KEYS = [m["key"] for m in METRICS]


def score_email(email: str, scenario: dict) -> dict:
    """Run all three metrics on one email; return per-metric results + composite."""
    results = {}
    for m in METRICS:
        results[m["key"]] = m["fn"](email, scenario)
    composite = sum(results[k]["score"] for k in METRIC_KEYS) / len(METRIC_KEYS)
    results["composite"] = round(composite, 4)
    return results
