"""Run the full evaluation and write all deliverables.

    python -m src.evaluate            # uses cached generations if present
    python -m src.evaluate --fresh    # regenerate every email

Writes:
    results/results.json   structured: config, metric defs, raw scores, averages
    results/results.csv    flat raw scores table
    results/emails.md      every generated email (for human inspection)
    report/REPORT.md       final report (prompt, metric defs, raw data, analysis)
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request

from . import config, generate, metrics, prompt


# --------------------------------------------------------------------------
# Pricing (so the recommendation can weigh quality against cost)
# --------------------------------------------------------------------------
def _fetch_pricing(models: list[str]) -> dict:
    """Fetch $/token pricing for the given models from OpenRouter's public
    models endpoint. Returns ``{model: {prompt, completion, per_mtok_*}}`` or an
    empty dict if the lookup fails (the report then stays cost-agnostic).
    """
    try:
        req = urllib.request.Request(
            f"{config.OPENROUTER_BASE_URL}/models",
            headers={"User-Agent": "email-assistant-eval"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            catalog = json.load(resp)["data"]
    except Exception:  # noqa: BLE001 - pricing is a nice-to-have, never fatal
        return {}

    by_id = {m["id"]: m for m in catalog}
    out = {}
    for model in models:
        m = by_id.get(model)
        if not m:
            continue
        try:
            prompt_p = float(m["pricing"]["prompt"])
            completion_p = float(m["pricing"]["completion"])
        except (KeyError, TypeError, ValueError):
            continue
        out[model] = {
            "prompt_per_token": prompt_p,
            "completion_per_token": completion_p,
            "prompt_per_mtok": round(prompt_p * 1_000_000, 4),
            "completion_per_mtok": round(completion_p * 1_000_000, 4),
            # Blended rank weighted toward output (email gen is output-heavy).
            "blended_rank": prompt_p * 0.4 + completion_p * 0.6,
        }
    return out


# --------------------------------------------------------------------------
# Generation cache (the expensive, non-deterministic step)
# --------------------------------------------------------------------------
def _load_cache() -> dict:
    if config.CACHE_PATH.exists():
        return json.loads(config.CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    config.CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _gen_email(model: str, scenario: dict, cache: dict, fresh: bool) -> dict:
    key = f"{model}::{scenario['id']}"
    if not fresh and key in cache:
        return cache[key]
    out = generate.generate_email(model, scenario)
    cache[key] = out
    _save_cache(cache)
    return out


# --------------------------------------------------------------------------
# Core run
# --------------------------------------------------------------------------
def run(fresh: bool = False) -> dict:
    scenarios = json.loads(config.SCENARIOS_PATH.read_text())
    cache = _load_cache()
    runs: list[dict] = []

    for model in config.MODELS:
        for scenario in scenarios:
            label = f"[{model}] {scenario['id']}"
            try:
                gen = _gen_email(model, scenario, cache, fresh)
                scores = metrics.score_email(gen["email"], scenario)
                runs.append(
                    {
                        "model": model,
                        "scenario_id": scenario["id"],
                        "intent": scenario["intent"],
                        "tone": scenario["tone"],
                        "email": gen["email"],
                        "scores": scores,
                        "error": None,
                    }
                )
                print(f"  ok   {label}  composite={scores['composite']}")
            except Exception as exc:  # noqa: BLE001 - record and continue
                runs.append(
                    {
                        "model": model,
                        "scenario_id": scenario["id"],
                        "intent": scenario["intent"],
                        "tone": scenario["tone"],
                        "email": "",
                        "scores": None,
                        "error": str(exc),
                    }
                )
                print(f"  FAIL {label}  {exc}")

    averages = _averages(runs)
    return {
        "config": {
            "model_a": config.MODEL_A,
            "model_b": config.MODEL_B,
            "judge_model": config.JUDGE_MODEL,
            "prompt_technique": "Role-play + Few-shot + Chain-of-thought (delimited output)",
        },
        "pricing": _fetch_pricing(config.MODELS),
        "metrics": [
            {k: m[k] for k in ("key", "name", "focus", "technique", "definition", "logic")}
            for m in metrics.METRICS
        ],
        "scenarios": scenarios,
        "runs": runs,
        "averages": averages,
    }


def _averages(runs: list[dict]) -> dict:
    avg: dict[str, dict] = {}
    keys = metrics.METRIC_KEYS + ["composite"]
    for model in config.MODELS:
        scored = [r for r in runs if r["model"] == model and r["scores"]]
        if not scored:
            avg[model] = {k: None for k in keys}
            continue
        model_avg = {}
        for k in metrics.METRIC_KEYS:
            model_avg[k] = round(sum(r["scores"][k]["score"] for r in scored) / len(scored), 4)
        model_avg["composite"] = round(
            sum(r["scores"]["composite"] for r in scored) / len(scored), 4
        )
        model_avg["n_scored"] = len(scored)
        avg[model] = model_avg
    return avg


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------
def write_json(data: dict) -> None:
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(data, indent=2))


def write_csv(data: dict) -> None:
    path = config.RESULTS_DIR / "results.csv"
    cols = ["model", "scenario_id", "intent", "tone", *metrics.METRIC_KEYS, "composite", "error"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in data["runs"]:
            if r["scores"]:
                row = [r["model"], r["scenario_id"], r["intent"], r["tone"]]
                row += [r["scores"][k]["score"] for k in metrics.METRIC_KEYS]
                row += [r["scores"]["composite"], ""]
            else:
                row = [r["model"], r["scenario_id"], r["intent"], r["tone"]]
                row += ["", "", "", "", r["error"]]
            w.writerow(row)


def write_emails(data: dict) -> None:
    lines = ["# Generated emails\n"]
    for r in data["runs"]:
        lines.append(f"## {r['scenario_id']} — `{r['model']}`\n")
        if r["scores"]:
            lines.append(f"*composite: {r['scores']['composite']}*\n")
            lines.append("```\n" + r["email"] + "\n```\n")
        else:
            lines.append(f"**ERROR:** {r['error']}\n")
    (config.RESULTS_DIR / "emails.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------
# Report (Section 3 analysis is computed from the numbers)
# --------------------------------------------------------------------------
def _fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"


def _analysis(data: dict) -> str:
    avg = data["averages"]
    models = config.MODELS
    scored_models = [m for m in models if avg[m].get("composite") is not None]
    if len(scored_models) < 2:
        return "_Not enough successful runs to compare models. Re-run with a valid API key._"

    winner = max(scored_models, key=lambda m: avg[m]["composite"])
    loser = min(scored_models, key=lambda m: avg[m]["composite"])
    w, l = avg[winner], avg[loser]

    # Loser's biggest weakness = metric with the largest winner-minus-loser gap.
    gaps = {k: round(w[k] - l[k], 4) for k in metrics.METRIC_KEYS}
    worst_metric_key = max(gaps, key=gaps.get)
    worst_metric = next(m for m in metrics.METRICS if m["key"] == worst_metric_key)

    composite_gap = w["composite"] - l["composite"]
    rel_gap = composite_gap / w["composite"] if w["composite"] else 0.0
    close = rel_gap <= 0.03  # within 3% -> cheaper model is "good enough"

    lines = [
        "## 3. Comparative analysis\n",
        f"**Models compared (same prompt):** `{models[0]}` vs `{models[1]}`  ",
        f"**Judge:** `{config.JUDGE_MODEL}` (independent of both, to avoid self-preference bias)\n",
        "### Which performed better?\n",
        f"**`{winner}`** had the higher overall composite "
        f"({_fmt_pct(w['composite'])}% vs {_fmt_pct(l['composite'])}% for "
        f"`{loser}`) — a gap of {composite_gap*100:.1f} points. Per-metric averages:\n",
        "| Metric | " + winner + " | " + loser + " | Gap |",
        "|---|---|---|---|",
    ]
    for m in metrics.METRICS:
        k = m["key"]
        lines.append(
            f"| {m['name']} | {_fmt_pct(w[k])}% | {_fmt_pct(l[k])}% | {gaps[k]*100:+.1f} |"
        )
    lines.append(f"| **Composite** | **{_fmt_pct(w['composite'])}%** | **{_fmt_pct(l['composite'])}%** | **{composite_gap*100:+.1f}** |\n")

    lines.append("### Biggest failure mode of the lower-performing model\n")
    lines.append(
        f"`{loser}`'s largest deficit was on **{worst_metric['name']}** "
        f"({_fmt_pct(l[worst_metric_key])}% vs {_fmt_pct(w[worst_metric_key])}%, "
        f"a {gaps[worst_metric_key]*100:.1f}-point gap)."
    )
    # Concrete, metric-specific evidence pulled from the per-scenario details.
    loser_runs = [r for r in data["runs"] if r["model"] == loser and r["scores"]]
    if worst_metric_key == "fact_intent_coverage":
        dropped = [
            r["scenario_id"]
            for r in loser_runs
            if r["scores"]["fact_intent_coverage"]["detail"]["facts_included"]
            < r["scores"]["fact_intent_coverage"]["detail"]["facts_total"]
        ]
        if dropped:
            lines.append(
                f" It omitted at least one required key fact in {len(dropped)} of "
                f"{len(loser_runs)} scenarios ({', '.join(dropped)}) — the classic "
                "failure for a smaller model: it summarises and quietly drops details."
            )
    elif worst_metric_key == "tone_accuracy":
        off = [
            r["scenario_id"]
            for r in loser_runs
            if r["scores"]["tone_accuracy"]["detail"]["rating"] <= 3
        ]
        if off:
            lines.append(
                f" It missed the requested register (rating ≤ 3) in "
                f"{len(off)} of {len(loser_runs)} scenarios ({', '.join(off)})."
            )
    elif worst_metric_key == "conciseness_format":
        bloated = [
            r["scenario_id"]
            for r in loser_runs
            if r["scores"]["conciseness_format"]["detail"]["length_ratio"] > 1.4
        ]
        if bloated:
            lines.append(
                f" It ran noticeably long (>1.4x the reference length) in "
                f"{len(bloated)} scenarios ({', '.join(bloated)})."
            )
    lines.append("")

    lines.append("### Production recommendation\n")
    pricing = data.get("pricing", {})
    have_prices = winner in pricing and loser in pricing
    if have_prices:
        cheaper = min((winner, loser), key=lambda m: pricing[m]["blended_rank"])
        pricier = pricier_model = winner if cheaper == loser else loser
        base = pricing[cheaper]["blended_rank"]
        ratio = (pricing[pricier_model]["blended_rank"] / base) if base else float("inf")
        cost_note = (
            f"On cost, `{cheaper}` is ~{ratio:.1f}x cheaper than `{pricier_model}` "
            f"(${pricing[cheaper]['completion_per_mtok']:.2f} vs "
            f"${pricing[pricier_model]['completion_per_mtok']:.2f} per 1M output tokens)."
        )
        if winner == cheaper:
            lines.append(
                f"**Recommend `{winner}`.** It is both the higher-scoring model "
                f"({_fmt_pct(w['composite'])}% composite) *and* the cheaper one, so "
                "there is no quality-versus-cost trade-off to make. " + cost_note
            )
        elif close:
            lines.append(
                f"**Recommend `{cheaper}` for routine email generation.** `{winner}` "
                f"leads by only {composite_gap*100:.1f} points ({rel_gap*100:.1f}%), "
                "which is within noise for this task, while the cheaper model is "
                f"materially less expensive. Reserve `{winner}` for high-stakes, "
                "manually-reviewed sends. " + cost_note
            )
        else:
            lines.append(
                f"**Recommend `{winner}`.** Its {composite_gap*100:.1f}-point composite "
                f"lead — driven mainly by **{worst_metric['name']}**, the metric that "
                "most affects whether an email is send-ready — outweighs the cost "
                f"saving from `{loser}` on a customer-facing channel. " + cost_note
            )
    else:
        if close:
            lines.append(
                f"`{winner}` leads `{loser}` by only {composite_gap*100:.1f} points "
                f"({rel_gap*100:.1f}%). Given how close they are, decide on "
                f"cost/latency: if `{loser}` is cheaper it is the better routine-use "
                f"choice, with `{winner}` reserved for high-stakes sends. (Live "
                "pricing lookup was unavailable; confirm rates on openrouter.ai/models.)"
            )
        else:
            lines.append(
                f"`{winner}` is the recommended production model — its "
                f"{composite_gap*100:.1f}-point lead is driven mainly by "
                f"**{worst_metric['name']}**, which most affects send-readiness. "
                "(Live pricing lookup was unavailable; weigh against current rates.)"
            )
    return "\n".join(lines)


def write_report(data: dict) -> None:
    avg = data["averages"]
    parts: list[str] = []
    parts.append("# Email Generation Assistant — Evaluation Report\n")
    parts.append(
        "An LLM-powered assistant that turns an **intent**, **key facts**, and a "
        "**tone** into a polished, ready-to-send email, plus a custom evaluation "
        "harness that scores its output on three metrics and compares two models.\n"
    )
    parts.append(
        f"- **Models compared:** `{data['config']['model_a']}` vs "
        f"`{data['config']['model_b']}` (same prompt)\n"
        f"- **Judge model:** `{data['config']['judge_model']}`\n"
        f"- **Prompting technique:** {data['config']['prompt_technique']}\n"
    )

    parts.append("\n---\n\n## 1. Prompt template\n")
    parts.append(prompt.PROMPT_TEMPLATE_DOC)

    parts.append("\n---\n\n## 2. Custom metrics\n")
    parts.append(
        "Three metrics, each normalised to `[0, 1]`. The set deliberately mixes "
        "automated techniques, as the brief requires (two LLM-as-judge, one "
        "pure-Python). The **composite** score is their mean.\n"
    )
    for i, m in enumerate(metrics.METRICS, 1):
        # Escape '*' so Markdown doesn't read formula multiplication as emphasis.
        defn = m["definition"].replace("*", "\\*")
        logic = m["logic"].replace("*", "\\*")
        parts.append(f"### Metric {i}: {m['name']}\n")
        parts.append(f"- **Focus:** {m['focus']}")
        parts.append(f"- **Technique:** {m['technique']}")
        parts.append(f"- **Definition:** {defn}")
        parts.append(f"- **Logic:** {logic}\n")

    parts.append("\n---\n\n## Raw evaluation data\n")
    parts.append("### Per-model averages\n")
    header = "| Model | " + " | ".join(m["name"] for m in metrics.METRICS) + " | Composite |"
    parts.append(header)
    parts.append("|" + "---|" * (len(metrics.METRICS) + 2))
    for model in config.MODELS:
        a = avg[model]
        cells = [_fmt_pct(a.get(m["key"])) + "%" for m in metrics.METRICS]
        parts.append(f"| `{model}` | " + " | ".join(cells) + f" | {_fmt_pct(a.get('composite'))}% |")

    pricing = data.get("pricing", {})
    if pricing:
        parts.append("\n### Model cost (OpenRouter, $ per 1M tokens)\n")
        parts.append("| Model | Input | Output |")
        parts.append("|---|---|---|")
        for model in config.MODELS:
            p = pricing.get(model)
            if p:
                parts.append(
                    f"| `{model}` | ${p['prompt_per_mtok']:.2f} | ${p['completion_per_mtok']:.2f} |"
                )

    parts.append("\n### Per-scenario composite scores\n")
    parts.append("| Scenario | " + " | ".join(f"`{m}`" for m in config.MODELS) + " |")
    parts.append("|" + "---|" * (len(config.MODELS) + 1))
    by_scenario: dict[str, dict] = {}
    for r in data["runs"]:
        by_scenario.setdefault(r["scenario_id"], {})[r["model"]] = r
    for sid in [s["id"] for s in data["scenarios"]]:
        cells = []
        for model in config.MODELS:
            r = by_scenario.get(sid, {}).get(model)
            cells.append(_fmt_pct(r["scores"]["composite"]) + "%" if r and r["scores"] else "FAIL")
        parts.append(f"| {sid} | " + " | ".join(cells) + " |")
    parts.append(
        "\nFull per-fact / per-metric detail and every generated email are in "
        "`results/results.json`, `results/results.csv`, and `results/emails.md`.\n"
    )

    parts.append("\n---\n\n" + _analysis(data) + "\n")
    (config.REPORT_DIR / "REPORT.md").write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the email-assistant evaluation.")
    ap.add_argument("--fresh", action="store_true", help="Regenerate every email (ignore cache).")
    args = ap.parse_args()

    print(f"Models: {config.MODELS} | Judge: {config.JUDGE_MODEL}\n")
    data = run(fresh=args.fresh)
    write_json(data)
    write_csv(data)
    write_emails(data)
    write_report(data)

    print("\nWrote:")
    print("  results/results.json")
    print("  results/results.csv")
    print("  results/emails.md")
    print("  report/REPORT.md")
    print("\nPer-model composite:")
    for model in config.MODELS:
        print(f"  {model}: {_fmt_pct(data['averages'][model].get('composite'))}%")


if __name__ == "__main__":
    main()
