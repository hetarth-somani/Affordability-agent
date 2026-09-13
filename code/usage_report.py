"""Writes the required evaluation/usage_report.md from recorded token usage."""

from __future__ import annotations

from pathlib import Path

from llm_client import UsageTracker

PRICING_USD_PER_MILLION = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.0, "output": 0.0},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "qwen/qwen3.6-27b": {"input": 0.30, "output": 0.60},
}
DEFAULT_PRICING = {"input": 0.0, "output": 0.0}


def _model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING_USD_PER_MILLION.get(model, DEFAULT_PRICING)
    return (
        input_tokens / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


def write_usage_report(usage: UsageTracker, request_count: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total_input = sum(usage.input_tokens_by_model.values())
    total_output = sum(usage.output_tokens_by_model.values())
    total_tokens = total_input + total_output
    total_cost = sum(
        _model_cost(model, usage.input_tokens_by_model[model], usage.output_tokens_by_model[model])
        for model in usage.calls_by_model
    )

    lines = [
        "# Token Usage and Cost Analysis",
        "",
        "Figures below cover the final full-dataset run that produced "
        "`output.csv`. All model calls go through Groq.",
        "",
        "## Per-model totals",
        "",
        "| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) |",
        "|---|---|---|---|---|---|---|",
    ]

    for model in sorted(usage.calls_by_model):
        calls = usage.calls_by_model[model]
        input_tokens = usage.input_tokens_by_model[model]
        output_tokens = usage.output_tokens_by_model[model]
        cost = _model_cost(model, input_tokens, output_tokens)
        lines.append(
            f"| Groq | {model} | {calls:,} | {input_tokens:,} | {output_tokens:,} | "
            f"{input_tokens + output_tokens:,} | ${cost:.4f} |"
        )

    per_request_tokens = total_tokens / request_count if request_count else 0.0
    per_request_cost = total_cost / request_count if request_count else 0.0

    lines.extend(
        [
            "",
            "## Overall totals",
            "",
            f"- Requests processed: **{request_count:,}**",
            f"- Model calls: **{usage.total_calls:,}**",
            f"- Input tokens: **{total_input:,}**",
            f"- Output tokens: **{total_output:,}**",
            f"- Total tokens: **{total_tokens:,}**",
            f"- Average tokens per request: **{per_request_tokens:,.1f}**",
            f"- Estimated total cost: **${total_cost:.4f}**",
            f"- Estimated cost per request: **${per_request_cost:.6f}**",
            "",
            "## Notes on efficiency",
            "",
            "- Image understanding runs once per image and is cached on disk, so "
            "re-runs and graders do not re-pay for it.",
            "- Message fact extraction runs once per unique message across the "
            "whole dataset rather than once per request.",
            "- Explanations are cached by request id plus the exact fact set, so "
            "an unchanged decision never regenerates its explanation.",
            "- All planning, forecasting, and validation are deterministic Python "
            "with no model calls.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
