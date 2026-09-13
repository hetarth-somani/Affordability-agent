"""Entry point: runs the full Buy or Wait? pipeline over dataset/requests.csv.

Stages, in order:
  load and index all input files
    -> evidence extraction (images for blank amounts, messages for amendments)
    -> per-request financial state reconstruction and 90-day forecast
    -> candidate plan generation and spec-ordered ranking
    -> typed validation of every output field
    -> grounded explanation generation
    -> output.csv in the repository root

Run with:  python code/main.py
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import timedelta
from pathlib import Path

from cache import DiskCache
from data_loader import load_all
from evidence_agent import (
    apply_image_extractions,
    extract_all_message_facts,
    validate_message_facts,
)
from explainer import generate_explanation
from fx import CurrencyConverter
from llm_client import USAGE
from plan_ranker import FLEXIBLE_STATES, select_plan
from state_builder import build_forecastable_events
from usage_report import write_usage_report
from validator import safe_fallback_result, validate_result

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_PATH = REPO_ROOT / "output.csv"
CACHE_DIR = REPO_ROOT / ".cache"
USAGE_REPORT_PATH = Path(__file__).resolve().parent / "evaluation" / "usage_report.md"

FORECAST_HORIZON_DAYS = 90

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


def format_amount(amount: float) -> str:
    if abs(amount - round(amount)) < 1e-6:
        return str(int(round(amount)))
    return f"{amount:.2f}"


def main() -> None:
    logger.info("Loading dataset from %s", DATASET_DIR)
    data = load_all(DATASET_DIR)
    evidence_cache = DiskCache(CACHE_DIR, "evidence")
    explanation_cache = DiskCache(CACHE_DIR, "explanations")

    logger.info("Extracting amounts from linked images")
    filled = apply_image_extractions(data, evidence_cache)
    logger.info("Resolved %d blank event amounts from images", filled)

    logger.info("Extracting structured facts from messages")
    message_facts = extract_all_message_facts(data, evidence_cache)
    message_facts = validate_message_facts(message_facts, data)
    logger.info("Extracted facts from %d messages", len(message_facts))

    fx = CurrencyConverter(data.exchange_rates)

    rows = []
    validation_failures = 0

    for index, request in enumerate(data.requests, start=1):
        if index % 25 == 0:
            logger.info("Processed %d/%d requests", index, len(data.requests))

        profile = data.profiles_by_user[request.user_id]
        horizon_end = request.request_date + timedelta(days=FORECAST_HORIZON_DAYS)
        events = build_forecastable_events(
            data, request.user_id, message_facts, request.request_date, horizon_end
        )
        payment_options = data.payment_options_by_request.get(request.request_id, [])

        result = select_plan(request, profile, events, payment_options, fx, horizon_end)

        flexible_ids = {e.event_id for e in events if e.flexibility in FLEXIBLE_STATES}
        report = validate_result(result, request, payment_options, flexible_ids)
        if not report.is_valid:
            validation_failures += 1
            logger.error(
                "Falling back to a no-action result for %s after validation failure",
                request.request_id,
            )
            result = safe_fallback_result()

        explanation = generate_explanation(request, profile, result, explanation_cache)

        rows.append(
            {
                "request_id": request.request_id,
                "amount_safe_to_pay": format_amount(result.amount_safe_to_pay),
                "affordability_status": result.affordability_status,
                "recommended_payment_method": result.recommended_payment_method,
                "payment_plan": result.payment_plan,
                "earliest_date_for_full_payment": (
                    result.earliest_date_for_full_payment.isoformat()
                    if result.earliest_date_for_full_payment
                    else ""
                ),
                "spending_changes_needed": result.spending_changes_needed,
                "decision_explanation": explanation,
            }
        )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), OUTPUT_PATH)
    if validation_failures:
        logger.warning("%d rows failed validation and used the fallback", validation_failures)

    write_usage_report(USAGE, len(rows), USAGE_REPORT_PATH)
    logger.info("Wrote usage report to %s", USAGE_REPORT_PATH)


if __name__ == "__main__":
    main()
