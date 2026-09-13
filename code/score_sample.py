"""Validate pipeline quality using sample_requests.csv as ground truth.

This script:
  1. Runs the full pipeline on the 25 labeled sample requests.
  2. Writes their predictions to  sample_output.csv  (same column layout as
     the final output.csv) so you can open it in Excel / Sheets and visually
     compare every field to sample_requests.csv.
  3. Prints a colour-free score report to stdout: per-field accuracy, a
     confusion matrix for affordability_status, and a line-by-line diff for
     every mismatch, formatted so you can read it without scrolling sideways.

Usage (run from repo root):
    python code/score_sample.py

The script never touches your real output.csv.

Field scoring rules (matching the problem spec):
  amount_safe_to_pay        within ± 1.0 of expected (currency-unit tolerance)
  affordability_status      exact string match
  recommended_payment_method exact string match
  payment_plan              exact string match (order and amounts must match)
  earliest_date_for_full_payment  exact string match; blank == blank
  spending_changes_needed   exact string match
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Allow running from repo root: `python code/score_sample.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from models import FinancialRequest
from plan_ranker import FLEXIBLE_STATES, select_plan
from state_builder import build_forecastable_events
from usage_report import write_usage_report
from validator import safe_fallback_result, validate_result

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_CSV = DATASET_DIR / "sample_requests.csv"
SAMPLE_OUTPUT_CSV = REPO_ROOT / "sample_output.csv"
CACHE_DIR = REPO_ROOT / ".cache"

FORECAST_HORIZON_DAYS = 90
AMOUNT_TOLERANCE = 1.0

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

SCORED_FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]

STATUS_VALUES = [
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
]


def _fmt(amount: float) -> str:
    if abs(amount - round(amount)) < 1e-6:
        return str(int(round(amount)))
    return f"{amount:.2f}"


def _normalize_date(value) -> str:
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "") else text


def _score_field(field: str, predicted: str, expected: str) -> bool:
    if field == "amount_safe_to_pay":
        try:
            return abs(float(predicted) - float(expected)) <= AMOUNT_TOLERANCE
        except (ValueError, TypeError):
            return False
    if field == "earliest_date_for_full_payment":
        return _normalize_date(predicted) == _normalize_date(expected)
    return str(predicted).strip() == str(expected).strip()


def main() -> None:
    print("Loading dataset …")
    data = load_all(DATASET_DIR)
    evidence_cache = DiskCache(CACHE_DIR, "evidence")
    explanation_cache = DiskCache(CACHE_DIR, "explanations")

    print("Extracting evidence (image amounts + message facts) …")
    apply_image_extractions(data, evidence_cache)
    facts = extract_all_message_facts(data, evidence_cache)
    facts = validate_message_facts(facts, data)
    fx = CurrencyConverter(data.exchange_rates)

    # Load sample requests (ground truth)
    sample_df = pd.read_csv(SAMPLE_CSV, dtype=str)

    predicted_rows: List[Dict] = []
    per_field_correct: Counter = Counter()
    per_type_correct: defaultdict = defaultdict(int)
    per_type_total: defaultdict = defaultdict(int)
    mismatches: List[Dict] = []
    confusion: Dict[str, Counter] = {s: Counter() for s in STATUS_VALUES}
    fully_correct_count = 0

    print(f"Running pipeline on {len(sample_df)} sample requests …\n")
    for row in sample_df.itertuples(index=False):
        request = FinancialRequest(
            request_id=row.request_id,
            user_id=row.user_id,
            request_date=date.fromisoformat(row.request_date),
            request_type=row.request_type,
            requested_amount=float(row.requested_amount),
            desired_completion_date=date.fromisoformat(row.desired_completion_date),
            allows_partial_payment=str(row.allows_partial_payment).strip().lower() == "true",
            request_text=row.request_text,
        )

        profile = data.profiles_by_user[request.user_id]
        horizon_end = request.request_date + timedelta(days=FORECAST_HORIZON_DAYS)
        events = build_forecastable_events(
            data, request.user_id, facts, request.request_date, horizon_end
        )
        payment_options = data.payment_options_by_request.get(request.request_id, [])

        result = select_plan(request, profile, events, payment_options, fx, horizon_end)

        flexible_ids = {e.event_id for e in events if e.flexibility in FLEXIBLE_STATES}
        report = validate_result(result, request, payment_options, flexible_ids)
        if not report.is_valid:
            result = safe_fallback_result()

        explanation = generate_explanation(request, profile, result, explanation_cache)

        pred: Dict = {
            "request_id": request.request_id,
            "amount_safe_to_pay": _fmt(result.amount_safe_to_pay),
            "affordability_status": result.affordability_status,
            "recommended_payment_method": result.recommended_payment_method,
            "payment_plan": result.payment_plan,
            "earliest_date_for_full_payment": _normalize_date(result.earliest_date_for_full_payment),
            "spending_changes_needed": result.spending_changes_needed,
            "decision_explanation": explanation,
        }
        predicted_rows.append(pred)

        gt: Dict = {
            "amount_safe_to_pay": str(row.amount_safe_to_pay),
            "affordability_status": str(row.affordability_status),
            "recommended_payment_method": str(row.recommended_payment_method),
            "payment_plan": str(row.payment_plan),
            "earliest_date_for_full_payment": _normalize_date(row.earliest_date_for_full_payment),
            "spending_changes_needed": str(row.spending_changes_needed),
        }

        per_type_total[request.request_type] += 1
        field_results = {}
        all_ok = True
        for f in SCORED_FIELDS:
            ok = _score_field(f, pred[f], gt[f])
            field_results[f] = ok
            if ok:
                per_field_correct[f] += 1
            else:
                all_ok = False

        # Confusion matrix for affordability_status
        expected_status = gt["affordability_status"]
        predicted_status = pred["affordability_status"]
        if expected_status in confusion:
            confusion[expected_status][predicted_status] += 1

        if all_ok:
            fully_correct_count += 1
            per_type_correct[request.request_type] += 1
        else:
            mismatches.append(
                {
                    "request_id": request.request_id,
                    "request_type": request.request_type,
                    "field_results": field_results,
                    "pred": pred,
                    "gt": gt,
                }
            )

    total = len(sample_df)

    # ── Write sample_output.csv ────────────────────────────────────────
    with SAMPLE_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(predicted_rows)
    print(f"Wrote predictions to:  {SAMPLE_OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Compare with:          dataset/sample_requests.csv\n")

    # ── Score report ───────────────────────────────────────────────────
    bar = "═" * 60
    print(bar)
    print(f"  SCORE REPORT  ({total} sample requests)")
    print(bar)

    print("\nPer-field accuracy:")
    max_field_len = max(len(f) for f in SCORED_FIELDS)
    for f in SCORED_FIELDS:
        correct = per_field_correct[f]
        pct = 100 * correct / total
        blocks = int(pct / 5)  # 20 blocks = 100%
        bar_str = "█" * blocks + "░" * (20 - blocks)
        print(f"  {f:<{max_field_len}}  {bar_str}  {correct:2d}/{total}  ({pct:5.1f}%)")

    print(f"\nFully correct rows: {fully_correct_count}/{total}  "
          f"({100*fully_correct_count/total:.1f}%)")

    print("\nPer-request_type (all fields correct):")
    for req_type in sorted(per_type_total):
        n = per_type_total[req_type]
        c = per_type_correct[req_type]
        print(f"  {req_type:<22}  {c}/{n}")

    print("\nAffordability status confusion matrix (row=expected, col=predicted):")
    col_w = 22
    cols = STATUS_VALUES
    header = " " * 22 + "".join(f"{c:<{col_w}}" for c in cols)
    print(f"  {header}")
    for expected_s in STATUS_VALUES:
        row_str = f"  {expected_s:<22}"
        for predicted_s in cols:
            count = confusion[expected_s][predicted_s]
            cell = str(count) if count else "·"
            row_str += f"{cell:<{col_w}}"
        print(row_str)

    if not mismatches:
        print("\n✓ All rows correct — nothing to fix!")
        return

    print(f"\n{'─'*60}")
    print(f"MISMATCHES ({len(mismatches)} rows):")
    print(f"{'─'*60}")
    for m in mismatches:
        print(f"\n  {m['request_id']}  [{m['request_type']}]")
        for f in SCORED_FIELDS:
            ok = m["field_results"][f]
            if not ok:
                pred_val = m["pred"][f]
                gt_val = m["gt"][f]
                print(f"    ✗ {f}")
                print(f"        got:      {pred_val!r}")
                print(f"        expected: {gt_val!r}")


if __name__ == "__main__":
    main()
