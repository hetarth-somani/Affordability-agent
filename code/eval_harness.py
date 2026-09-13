"""Scores the pipeline's predictions against the 25 labeled examples in
sample_requests.csv, field by field, with per-request_type breakdown and a
full mismatch listing to drive iteration.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from cache import DiskCache
from data_loader import load_all
from evidence_agent import apply_image_extractions, extract_all_message_facts, validate_message_facts
from fx import CurrencyConverter
from models import FinancialRequest
from plan_ranker import select_plan
from state_builder import build_forecastable_events

REPO_ROOT = Path(__file__).resolve().parent.parent
AMOUNT_TOLERANCE = 1.0
FORECAST_HORIZON_DAYS = 90


@dataclass
class FieldScores:
    amount_safe_to_pay: bool
    affordability_status: bool
    recommended_payment_method: bool
    payment_plan: bool
    earliest_date_for_full_payment: bool
    spending_changes_needed: bool

    @property
    def all_correct(self) -> bool:
        return all(
            [
                self.amount_safe_to_pay,
                self.affordability_status,
                self.recommended_payment_method,
                self.payment_plan,
                self.earliest_date_for_full_payment,
                self.spending_changes_needed,
            ]
        )


def _amounts_close(predicted: float, expected: float) -> bool:
    return abs(predicted - expected) <= AMOUNT_TOLERANCE


def _dates_match(predicted: Optional[date], expected: str) -> bool:
    expected = "" if pd.isna(expected) else str(expected).strip()
    predicted_str = predicted.isoformat() if predicted else ""
    return predicted_str == expected


def run_eval() -> None:
    data = load_all(REPO_ROOT / "dataset")
    cache = DiskCache(REPO_ROOT / ".cache", "evidence")

    apply_image_extractions(data, cache)
    facts = extract_all_message_facts(data, cache)
    facts = validate_message_facts(facts, data)
    fx = CurrencyConverter(data.exchange_rates)

    sample_df = pd.read_csv(REPO_ROOT / "dataset" / "sample_requests.csv", dtype=str)

    per_field_correct: Counter = Counter()
    per_type_totals: defaultdict = defaultdict(int)
    per_type_correct: defaultdict = defaultdict(int)
    mismatches: List[str] = []
    total = 0

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
        total += 1

        profile = data.profiles_by_user[request.user_id]
        horizon_end = request.request_date + timedelta(days=FORECAST_HORIZON_DAYS)
        events = build_forecastable_events(data, request.user_id, facts, request.request_date, horizon_end)
        payment_options = data.payment_options_by_request.get(request.request_id, [])

        result = select_plan(request, profile, events, payment_options, fx, horizon_end)

        scores = FieldScores(
            amount_safe_to_pay=_amounts_close(result.amount_safe_to_pay, float(row.amount_safe_to_pay)),
            affordability_status=result.affordability_status == row.affordability_status,
            recommended_payment_method=result.recommended_payment_method == row.recommended_payment_method,
            payment_plan=result.payment_plan == row.payment_plan,
            earliest_date_for_full_payment=_dates_match(
                result.earliest_date_for_full_payment, row.earliest_date_for_full_payment
            ),
            spending_changes_needed=result.spending_changes_needed == row.spending_changes_needed,
        )

        for field_name in scores.__dataclass_fields__:
            if getattr(scores, field_name):
                per_field_correct[field_name] += 1

        per_type_totals[request.request_type] += 1
        if scores.all_correct:
            per_type_correct[request.request_type] += 1
        else:
            mismatches.append(
                f"{request.request_id} ({request.request_type}): "
                f"got amount={result.amount_safe_to_pay} status={result.affordability_status} "
                f"method={result.recommended_payment_method} plan={result.payment_plan} "
                f"earliest={result.earliest_date_for_full_payment} changes={result.spending_changes_needed} "
                f"| expected amount={row.amount_safe_to_pay} status={row.affordability_status} "
                f"method={row.recommended_payment_method} plan={row.payment_plan} "
                f"earliest={row.earliest_date_for_full_payment} changes={row.spending_changes_needed}"
            )

    print(f"Total samples evaluated: {total}\n")
    print("Per-field accuracy:")
    for field_name, correct_count in per_field_correct.items():
        print(f"  {field_name}: {correct_count}/{total} ({100 * correct_count / total:.1f}%)")

    print("\nPer-request_type accuracy (all fields correct):")
    for request_type, type_total in sorted(per_type_totals.items()):
        correct_count = per_type_correct[request_type]
        print(f"  {request_type}: {correct_count}/{type_total}")

    fully_correct = total - len(mismatches)
    print(f"\nFully correct rows: {fully_correct}/{total}")

    print("\nMismatches:")
    for mismatch in mismatches:
        print(f"  {mismatch}")


if __name__ == "__main__":
    run_eval()
