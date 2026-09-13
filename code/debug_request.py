"""Diagnostic tool: prints a detailed balance breakdown for one request.

Usage:
    python code/debug_request.py request_01
    python code/debug_request.py request_01 request_16 request_25

For each request_id, prints:
  - User profile summary
  - All events in the 90-day window (real and projected)
  - Balance timeline checkpoints
  - max_safe_payment_on result
  - expected vs computed values (for sample_requests only)
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Make sure code/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_all
from forecast_engine import build_balance_timeline, max_safe_payment_on, earliest_safe_full_payment_date
from fx import CurrencyConverter
from models import FinancialRequest
from state_builder import build_forecastable_events
from plan_ranker import select_plan
from cache import DiskCache
from evidence_agent import apply_image_extractions, extract_all_message_facts, validate_message_facts

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / ".cache"
FORECAST_HORIZON_DAYS = 90


def load_ground_truth() -> dict:
    sample_path = DATASET_DIR / "sample_requests.csv"
    if not sample_path.exists():
        return {}
    df = pd.read_csv(sample_path, dtype=str)
    return {row.request_id: row._asdict() for row in df.itertuples(index=False)}


def main(request_ids: list[str]) -> None:
    print("Loading dataset…")
    data = load_all(DATASET_DIR)
    cache = DiskCache(CACHE_DIR, "evidence")
    apply_image_extractions(data, cache)
    facts = extract_all_message_facts(data, cache)
    facts = validate_message_facts(facts, data)
    fx = CurrencyConverter(data.exchange_rates)
    ground_truth = load_ground_truth()

    # Build a quick request lookup from all sources
    all_requests = {r.request_id: r for r in data.requests}
    # Also load sample_requests if not already present
    sample_path = DATASET_DIR / "sample_requests.csv"
    if sample_path.exists():
        import math
        from datetime import date as date_type
        df = pd.read_csv(sample_path, dtype=str)
        for row in df.itertuples(index=False):
            from datetime import datetime
            req_id = row.request_id
            if req_id not in all_requests:
                all_requests[req_id] = FinancialRequest(
                    request_id=req_id,
                    user_id=row.user_id,
                    request_date=datetime.strptime(row.request_date, "%Y-%m-%d").date(),
                    request_type=row.request_type,
                    requested_amount=float(row.requested_amount),
                    desired_completion_date=datetime.strptime(row.desired_completion_date, "%Y-%m-%d").date(),
                    allows_partial_payment=str(row.allows_partial_payment).strip().lower() == "true",
                    request_text=row.request_text,
                )

    for req_id in request_ids:
        request = all_requests.get(req_id)
        if request is None:
            print(f"\n[!] {req_id} not found in requests or sample_requests")
            continue

        profile = data.profiles_by_user.get(request.user_id)
        if profile is None:
            print(f"\n[!] Profile not found for user {request.user_id}")
            continue

        horizon_end = request.request_date + timedelta(days=FORECAST_HORIZON_DAYS)
        events = build_forecastable_events(data, request.user_id, facts, request.request_date, horizon_end)
        payment_options = data.payment_options_by_request.get(req_id, [])

        timeline = build_balance_timeline(
            events, profile.current_available_balance, profile.home_currency, fx,
            request.request_date, horizon_end,
        )

        safe = max_safe_payment_on(timeline, request.request_date, horizon_end,
                                   profile.minimum_balance_to_keep, request.requested_amount)
        earliest = earliest_safe_full_payment_date(timeline, request.request_date, horizon_end,
                                                   profile.minimum_balance_to_keep, request.requested_amount)
        result = select_plan(request, profile, events, payment_options, fx, horizon_end)

        sep = "─" * 70
        print(f"\n{sep}")
        print(f"  REQUEST: {req_id}  ({request.request_type})  user={request.user_id}")
        print(sep)
        print(f"  request_date:             {request.request_date}")
        print(f"  requested_amount:         {request.requested_amount} {profile.home_currency}")
        print(f"  desired_completion_date:  {request.desired_completion_date}")
        print(f"  allows_partial_payment:   {request.allows_partial_payment}")
        print()
        print(f"  PROFILE")
        print(f"    home_currency:           {profile.home_currency}")
        print(f"    current_available_bal:   {profile.current_available_balance:,.2f}")
        print(f"    minimum_balance_to_keep: {profile.minimum_balance_to_keep:,.2f}")
        print(f"    payment_methods:         {profile.payment_methods_user_will_consider}")
        print(f"    max_installment_months:  {profile.max_installment_months}")
        print(f"    protect:  {profile.expense_categories_to_protect}")
        print(f"    reducible: {profile.expense_categories_user_is_willing_to_reduce}")
        print(f"    stoppable: {profile.expense_categories_user_is_willing_to_stop}")
        print()

        # Events in window
        from forecast_engine import _effective_date as eff_date
        window_events = sorted(
            [e for e in events if request.request_date <= eff_date(e) <= horizon_end],
            key=eff_date
        )
        real = [e for e in window_events if not e.is_projected]
        projected_evs = [e for e in window_events if e.is_projected]
        print(f"  EVENTS IN 90-DAY WINDOW: {len(window_events)} total "
              f"({len(real)} real, {len(projected_evs)} projected)")

        debit_total = sum((e.amount or 0) for e in window_events if e.direction == "debit")
        credit_total = sum((e.amount or 0) for e in window_events if e.direction == "credit")
        print(f"    Total debits  (home ccy approx): {debit_total:,.2f}")
        print(f"    Total credits (home ccy approx): {credit_total:,.2f}")
        print()
        if projected_evs:
            print(f"  PROJECTED EVENTS ({len(projected_evs)}):")
            for e in projected_evs:
                print(f"    {eff_date(e)}  {e.direction:6s}  {(e.amount or 0):>14,.2f} {e.currency}  [{e.category}]")
        print()

        print(f"  BALANCE TIMELINE ({len(timeline.dates)} checkpoints):")
        print(f"    Start balance: {profile.current_available_balance:,.2f} {profile.home_currency}")
        for d, bal in zip(timeline.dates[:20], timeline.cumulative_balances[:20]):
            print(f"    {d}  →  {bal:>14,.2f}")
        if len(timeline.dates) > 20:
            print(f"    … ({len(timeline.dates) - 20} more checkpoints)")
        if timeline.dates:
            from forecast_engine import BalanceTimeline
            min_bal = timeline.min_balance_from(request.request_date, horizon_end)
            print(f"    MIN balance in window: {min_bal:,.2f}")
        print()

        print(f"  COMPUTED RESULT:")
        print(f"    amount_safe_to_pay:          {safe:,.2f}")
        print(f"    earliest_safe_full_payment:  {earliest}")
        print(f"    status:                      {result.affordability_status}")
        print(f"    method:                      {result.recommended_payment_method}")
        print(f"    payment_plan:                {result.payment_plan}")
        print(f"    earliest_date_for_full_pay:  {result.earliest_date_for_full_payment}")
        print(f"    spending_changes_needed:     {result.spending_changes_needed}")

        gt = ground_truth.get(req_id)
        if gt:
            print()
            print(f"  GROUND TRUTH:")
            print(f"    amount_safe_to_pay:          {gt.get('amount_safe_to_pay')}")
            print(f"    status:                      {gt.get('affordability_status')}")
            print(f"    method:                      {gt.get('recommended_payment_method')}")
            print(f"    payment_plan:                {gt.get('payment_plan')}")
            print(f"    earliest_date_for_full_pay:  {gt.get('earliest_date_for_full_payment')}")
            print(f"    spending_changes_needed:     {gt.get('spending_changes_needed')}")

        # Payment options available
        if payment_options:
            print()
            print(f"  PAYMENT OPTIONS ({len(payment_options)}):")
            for opt in payment_options:
                print(f"    {opt.payment_option_id}: {opt.payment_method}  "
                      f"{opt.number_of_payments}x{opt.payment_amount:.2f}  "
                      f"first={opt.first_payment_date}  freq={opt.payment_frequency_days}d  "
                      f"total={opt.total_payable_amount:.2f}  fee={opt.financing_fee:.2f}")


if __name__ == "__main__":
    ids = sys.argv[1:] if len(sys.argv) > 1 else ["request_01"]
    main(ids)
