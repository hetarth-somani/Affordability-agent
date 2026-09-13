"""Typed, field-by-field validation of every output row before it is written.

Each of the eight output fields is checked independently for type, allowed
values, format, and cross-field consistency. A row that fails validation is
never silently written: the specific violations are logged and the row is
replaced by a conservative, schema-valid fallback that recommends no action,
so a malformed model or planner result degrades into an explicit "do not
proceed" rather than a plausible-looking wrong answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Set

from models import FinancialRequest, PaymentOption
from plan_ranker import PlanResult

logger = logging.getLogger(__name__)

ALLOWED_AFFORDABILITY_STATUS = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

ALLOWED_PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}

PAYMENT_PLAN_ENTRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:\d+(\.\d+)?$")
SPENDING_STOP_RE = re.compile(r"^stop:[A-Za-z0-9_]+$")
SPENDING_REDUCE_RE = re.compile(r"^reduce_to:[A-Za-z0-9_]+:\d+(\.\d+)?$")

MAX_SPENDING_CHANGES = 3
AMOUNT_EPSILON = 0.01


@dataclass
class ValidationReport:
    request_id: str
    violations: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.violations


def _parse_plan_entries(payment_plan: str) -> Optional[List[tuple[date, float]]]:
    if payment_plan == "none":
        return []
    entries: List[tuple[date, float]] = []
    for token in payment_plan.split("|"):
        if not PAYMENT_PLAN_ENTRY_RE.match(token):
            return None
        raw_date, raw_amount = token.split(":")
        entries.append((date.fromisoformat(raw_date), float(raw_amount)))
    return entries


def _validate_payment_plan(
    result: PlanResult,
    request: FinancialRequest,
    payment_options: List[PaymentOption],
    report: ValidationReport,
) -> None:
    entries = _parse_plan_entries(result.payment_plan)
    if entries is None:
        report.violations.append(f"payment_plan has malformed entries: {result.payment_plan!r}")
        return

    if result.recommended_payment_method == "not_recommended":
        if entries:
            report.violations.append("payment_plan must be 'none' for method 'not_recommended'")
        return

    if result.recommended_payment_method == "wait":
        # wait plans may have exactly one entry: earliest_date:requested_amount
        if len(entries) > 1:
            report.violations.append("wait payment_plan may have at most one entry")
        return

    if not entries:
        report.violations.append(
            f"payment_plan must list payments for method {result.recommended_payment_method}"
        )
        return

    dates = [entry_date for entry_date, _ in entries]
    if dates != sorted(dates):
        report.violations.append("payment_plan entries are not in chronological order")

    total = sum(amount for _, amount in entries)

    if result.recommended_payment_method == "full_payment":
        if abs(total - request.requested_amount) > AMOUNT_EPSILON:
            report.violations.append(
                f"full_payment plan total {total} does not equal requested_amount "
                f"{request.requested_amount}"
            )

    elif result.recommended_payment_method == "partial_payment":
        if len(entries) != 2:
            report.violations.append("partial_payment plan must contain exactly two payments")
        if abs(total - request.requested_amount) > AMOUNT_EPSILON:
            report.violations.append(
                f"partial_payment plan total {total} does not equal requested_amount "
                f"{request.requested_amount}"
            )
        if result.affordability_status != "affordable_with_plan":
            report.violations.append(
                "partial_payment requires affordability_status 'affordable_with_plan'"
            )
        if not request.allows_partial_payment:
            report.violations.append("partial_payment used on a request that does not allow it")
        if entries and entries[0][0] != request.request_date:
            report.violations.append("partial_payment first payment must fall on request_date")

    elif result.recommended_payment_method == "installments":
        if not _matches_any_option(entries, payment_options):
            report.violations.append(
                "installment plan does not exactly match any supplied payment option"
            )


def _matches_any_option(
    entries: List[tuple[date, float]], payment_options: List[PaymentOption]
) -> bool:
    for option in payment_options:
        frequency = option.payment_frequency_days or 0
        expected = [
            (
                option.first_payment_date
                if i == 0
                else option.first_payment_date.fromordinal(
                    option.first_payment_date.toordinal() + frequency * i
                ),
                option.payment_amount,
            )
            for i in range(option.number_of_payments)
        ]
        if len(expected) != len(entries):
            continue
        if all(
            expected_date == actual_date and abs(expected_amount - actual_amount) <= AMOUNT_EPSILON
            for (expected_date, expected_amount), (actual_date, actual_amount) in zip(expected, entries)
        ):
            return True
    return False


def _validate_spending_changes(
    result: PlanResult, flexible_event_ids: Set[str], report: ValidationReport
) -> None:
    if result.spending_changes_needed == "none":
        return

    tokens = result.spending_changes_needed.split("|")
    if len(tokens) > MAX_SPENDING_CHANGES:
        report.violations.append(
            f"spending_changes_needed has {len(tokens)} changes, maximum is {MAX_SPENDING_CHANGES}"
        )

    referenced_ids: List[str] = []
    for token in tokens:
        if SPENDING_STOP_RE.match(token):
            referenced_ids.append(token.split(":")[1])
        elif SPENDING_REDUCE_RE.match(token):
            referenced_ids.append(token.split(":")[1])
        else:
            report.violations.append(f"spending change has invalid format: {token!r}")
            continue

    for event_id in referenced_ids:
        if event_id not in flexible_event_ids:
            report.violations.append(
                f"spending change references {event_id!r}, which is not a known flexible event"
            )

    if len(set(referenced_ids)) != len(referenced_ids):
        report.violations.append("spending changes reference the same event more than once")


def validate_result(
    result: PlanResult,
    request: FinancialRequest,
    payment_options: List[PaymentOption],
    flexible_event_ids: Set[str],
) -> ValidationReport:
    """Checks every output field of one result against the output contract."""
    report = ValidationReport(request_id=request.request_id)

    if not isinstance(result.amount_safe_to_pay, (int, float)):
        report.violations.append("amount_safe_to_pay is not numeric")
    elif not (-AMOUNT_EPSILON <= result.amount_safe_to_pay <= request.requested_amount + AMOUNT_EPSILON):
        report.violations.append(
            f"amount_safe_to_pay {result.amount_safe_to_pay} outside [0, "
            f"{request.requested_amount}]"
        )

    if result.affordability_status not in ALLOWED_AFFORDABILITY_STATUS:
        report.violations.append(
            f"affordability_status {result.affordability_status!r} is not an allowed value"
        )

    if result.recommended_payment_method not in ALLOWED_PAYMENT_METHODS:
        report.violations.append(
            f"recommended_payment_method {result.recommended_payment_method!r} is not allowed"
        )

    _validate_payment_plan(result, request, payment_options, report)

    if result.affordability_status == "affordable_now":
        if result.earliest_date_for_full_payment != request.request_date:
            report.violations.append(
                "affordable_now requires earliest_date_for_full_payment to equal request_date"
            )
    elif result.earliest_date_for_full_payment is not None:
        if not isinstance(result.earliest_date_for_full_payment, date):
            report.violations.append("earliest_date_for_full_payment is not a date")
        elif result.earliest_date_for_full_payment < request.request_date:
            report.violations.append(
                "earliest_date_for_full_payment falls before request_date"
            )

    _validate_spending_changes(result, flexible_event_ids, report)

    if report.violations:
        logger.warning(
            "Validation failed for %s: %s", request.request_id, "; ".join(report.violations)
        )

    return report


def safe_fallback_result() -> PlanResult:
    """A conservative, always-schema-valid result used when validation fails."""
    return PlanResult(
        amount_safe_to_pay=0.0,
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment=None,
        spending_changes_needed="none",
    )
