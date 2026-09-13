"""Selects the safest, spec-compliant payment plan for one request.

Builds every eligible candidate plan (full payment, partial payment,
installments matching a supplied option, and spending-change-adjusted
variants), keeps only the ones that pass the 90-day safety check, and
applies the exact six-rule tie-break order from the problem statement to
pick a single winner.

Spending-change candidates always cite a real, non-projected event_id as
the reference for a flexible (category, direction), since a synthetic
projected id would not be a valid, verifiable identifier.  The recoverable
amount is applied to every future occurrence of that category within the
forecast horizon (real or projected), because stopping or reducing a
recurring expense is an ongoing decision, not a one-time edit to a single
transaction.

max_installment_months enforcement: installment options whose number of
payments exceeds the user's max_installment_months preference are rejected
before the safety check.
"""

from __future__ import annotations

import copy
from itertools import combinations, product
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from fx import CurrencyConverter
from forecast_engine import (
    BalanceTimeline,
    build_balance_timeline,
    earliest_safe_full_payment_date,
    max_safe_payment_on,
)
from models import FinancialEvent, FinancialRequest, PaymentOption, UserProfile

FLEXIBLE_STATES = {"reducible", "stoppable", "reducible_or_stoppable"}


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CandidatePlan:
    method: str
    payments: List[Tuple[date, float]]
    completes_by_deadline: bool
    requires_spending_changes: bool
    spending_changes: List[str] = field(default_factory=list)
    payment_option_id: Optional[str] = None

    @property
    def total_paid(self) -> float:
        return sum(amount for _, amount in self.payments)

    @property
    def first_payment_date(self) -> Optional[date]:
        return self.payments[0][0] if self.payments else None

    @property
    def sort_key(self):
        """Six-rule tie-break as specified in the problem statement."""
        return (
            0 if self.completes_by_deadline else 1,         # 1. meets deadline
            0 if not self.requires_spending_changes else 1, # 2. no spending changes
            round(self.total_paid, 2),                       # 3. minimize total cost
            self.first_payment_date or date.max,             # 4. start earlier
            len(self.payments),                              # 5. fewer payments
            self.payment_option_id or "",                    # 6. lowest option id
        )


@dataclass
class PlanResult:
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: Optional[date]
    spending_changes_needed: str


# ═══════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════

def _format_amount(amount: float) -> str:
    if abs(amount - round(amount)) < 1e-6:
        return str(int(round(amount)))
    return f"{amount:.2f}"


def _format_plan(payments: List[Tuple[date, float]]) -> str:
    if not payments:
        return "none"
    return "|".join(f"{d.isoformat()}:{_format_amount(amount)}" for d, amount in payments)


def _format_spending_changes(changes: List[str]) -> str:
    return "|".join(changes) if changes else "none"


# ═══════════════════════════════════════════════════════════════════════
# Installment option eligibility helpers
# ═══════════════════════════════════════════════════════════════════════

def _installment_duration_months(option: PaymentOption) -> int:
    """Returns the number of calendar months spanned by the option."""
    if option.number_of_payments <= 1:
        return 1
    frequency = option.payment_frequency_days or 30
    total_days = frequency * (option.number_of_payments - 1)
    return math.ceil(total_days / 30)


def _option_within_max_months(option: PaymentOption, max_months: Optional[int]) -> bool:
    if max_months is None:
        return False  # user does not consider installments
    return _installment_duration_months(option) <= max_months


def _installment_schedule(option: PaymentOption) -> List[Tuple[date, float]]:
    frequency = option.payment_frequency_days or 0
    return [
        (option.first_payment_date + timedelta(days=frequency * i), option.payment_amount)
        for i in range(option.number_of_payments)
    ]


def _is_schedule_safe(
    schedule: List[Tuple[date, float]],
    timeline: BalanceTimeline,
    minimum_balance_to_keep: float,
    horizon_end: date,
) -> bool:
    """Returns True if every payment in the schedule keeps the balance
    above the minimum through the end of the horizon.
    """
    running_deduction = 0.0
    for payment_date, amount in schedule:
        running_deduction += amount
        floor_after = timeline.min_balance_from(payment_date, horizon_end) - running_deduction
        if floor_after < minimum_balance_to_keep:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# Candidate builders
# ═══════════════════════════════════════════════════════════════════════

def _build_full_payment_candidate(
    request: FinancialRequest, profile: UserProfile, timeline: BalanceTimeline
) -> Optional[CandidatePlan]:
    if "full_payment" not in profile.payment_methods_user_will_consider:
        return None
    safe_today = max_safe_payment_on(
        timeline, request.request_date,
        timeline.dates[-1] if timeline.dates else request.request_date,
        profile.minimum_balance_to_keep, request.requested_amount,
    )
    if safe_today < request.requested_amount:
        return None
    return CandidatePlan(
        method="full_payment",
        payments=[(request.request_date, request.requested_amount)],
        completes_by_deadline=request.request_date <= request.desired_completion_date,
        requires_spending_changes=False,
    )


def _build_partial_payment_candidate(
    request: FinancialRequest,
    profile: UserProfile,
    timeline: BalanceTimeline,
    horizon_end: date,
) -> Optional[CandidatePlan]:
    if not request.allows_partial_payment:
        return None
    if "partial_payment" not in profile.payment_methods_user_will_consider:
        return None

    safe_today = max_safe_payment_on(
        timeline, request.request_date, horizon_end,
        profile.minimum_balance_to_keep, request.requested_amount,
    )
    if not (0 < safe_today < request.requested_amount):
        return None

    remaining = request.requested_amount - safe_today
    earliest_full = earliest_safe_full_payment_date(
        timeline, request.request_date, horizon_end,
        profile.minimum_balance_to_keep, request.requested_amount,
    )
    if earliest_full is None or earliest_full > request.desired_completion_date:
        return None

    return CandidatePlan(
        method="partial_payment",
        payments=[(request.request_date, safe_today), (earliest_full, remaining)],
        completes_by_deadline=earliest_full <= request.desired_completion_date,
        requires_spending_changes=False,
    )


def _build_installment_candidates(
    request: FinancialRequest,
    profile: UserProfile,
    timeline: BalanceTimeline,
    payment_options: List[PaymentOption],
    horizon_end: date,
) -> List[CandidatePlan]:
    if "installments" not in profile.payment_methods_user_will_consider:
        return []

    candidates: List[CandidatePlan] = []
    for option in payment_options:
        if option.payment_method != "installments":
            continue
        # Enforce max_installment_months preference
        if not _option_within_max_months(option, profile.max_installment_months):
            continue
        schedule = _installment_schedule(option)
        last_payment_date = schedule[-1][0] if schedule else option.first_payment_date
        if last_payment_date > request.desired_completion_date:
            continue
        if not _is_schedule_safe(schedule, timeline, profile.minimum_balance_to_keep, horizon_end):
            continue
        candidates.append(
            CandidatePlan(
                method="installments",
                payments=schedule,
                completes_by_deadline=True,
                requires_spending_changes=False,
                payment_option_id=option.payment_option_id,
            )
        )
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Spending-change machinery
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FlexibleReference:
    reference_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    reference_amount: float
    future_event_ids: List[str]
    category: str


def _flexible_reference_map(
    events: List[FinancialEvent],
    profile: UserProfile,
    request_date: date,
    horizon_end: date,
) -> Dict[Tuple[str, str], FlexibleReference]:
    """Builds one recoverable-spending reference per flexible (category,
    direction).  The cited event_id is always a real, non-projected event -
    the most recent one with that category - so spending_changes_needed
    never references a synthetic id.

    Only categories the user has explicitly permitted to reduce or stop are
    included.  Protected categories are excluded entirely.
    """
    stoppable_cats = set(profile.expense_categories_user_is_willing_to_stop)
    reducible_cats = set(profile.expense_categories_user_is_willing_to_reduce)
    protected_cats = set(profile.expense_categories_to_protect)

    real_flexible_by_key: Dict[Tuple[str, str], FinancialEvent] = {}
    future_ids_by_key: Dict[Tuple[str, str], List[str]] = {}

    for event in events:
        if event.flexibility not in FLEXIBLE_STATES:
            continue
        if event.status not in ("settled", "scheduled"):
            continue
        if event.category in protected_cats:
            continue
        # Verify user permits the action for this category
        can_stop = (
            event.flexibility in ("stoppable", "reducible_or_stoppable")
            and event.category in stoppable_cats
        )
        can_reduce = (
            event.flexibility in ("reducible", "reducible_or_stoppable")
            and event.category in reducible_cats
        )
        if not can_stop and not can_reduce:
            continue

        key = (event.category, event.direction)

        if not event.is_projected:
            current = real_flexible_by_key.get(key)
            if current is None or event.event_date > current.event_date:
                real_flexible_by_key[key] = event

        if request_date <= _date_for(event) <= horizon_end:
            future_ids_by_key.setdefault(key, []).append(event.event_id)

    references: Dict[Tuple[str, str], FlexibleReference] = {}
    for key, reference_event in real_flexible_by_key.items():
        if key not in future_ids_by_key:
            continue
        references[key] = FlexibleReference(
            reference_event_id=reference_event.event_id,
            flexibility=reference_event.flexibility,
            minimum_allowed_amount=reference_event.minimum_allowed_amount,
            reference_amount=reference_event.amount or 0.0,
            future_event_ids=future_ids_by_key[key],
            category=key[0],
        )
    return references


def _date_for(event: FinancialEvent) -> date:
    return event.settlement_date or event.event_date


def _recoverable_amount(reference: FlexibleReference, profile: UserProfile) -> Tuple[str, float]:
    """Returns (action, recoverable_per_occurrence) respecting user preferences."""
    stoppable_cats = set(profile.expense_categories_user_is_willing_to_stop)
    reducible_cats = set(profile.expense_categories_user_is_willing_to_reduce)

    can_stop = reference.flexibility in ("stoppable", "reducible_or_stoppable") and reference.category in stoppable_cats
    can_reduce = reference.flexibility in ("reducible", "reducible_or_stoppable") and reference.category in reducible_cats

    if can_stop:
        return "stop", reference.reference_amount
    if can_reduce:
        floor = reference.minimum_allowed_amount or 0.0
        return "reduce", max(0.0, reference.reference_amount - floor)
    return "stop", reference.reference_amount


def _apply_spending_change(
    working_events: Dict[str, FinancialEvent],
    reference: FlexibleReference,
    action: str,
) -> float:
    new_amount = (reference.minimum_allowed_amount or 0.0) if action == "reduce" else 0.0
    for event_id in reference.future_event_ids:
        if event_id in working_events:
            working_events[event_id] = copy.deepcopy(working_events[event_id])
            working_events[event_id].amount = new_amount
    return new_amount


def _change_options_for_reference(
    reference: FlexibleReference, profile: UserProfile
) -> List[Tuple[str, float]]:
    stoppable_cats = set(profile.expense_categories_user_is_willing_to_stop)
    reducible_cats = set(profile.expense_categories_user_is_willing_to_reduce)

    can_stop = reference.flexibility in ("stoppable", "reducible_or_stoppable") and reference.category in stoppable_cats
    can_reduce = reference.flexibility in ("reducible", "reducible_or_stoppable") and reference.category in reducible_cats

    options = []
    if can_stop:
        options.append(("stop", 0.0))
    if can_reduce:
        min_amt = reference.minimum_allowed_amount or 0.0
        if min_amt < reference.reference_amount:
            options.append(("reduce", min_amt))
    return options


def _change_combinations(
    references: Dict[Tuple[str, str], FlexibleReference], profile: UserProfile
) -> List[List[Tuple[FlexibleReference, str, float]]]:
    ref_list = list(references.values())
    all_combos = []
    for count in range(1, min(3, len(ref_list)) + 1):
        for selected in combinations(ref_list, count):
            item_choices = []
            for ref in selected:
                opts = _change_options_for_reference(ref, profile)
                if opts:
                    item_choices.append([(ref, action, amt) for action, amt in opts])
            if len(item_choices) == count:
                for combo in product(*item_choices):
                    all_combos.append(list(combo))
    return all_combos


def _try_with_spending_changes(
    request: FinancialRequest,
    profile: UserProfile,
    base_events: List[FinancialEvent],
    fx: CurrencyConverter,
    payment_options: List[PaymentOption],
    horizon_end: date,
    max_changes: int = 3,
) -> Tuple[Optional[CandidatePlan], List[str]]:
    """Evaluates valid combinations of up to 3 flexible spending changes,
    finding the optimal plan per official ranking rules.
    """
    references = _flexible_reference_map(base_events, profile, request.request_date, horizon_end)
    combos = _change_combinations(references, profile)

    valid_plans: List[Tuple[CandidatePlan, List[str]]] = []

    for combo in combos:
        working_events = {e.event_id: copy.deepcopy(e) for e in base_events}
        applied_changes: List[str] = []

        for ref, action, new_amount in combo:
            _apply_spending_change(working_events, ref, action)
            applied_changes.append(
                f"stop:{ref.reference_event_id}" if action == "stop"
                else f"reduce_to:{ref.reference_event_id}:{_format_amount(new_amount)}"
            )

        timeline = build_balance_timeline(
            list(working_events.values()), profile.current_available_balance,
            profile.home_currency, fx, request.request_date, horizon_end,
        )

        full_candidate = _build_full_payment_candidate(request, profile, timeline)
        if full_candidate is not None and full_candidate.completes_by_deadline:
            full_candidate.requires_spending_changes = True
            full_candidate.spending_changes = list(applied_changes)
            valid_plans.append((full_candidate, applied_changes))

        partial_candidate = _build_partial_payment_candidate(request, profile, timeline, horizon_end)
        if partial_candidate is not None and partial_candidate.completes_by_deadline:
            partial_candidate.requires_spending_changes = True
            partial_candidate.spending_changes = list(applied_changes)
            valid_plans.append((partial_candidate, applied_changes))

        for inst in _build_installment_candidates(request, profile, timeline, payment_options, horizon_end):
            if inst.completes_by_deadline:
                inst.requires_spending_changes = True
                inst.spending_changes = list(applied_changes)
                valid_plans.append((inst, applied_changes))

    if not valid_plans:
        return None, []

    # Sort plans using sort_key and prefer fewer changes
    best_plan, best_changes = min(valid_plans, key=lambda pair: (pair[0].sort_key, len(pair[1])))
    return best_plan, best_changes


# ═══════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════

def select_plan(
    request: FinancialRequest,
    profile: UserProfile,
    events: List[FinancialEvent],
    payment_options: List[PaymentOption],
    fx: CurrencyConverter,
    horizon_end: date,
) -> PlanResult:
    """Evaluates all eligible payment plans and returns the best one per the
    six-rule tie-break defined in the problem specification.
    """
    timeline = build_balance_timeline(
        events, profile.current_available_balance, profile.home_currency, fx,
        request.request_date, horizon_end,
    )

    amount_safe_to_pay = max_safe_payment_on(
        timeline, request.request_date, horizon_end,
        profile.minimum_balance_to_keep, request.requested_amount,
    )
    earliest_full = earliest_safe_full_payment_date(
        timeline, request.request_date, horizon_end,
        profile.minimum_balance_to_keep, request.requested_amount,
    )

    # Build all candidate plans without spending changes
    candidates: List[CandidatePlan] = []
    full_candidate = _build_full_payment_candidate(request, profile, timeline)
    if full_candidate:
        candidates.append(full_candidate)
    partial_candidate = _build_partial_payment_candidate(request, profile, timeline, horizon_end)
    if partial_candidate:
        candidates.append(partial_candidate)
    candidates.extend(
        _build_installment_candidates(request, profile, timeline, payment_options, horizon_end)
    )

    on_deadline_candidates = [c for c in candidates if c.completes_by_deadline]

    # ── Decision logic ─────────────────────────────────────────────────
    if full_candidate and full_candidate.first_payment_date == request.request_date:
        # Full amount safe today: affordable_now
        winner = full_candidate
        status = "affordable_now"

    elif on_deadline_candidates:
        # At least one plan completes by the deadline
        winner = min(on_deadline_candidates, key=lambda c: c.sort_key)
        status = "affordable_with_plan"

    else:
        # Try spending changes
        spending_change_candidate, _ = _try_with_spending_changes(
            request, profile, events, fx, payment_options, horizon_end
        )
        if spending_change_candidate and spending_change_candidate.completes_by_deadline:
            winner = spending_change_candidate
            status = "affordable_with_plan"

        elif "full_payment" in profile.payment_methods_user_will_consider and earliest_full is not None:
            # Full payment becomes safe later within the horizon — show the user
            # when they can come back and pay the full amount.
            winner = CandidatePlan(
                method="wait",
                payments=[(earliest_full, request.requested_amount)],
                completes_by_deadline=earliest_full <= request.desired_completion_date,
                requires_spending_changes=False,
            )
            status = "affordable_later"

        else:
            # Nothing works within the forecast period
            winner = CandidatePlan(
                method="not_recommended",
                payments=[],
                completes_by_deadline=False,
                requires_spending_changes=False,
            )
            status = "not_affordable"

    # Determine earliest_date_for_full_payment based on status and method
    if status == "affordable_now":
        result_earliest = request.request_date
    elif status == "not_affordable":
        # Per spec: empty when no full payment is safe within the forecast period
        result_earliest = None
    elif winner.method == "installments" and winner.payments:
        # For installments, this is the date the last installment completes
        # (when the full requested amount has finished being paid)
        last_installment_date = winner.payments[-1][0]
        # Also surface when a single full payment would first become safe,
        # but use whichever is earlier
        result_earliest = (
            earliest_full
            if earliest_full is not None and earliest_full <= last_installment_date
            else last_installment_date
        )
    else:
        result_earliest = earliest_full

    return PlanResult(
        amount_safe_to_pay=round(amount_safe_to_pay, 2),
        affordability_status=status,
        recommended_payment_method=winner.method,
        payment_plan=_format_plan(winner.payments),
        earliest_date_for_full_payment=result_earliest,
        spending_changes_needed=_format_spending_changes(winner.spending_changes),
    )
