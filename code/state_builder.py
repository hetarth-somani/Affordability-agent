"""Builds each user's resolved, forecast-ready event timeline.

Three responsibilities, applied in order:
1. Conflict resolution: deduplicate repeated representations of the same
   event, then apply message-derived cancellations and amendments.
2. Recurrence detection: identify (category, direction) pairs that repeat
   on a roughly regular interval, isolating the dominant cadence from
   unrelated one-off events sharing the same category.
3. Forward projection: for recurring patterns with no explicit future row
   inside the forecast horizon, synthesize projected occurrences.  At each
   expected slot, a real future event of the same (category, direction)
   whose effective date falls within a tolerance window of that slot is
   treated as the real occurrence for that period: it is already present
   in the cash-flow data, so no synthetic duplicate is created there, and
   projection continues from its date.  Monthly cadences are projected using
   calendar-month arithmetic on the anchor day-of-month rather than a
   fixed day-count step, since a fixed step drifts away from the true
   monthly date over multiple cycles.  Projected events carry over the real
   flexibility and minimum_allowed_amount of the pattern they were derived
   from, and are always marked ``is_projected=True`` so downstream code can
   distinguish inference from committed data.
"""

from __future__ import annotations

import calendar
import copy
import logging
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from evidence_agent import MessageFact
from models import DataBundle, FinancialEvent

logger = logging.getLogger(__name__)

# ── Deduplication constants ────────────────────────────────────────────
DEDUP_WINDOW_DAYS = 5
DEDUP_AMOUNT_TOLERANCE = 0.01

# ── Recurrence detection constants ─────────────────────────────────────
MIN_OCCURRENCES_FOR_RECURRENCE = 3
INTERVAL_TOLERANCE_RATIO = 0.4
AMOUNT_HISTORY_WINDOW = 3

# ── Projection constants ──────────────────────────────────────────────
MONTHLY_GAP_MIN = 27
MONTHLY_GAP_MAX = 31
MAX_PROJECTION_ITERATIONS = 60

# ── Categories / event types that are never recurring ──────────────────
NON_RECURRING_CATEGORIES = {"family_transfer", "refund"}
NON_RECURRING_EVENT_TYPES = {
    "investment_purchase",
    "investment_sale",
    "investment_valuation",
}


def _effective_date(event: FinancialEvent) -> date:
    return event.settlement_date or event.event_date


def _is_amount_close(a: float, b: float, tolerance: float = DEDUP_AMOUNT_TOLERANCE) -> bool:
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance


# ═══════════════════════════════════════════════════════════════════════
# 1.  Deduplication
# ═══════════════════════════════════════════════════════════════════════

def deduplicate_events(events: List[FinancialEvent]) -> List[FinancialEvent]:
    """Drops cancelled events that are a duplicate representation of a
    nearby settled event with the same category, direction, and amount.
    """
    kept: List[FinancialEvent] = []
    dropped_ids: set[str] = set()

    for event in events:
        if event.event_id in dropped_ids or event.status != "cancelled":
            continue
        for other in events:
            if other.event_id == event.event_id or other.status not in ("settled", "scheduled", "pending"):
                continue
            same_shape = (
                other.category == event.category
                and other.direction == event.direction
                and _is_amount_close(other.amount or 0.0, event.amount or 0.0)
            )
            close_in_time = abs((other.event_date - event.event_date).days) <= DEDUP_WINDOW_DAYS
            if same_shape and close_in_time:
                dropped_ids.add(event.event_id)
                logger.info(
                    "Dropping %s as a duplicate of %s (same category/amount, %d days apart)",
                    event.event_id,
                    other.event_id,
                    abs((other.event_date - event.event_date).days),
                )
                break

    return [event for event in events if event.event_id not in dropped_ids]


# ═══════════════════════════════════════════════════════════════════════
# 2.  Message-fact application
# ═══════════════════════════════════════════════════════════════════════

def apply_message_facts(
    events_by_id: Dict[str, FinancialEvent],
    facts_by_message_id: Dict[str, MessageFact],
    user_id: str,
) -> None:
    """Mutates events in place using verified (dataset-linked) message facts.

    Only facts whose related_event_id resolves to a real event owned by this
    user are ever applied; unresolved facts are logged and otherwise ignored,
    never used to invent a new event.
    """
    for fact in facts_by_message_id.values():
        if fact.related_event_id is None:
            continue
        event = events_by_id.get(fact.related_event_id)
        if event is None or event.user_id != user_id:
            continue

        if fact.signal_type == "cancellation":
            event.status = "cancelled"
            logger.info("Event %s cancelled by message %s", event.event_id, fact.message_id)

        elif fact.signal_type in ("income_change", "expense_change") and fact.new_amount is not None:
            event.amount = float(fact.new_amount)
            if fact.new_currency:
                event.currency = fact.new_currency
            logger.info(
                "Event %s amount amended to %s %s by message %s",
                event.event_id,
                event.amount,
                event.currency,
                fact.message_id,
            )

        elif fact.signal_type == "delay" and fact.effective_date is not None:
            new_date = date.fromisoformat(fact.effective_date)
            if event.settlement_date is None or new_date > event.settlement_date:
                event.settlement_date = new_date
                logger.info("Event %s delayed to %s by message %s", event.event_id, new_date, fact.message_id)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Recurrence detection
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RecurrencePattern:
    category: str
    direction: str
    frequency_days: int
    projected_amount: float
    last_known_date: date
    occurrence_count: int
    anchor_day_of_month: Optional[int] = None
    flexibility: str = "fixed"
    minimum_allowed_amount: Optional[float] = None


def _longest_consistent_chain(
    ordered_events: List[FinancialEvent], tolerance_ratio: float
) -> List[FinancialEvent]:
    """Finds the longest run of events whose consecutive gaps stay within
    tolerance of that run's own median gap.  This isolates the dominant
    recurring cadence (for example, a salary paid on the 15th of every
    month) from unrelated one-off events sharing the same category and
    direction (a bonus, an arrears payment, a duplicate entry).
    """
    best_chain: List[FinancialEvent] = []
    n = len(ordered_events)

    for start in range(n):
        chain = [ordered_events[start]]
        i = start
        while i + 1 < n:
            candidate = ordered_events[i + 1]
            gap = (candidate.event_date - chain[-1].event_date).days
            if len(chain) == 1:
                chain.append(candidate)
                i += 1
                continue
            gaps = [
                (chain[j + 1].event_date - chain[j].event_date).days
                for j in range(len(chain) - 1)
            ]
            median_gap = statistics.median(gaps)
            if median_gap > 0 and abs(gap - median_gap) <= median_gap * tolerance_ratio:
                chain.append(candidate)
                i += 1
            else:
                break
        if len(chain) > len(best_chain):
            best_chain = chain

    return best_chain


def detect_recurring_patterns(events: List[FinancialEvent]) -> Dict[Tuple[str, str], RecurrencePattern]:
    """Detects recurring (category, direction) pairs from settled/scheduled
    history by finding the longest internally-consistent chain of events.
    """
    groups: Dict[Tuple[str, str], List[FinancialEvent]] = {}
    for event in events:
        if event.status not in ("settled", "scheduled"):
            continue
        if event.category in NON_RECURRING_CATEGORIES:
            continue
        if event.event_type in NON_RECURRING_EVENT_TYPES:
            continue
        groups.setdefault((event.category, event.direction), []).append(event)

    patterns: Dict[Tuple[str, str], RecurrencePattern] = {}
    for key, group_events in groups.items():
        ordered = sorted(group_events, key=lambda e: e.event_date)
        if len(ordered) < MIN_OCCURRENCES_FOR_RECURRENCE:
            continue

        chain = _longest_consistent_chain(ordered, INTERVAL_TOLERANCE_RATIO)
        if len(chain) < MIN_OCCURRENCES_FOR_RECURRENCE:
            continue

        gaps = [
            (chain[i + 1].event_date - chain[i].event_date).days
            for i in range(len(chain) - 1)
        ]
        median_gap = statistics.median(gaps)
        if median_gap <= 0:
            continue

        recent_amounts = [e.amount for e in chain[-AMOUNT_HISTORY_WINDOW:] if e.amount is not None]
        if not recent_amounts:
            continue

        anchor_day_of_month = (
            chain[-1].event_date.day
            if MONTHLY_GAP_MIN <= median_gap <= MONTHLY_GAP_MAX
            else None
        )

        patterns[key] = RecurrencePattern(
            category=key[0],
            direction=key[1],
            frequency_days=round(median_gap),
            projected_amount=statistics.median(recent_amounts),
            last_known_date=chain[-1].event_date,
            occurrence_count=len(chain),
            anchor_day_of_month=anchor_day_of_month,
            flexibility=chain[-1].flexibility,
            minimum_allowed_amount=chain[-1].minimum_allowed_amount,
        )

    return patterns


# ═══════════════════════════════════════════════════════════════════════
# 4.  Forward projection
# ═══════════════════════════════════════════════════════════════════════

def _add_calendar_months(base_date: date, months: int, anchor_day: int) -> date:
    total_month_index = base_date.month - 1 + months
    year = base_date.year + total_month_index // 12
    month = total_month_index % 12 + 1
    last_day_of_month = calendar.monthrange(year, month)[1]
    return date(year, month, min(anchor_day, last_day_of_month))


def project_recurring_events(
    user_id: str,
    currency: str,
    patterns: Dict[Tuple[str, str], RecurrencePattern],
    real_future_events: Dict[Tuple[str, str], List[FinancialEvent]],
    horizon_start: date,
    horizon_end: date,
) -> List[FinancialEvent]:
    """Synthesizes future occurrences for detected recurring patterns.

    At each expected slot, a real future event of the same (category,
    direction) whose effective date falls within a tolerance window of
    that slot is treated as the real occurrence for that period — no
    synthetic duplicate is created there, and projection continues from
    its date.
    """
    projected: List[FinancialEvent] = []
    counter = 0

    for key, pattern in patterns.items():
        real_events = sorted(real_future_events.get(key, []), key=_effective_date)
        consumed_ids: set[str] = set()

        anchor_date = pattern.last_known_date
        anchor_day = pattern.anchor_day_of_month
        tolerance_days = (
            10 if anchor_day is not None else max(1, pattern.frequency_days // 3)
        )

        iterations = 0
        while iterations < MAX_PROJECTION_ITERATIONS:
            iterations += 1

            if anchor_day is not None:
                candidate_date = _add_calendar_months(anchor_date, 1, anchor_day)
            else:
                candidate_date = anchor_date + timedelta(days=pattern.frequency_days)

            if candidate_date > horizon_end:
                break

            nearby_real_event = next(
                (
                    e for e in real_events
                    if e.event_id not in consumed_ids
                    and abs((_effective_date(e) - candidate_date).days) <= tolerance_days
                ),
                None,
            )

            if nearby_real_event is not None:
                consumed_ids.add(nearby_real_event.event_id)
                anchor_date = candidate_date
                continue

            if candidate_date >= horizon_start:
                counter += 1
                projected.append(
                    FinancialEvent(
                        event_id=f"projected_{user_id}_{pattern.category}_{counter}",
                        user_id=user_id,
                        event_type="expense" if pattern.direction == "debit" else "income",
                        description=f"Projected recurring {pattern.category}",
                        category=pattern.category,
                        direction=pattern.direction,
                        amount=pattern.projected_amount,
                        currency=currency,
                        event_date=candidate_date,
                        settlement_date=candidate_date,
                        status="scheduled",
                        linked_event_id=None,
                        flexibility=pattern.flexibility,
                        minimum_allowed_amount=pattern.minimum_allowed_amount,
                        amount_source="recurrence_projection",
                        is_projected=True,
                    )
                )
            anchor_date = candidate_date

    return projected


# ═══════════════════════════════════════════════════════════════════════
# 5.  Public entry point
# ═══════════════════════════════════════════════════════════════════════

def build_forecastable_events(
    data: DataBundle,
    user_id: str,
    message_facts: Dict[str, MessageFact],
    horizon_start: date,
    horizon_end: date,
) -> List[FinancialEvent]:
    """Returns the full set of events to feed the 90-day forecast for one
    user: resolved historical/committed events plus projected recurring
    occurrences, with pending credits and cancelled/failed events excluded
    per the safety-check rules.
    """
    raw_events = [copy.deepcopy(e) for e in data.events_by_user.get(user_id, [])]
    events_by_id = {e.event_id: e for e in raw_events}

    # Apply verified message-derived amendments
    user_facts = {
        message_id: fact
        for message_id, fact in message_facts.items()
        if fact.related_event_id in events_by_id
    }
    apply_message_facts(events_by_id, user_facts, user_id)

    # Deduplicate cancelled/settled pairs
    deduped = deduplicate_events(list(events_by_id.values()))

    # Filter to cash-flow-relevant events only
    usable_for_cash_flow = [
        e for e in deduped
        if e.status not in ("cancelled", "failed")
        and not (e.status == "pending" and e.direction == "credit")
        and e.direction != "non_cash"
    ]

    # Detect and project recurring patterns
    patterns = detect_recurring_patterns(deduped)

    real_future_events_by_key: Dict[Tuple[str, str], List[FinancialEvent]] = {}
    for e in usable_for_cash_flow:
        if _effective_date(e) >= horizon_start:
            real_future_events_by_key.setdefault((e.category, e.direction), []).append(e)

    profile = data.profiles_by_user[user_id]
    projected = project_recurring_events(
        user_id=user_id,
        currency=profile.home_currency,
        patterns=patterns,
        real_future_events=real_future_events_by_key,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )

    return usable_for_cash_flow + projected
