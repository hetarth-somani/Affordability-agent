"""Event-driven 90-day balance forecasting and safety checks.

Balance only changes at event dates, so the full trajectory is represented
as a sorted list of (date, cumulative_balance_in_home_currency) checkpoints.
All amounts are converted to the user's home currency at each event's
effective date before being folded into the running total.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

from fx import CurrencyConverter
from models import FinancialEvent


def _effective_date(event: FinancialEvent) -> date:
    return event.settlement_date or event.event_date


def _signed_home_currency_amount(
    event: FinancialEvent, home_currency: str, fx: CurrencyConverter
) -> float:
    amount = event.amount or 0.0
    converted = fx.convert(amount, event.currency, home_currency, _effective_date(event))
    return converted if event.direction == "credit" else -converted


@dataclass(frozen=True)
class BalanceTimeline:
    """Sorted checkpoints of (date, cumulative_balance) for one user."""

    dates: List[date]
    cumulative_balances: List[float]
    starting_balance: float

    def balance_as_of(self, on_date: date) -> float:
        """Balance at the end of `on_date`, after that day's own events."""
        index = bisect.bisect_right(self.dates, on_date) - 1
        if index < 0:
            return self.starting_balance
        return self.cumulative_balances[index]

    def min_balance_from(self, start_date: date, end_date: date) -> float:
        """Minimum balance over [start_date, end_date], including the level
        already in effect at start_date.
        """
        candidates = [self.balance_as_of(start_date)]
        start_index = bisect.bisect_right(self.dates, start_date)
        end_index = bisect.bisect_right(self.dates, end_date)
        candidates.extend(self.cumulative_balances[start_index:end_index])
        return min(candidates)


def build_balance_timeline(
    events: List[FinancialEvent],
    starting_balance: float,
    home_currency: str,
    fx: CurrencyConverter,
    horizon_start: date,
    horizon_end: date,
) -> BalanceTimeline:
    """Builds a cumulative balance timeline from `horizon_start` to `horizon_end`.

    Events are placed at their effective date (settlement_date if present,
    otherwise event_date).  Multiple events on the same day are collapsed
    into a single end-of-day balance checkpoint.
    """
    in_window = [
        e for e in events
        if horizon_start <= _effective_date(e) <= horizon_end
    ]
    in_window.sort(key=lambda e: (_effective_date(e), 0 if e.direction == "credit" else 1))

    dates: List[date] = []
    cumulative_balances: List[float] = []
    running_total = starting_balance

    for event in in_window:
        running_total += _signed_home_currency_amount(event, home_currency, fx)
        ev_date = _effective_date(event)
        if dates and dates[-1] == ev_date:
            cumulative_balances[-1] = running_total
        else:
            dates.append(ev_date)
            cumulative_balances.append(running_total)

    return BalanceTimeline(
        dates=dates,
        cumulative_balances=cumulative_balances,
        starting_balance=starting_balance,
    )


def max_safe_payment_on(
    timeline: BalanceTimeline,
    payment_date: date,
    horizon_end: date,
    minimum_balance_to_keep: float,
    requested_amount: float,
) -> float:
    """Largest amount payable on `payment_date` that never breaches the
    minimum balance at any point through `horizon_end`.
    """
    floor_balance = timeline.min_balance_from(payment_date, horizon_end)
    safe_amount = floor_balance - minimum_balance_to_keep
    return max(0.0, min(safe_amount, requested_amount))


def earliest_safe_full_payment_date(
    timeline: BalanceTimeline,
    request_date: date,
    horizon_end: date,
    minimum_balance_to_keep: float,
    requested_amount: float,
) -> Optional[date]:
    """First date on or after `request_date` at which paying the full
    requested amount stays safe for the rest of the forecast horizon.
    Returns None if no such date exists within the horizon.
    """
    candidate_dates = [request_date] + [d for d in timeline.dates if d > request_date]
    for candidate in candidate_dates:
        if candidate > horizon_end:
            break
        floor_balance = timeline.min_balance_from(candidate, horizon_end)
        if floor_balance - minimum_balance_to_keep >= requested_amount:
            return candidate
    return None
