"""Typed data structures for the Buy or Wait? pipeline.

Every entity loaded from the dataset CSVs is represented as a dataclass with
explicit types. Frozen (immutable) dataclasses are used for entities that
should never be mutated after construction; FinancialEvent is intentionally
mutable so that image-extraction and message-amendment stages can update
amounts, statuses, and dates in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str]
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]
    max_installment_months: Optional[int]


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # debit | credit | non_cash
    amount: Optional[float]
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str  # settled | pending | scheduled | failed | cancelled | unrealized
    linked_event_id: Optional[str]
    flexibility: str  # fixed | reducible | stoppable | reducible_or_stoppable
    minimum_allowed_amount: Optional[float]
    amount_source: str = "csv"  # csv | image_extraction | recurrence_projection
    is_projected: bool = False


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    image_path: Path


@dataclass(frozen=True)
class FinancialRequest:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass
class DataBundle:
    """Aggregate container with O(1) indexed lookups across all dataset files."""

    requests: List[FinancialRequest]
    profiles_by_user: Dict[str, UserProfile]
    events_by_user: Dict[str, List[FinancialEvent]] = field(default_factory=dict)
    events_by_id: Dict[str, FinancialEvent] = field(default_factory=dict)
    payment_options_by_request: Dict[str, List[PaymentOption]] = field(default_factory=dict)
    messages_by_user: Dict[str, List[Message]] = field(default_factory=dict)
    messages_by_request: Dict[str, List[Message]] = field(default_factory=dict)
    messages_by_related_event: Dict[str, List[Message]] = field(default_factory=dict)
    images_by_request: Dict[str, List[ImageRef]] = field(default_factory=dict)
    images_by_related_event: Dict[str, ImageRef] = field(default_factory=dict)
    exchange_rates: Dict[Tuple[date, str, str], float] = field(default_factory=dict)
