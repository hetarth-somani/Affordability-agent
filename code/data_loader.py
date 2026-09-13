"""Loads and indexes all dataset CSV files into typed, queryable structures.

Every CSV is parsed once into strongly-typed dataclass instances and indexed
by primary and foreign keys for O(1) lookups during per-request processing.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from models import (
    DataBundle,
    FinancialEvent,
    FinancialRequest,
    ImageRef,
    Message,
    PaymentOption,
    UserProfile,
)


def _parse_pipe_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _parse_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def _parse_optional_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _parse_optional_int(value) -> Optional[int]:
    parsed = _parse_optional_float(value)
    return int(parsed) if parsed is not None else None


def _parse_optional_str(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text if text else None


def _parse_date(value) -> date:
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()


def _parse_optional_date(value) -> Optional[date]:
    text = _parse_optional_str(value)
    return _parse_date(text) if text else None


def _parse_datetime(value) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def load_all(dataset_dir: Path, media_dir: Optional[Path] = None) -> DataBundle:
    """Loads every dataset CSV and builds fully indexed lookup structures."""

    media_dir = media_dir or (dataset_dir / "media" / "images")

    profiles_df = pd.read_csv(dataset_dir / "financial_profiles.csv", dtype=str)
    events_df = pd.read_csv(dataset_dir / "financial_events.csv", dtype=str)
    payment_options_df = pd.read_csv(dataset_dir / "request_payment_options.csv", dtype=str)
    messages_df = pd.read_csv(dataset_dir / "messages.csv", dtype=str)
    images_df = pd.read_csv(dataset_dir / "images.csv", dtype=str)
    requests_df = pd.read_csv(dataset_dir / "requests.csv", dtype=str)
    rates_df = pd.read_csv(dataset_dir / "exchange_rates.csv", dtype=str)

    # ── User profiles ──────────────────────────────────────────────────
    profiles_by_user: Dict[str, UserProfile] = {}
    for row in profiles_df.itertuples(index=False):
        profiles_by_user[row.user_id] = UserProfile(
            user_id=row.user_id,
            home_currency=row.home_currency,
            current_available_balance=float(row.current_available_balance),
            minimum_balance_to_keep=float(row.minimum_balance_to_keep),
            financial_priorities=_parse_pipe_list(row.financial_priorities),
            expense_categories_to_protect=_parse_pipe_list(row.expense_categories_to_protect),
            expense_categories_user_is_willing_to_reduce=_parse_pipe_list(
                row.expense_categories_user_is_willing_to_reduce
            ),
            expense_categories_user_is_willing_to_stop=_parse_pipe_list(
                row.expense_categories_user_is_willing_to_stop
            ),
            payment_methods_user_will_consider=_parse_pipe_list(
                row.payment_methods_user_will_consider
            ),
            max_installment_months=_parse_optional_int(row.max_installment_months),
        )

    # ── Financial events ───────────────────────────────────────────────
    events_by_user: Dict[str, List[FinancialEvent]] = {}
    events_by_id: Dict[str, FinancialEvent] = {}
    for row in events_df.itertuples(index=False):
        event = FinancialEvent(
            event_id=row.event_id,
            user_id=row.user_id,
            event_type=row.event_type,
            description=row.description,
            category=row.category,
            direction=row.direction,
            amount=_parse_optional_float(row.amount),
            currency=row.currency,
            event_date=_parse_date(row.event_date),
            settlement_date=_parse_optional_date(row.settlement_date),
            status=row.status,
            linked_event_id=_parse_optional_str(row.linked_event_id),
            flexibility=row.flexibility,
            minimum_allowed_amount=_parse_optional_float(row.minimum_allowed_amount),
        )
        events_by_id[event.event_id] = event
        events_by_user.setdefault(event.user_id, []).append(event)

    # ── Payment options ────────────────────────────────────────────────
    payment_options_by_request: Dict[str, List[PaymentOption]] = {}
    for row in payment_options_df.itertuples(index=False):
        option = PaymentOption(
            payment_option_id=row.payment_option_id,
            request_id=row.request_id,
            payment_method=row.payment_method,
            payment_amount=float(row.payment_amount),
            number_of_payments=int(row.number_of_payments),
            first_payment_date=_parse_date(row.first_payment_date),
            payment_frequency_days=_parse_optional_int(row.payment_frequency_days),
            financing_fee=float(row.financing_fee),
            total_payable_amount=float(row.total_payable_amount),
        )
        payment_options_by_request.setdefault(option.request_id, []).append(option)

    # ── Messages ───────────────────────────────────────────────────────
    messages_by_user: Dict[str, List[Message]] = {}
    messages_by_request: Dict[str, List[Message]] = {}
    messages_by_related_event: Dict[str, List[Message]] = {}
    for row in messages_df.itertuples(index=False):
        message = Message(
            message_id=row.message_id,
            user_id=row.user_id,
            request_id=_parse_optional_str(row.request_id),
            related_event_id=_parse_optional_str(row.related_event_id),
            sent_at=_parse_datetime(row.sent_at),
            source_type=row.source_type,
            message_text=row.message_text,
        )
        messages_by_user.setdefault(message.user_id, []).append(message)
        if message.request_id:
            messages_by_request.setdefault(message.request_id, []).append(message)
        if message.related_event_id:
            messages_by_related_event.setdefault(message.related_event_id, []).append(message)

    # ── Images ─────────────────────────────────────────────────────────
    images_by_request: Dict[str, List[ImageRef]] = {}
    images_by_related_event: Dict[str, ImageRef] = {}
    for row in images_df.itertuples(index=False):
        image = ImageRef(
            image_id=row.image_id,
            user_id=row.user_id,
            request_id=_parse_optional_str(row.request_id),
            related_event_id=_parse_optional_str(row.related_event_id),
            image_path=media_dir / f"{row.image_id}.png",
        )
        if image.request_id:
            images_by_request.setdefault(image.request_id, []).append(image)
        if image.related_event_id:
            images_by_related_event[image.related_event_id] = image

    # ── Exchange rates ─────────────────────────────────────────────────
    exchange_rates: Dict[Tuple[date, str, str], float] = {}
    for row in rates_df.itertuples(index=False):
        key = (_parse_date(row.rate_date), row.from_currency, row.to_currency)
        exchange_rates[key] = float(row.rate)

    # ── Requests ───────────────────────────────────────────────────────
    requests: List[FinancialRequest] = [
        FinancialRequest(
            request_id=row.request_id,
            user_id=row.user_id,
            request_date=_parse_date(row.request_date),
            request_type=row.request_type,
            requested_amount=float(row.requested_amount),
            desired_completion_date=_parse_date(row.desired_completion_date),
            allows_partial_payment=_parse_bool(row.allows_partial_payment),
            request_text=row.request_text,
        )
        for row in requests_df.itertuples(index=False)
    ]

    return DataBundle(
        requests=requests,
        profiles_by_user=profiles_by_user,
        events_by_user=events_by_user,
        events_by_id=events_by_id,
        payment_options_by_request=payment_options_by_request,
        messages_by_user=messages_by_user,
        messages_by_request=messages_by_request,
        messages_by_related_event=messages_by_related_event,
        images_by_request=images_by_request,
        images_by_related_event=images_by_related_event,
        exchange_rates=exchange_rates,
    )
