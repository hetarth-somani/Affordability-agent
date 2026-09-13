"""Generates the decision_explanation field for one validated result.

The model is given only pre-computed facts and is required to write a short
explanation from them. The output is then verified twice: every number and
date must appear in that fact set, and the wording must not assert a
relationship the result contradicts (for example claiming the full requested
amount is affordable when only part of it is). Failing either check, the row
falls back to a deterministic template built from the same facts, so an
unsupported figure or claim is structurally impossible to ship rather than
merely discouraged by prompt wording.
"""

from __future__ import annotations

import logging
import re
from typing import List

from cache import DiskCache
from llm_client import call_text_json
from models import FinancialRequest, UserProfile
from plan_ranker import PlanResult

logger = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
MAX_EXPLANATION_CHARS = 300

EXPLANATION_SYSTEM_PROMPT = (
    "You write one short explanation for a financial affordability decision. "
    "Write two sentences at most, in plain English, in the third person about "
    "the user's situation.\n\n"
    "Strict rules:\n"
    "- Use ONLY the numbers, dates, and identifiers present in the facts given "
    "to you. Never introduce a figure or date that is not in the facts.\n"
    "- Always write amounts with the currency CODE exactly as given in the "
    "facts (for example 'EUR 1302.40'). Never use a currency symbol.\n"
    "- Write in plain language. Never use the raw field names from the facts "
    "(such as amount_safe_to_pay or recommended_payment_method) in your text.\n"
    "- State only relationships that the facts directly support. Do not claim "
    "the requested amount is affordable, or that the safe amount equals the "
    "request, unless those two figures are actually the same.\n"
    "- Describe what the recommendation is and the main financial reason for "
    "it, referring to the minimum balance the user keeps.\n"
    "- Do not speculate about anything not stated in the facts.\n\n"
    'Respond with a single JSON object: {"explanation": "<text>"}'
)

CONTRADICTION_PATTERNS = (
    re.compile(r"\bcan afford the (?:requested|full)\b", re.IGNORECASE),
    re.compile(r"\bequals? the request(?:ed amount)?\b", re.IGNORECASE),
    re.compile(r"\bmatches the requested\b", re.IGNORECASE),
    re.compile(r"\bfully affordable\b", re.IGNORECASE),
)

RAW_FIELD_NAMES = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "minimum_balance_to_keep",
    "current_available_balance",
    "requested_amount",
    "full_payment",
    "partial_payment",
    "not_recommended",
)


def _canonical_numbers(values: List[str]) -> set[str]:
    """Returns a set of number strings normalised for comparison, including
    both the plain and thousands-separated forms of each value.
    """
    canonical: set[str] = set()
    for value in values:
        stripped = value.replace(",", "")
        canonical.add(stripped)
        try:
            numeric = float(stripped)
        except ValueError:
            continue
        canonical.add(f"{numeric:.2f}")
        canonical.add(f"{numeric:,.2f}")
        if abs(numeric - round(numeric)) < 1e-6:
            canonical.add(str(int(round(numeric))))
            canonical.add(f"{int(round(numeric)):,}")
    return canonical


def _build_fact_lines(
    request: FinancialRequest, profile: UserProfile, result: PlanResult
) -> List[str]:
    lines = [
        f"currency: {profile.home_currency}",
        f"requested_amount: {request.requested_amount}",
        f"request_date: {request.request_date.isoformat()}",
        f"desired_completion_date: {request.desired_completion_date.isoformat()}",
        f"current_available_balance: {profile.current_available_balance}",
        f"minimum_balance_to_keep: {profile.minimum_balance_to_keep}",
        f"amount_safe_to_pay: {result.amount_safe_to_pay}",
        f"affordability_status: {result.affordability_status}",
        f"recommended_payment_method: {result.recommended_payment_method}",
        f"payment_plan: {result.payment_plan}",
        f"spending_changes_needed: {result.spending_changes_needed}",
    ]
    if result.earliest_date_for_full_payment is not None:
        lines.append(
            f"earliest_date_for_full_payment: {result.earliest_date_for_full_payment.isoformat()}"
        )
    return lines


def _supported_tokens(fact_lines: List[str]) -> set[str]:
    raw_numbers: List[str] = []
    for line in fact_lines:
        raw_numbers.extend(NUMBER_RE.findall(line))
    return _canonical_numbers(raw_numbers)


def _is_grounded(explanation: str, supported: set[str]) -> bool:
    for token in NUMBER_RE.findall(explanation):
        normalised = token.replace(",", "")
        if normalised in supported or token in supported:
            continue
        try:
            numeric = float(normalised)
        except ValueError:
            return False
        if f"{numeric:.2f}" in supported or str(int(numeric)) in supported:
            continue
        return False
    return True


def _makes_unsupported_claims(
    explanation: str, request: FinancialRequest, result: PlanResult
) -> bool:
    """Rejects explanations that assert a relationship the result contradicts,
    or that leak raw schema field names instead of plain language.
    """
    lowered = explanation.lower()

    if any(field_name in lowered for field_name in RAW_FIELD_NAMES):
        return True

    covers_full_amount = abs(result.amount_safe_to_pay - request.requested_amount) < 0.01
    if not covers_full_amount:
        if any(pattern.search(explanation) for pattern in CONTRADICTION_PATTERNS):
            return True

    return False


def _template_explanation(
    request: FinancialRequest, profile: UserProfile, result: PlanResult
) -> str:
    currency = profile.home_currency
    method = result.recommended_payment_method

    if method == "full_payment":
        body = (
            f"Pay the full {currency} {request.requested_amount:,.2f} on "
            f"{request.request_date.isoformat()}."
        )
    elif method == "partial_payment":
        body = (
            f"Pay {currency} {result.amount_safe_to_pay:,.2f} now and the remainder on "
            f"{result.earliest_date_for_full_payment.isoformat()}."
        )
    elif method == "installments":
        body = f"Use the installment schedule {result.payment_plan}."
    elif method == "wait":
        earliest = (
            result.earliest_date_for_full_payment.isoformat()
            if result.earliest_date_for_full_payment
            else "a later date"
        )
        body = f"Wait until {earliest}, when the full amount becomes safe to pay."
    else:
        body = "Do not proceed with this request within the forecast period."

    reason = (
        f"At most {currency} {result.amount_safe_to_pay:,.2f} can be paid on "
        f"{request.request_date.isoformat()} while keeping the balance above the "
        f"{currency} {profile.minimum_balance_to_keep:,.2f} minimum."
    )

    if result.spending_changes_needed != "none":
        reason += f" This assumes the spending changes {result.spending_changes_needed}."

    return f"{body} {reason}"


def generate_explanation(
    request: FinancialRequest,
    profile: UserProfile,
    result: PlanResult,
    cache: DiskCache,
) -> str:
    """Returns a grounded explanation, falling back to a deterministic
    template if the model output is missing, introduces unsupported values,
    or asserts a claim the computed result contradicts.
    """
    fact_lines = _build_fact_lines(request, profile, result)
    cache_key = {"kind": "explanation", "request_id": request.request_id, "facts": fact_lines}

    cached = cache.get(cache_key)
    if cached is None:
        user_prompt = "Facts:\n" + "\n".join(fact_lines)
        cached = call_text_json(EXPLANATION_SYSTEM_PROMPT, user_prompt, max_tokens=700)
        if cached is not None:
            cache.set(cache_key, cached)

    if cached is None:
        logger.warning("No model explanation for %s; using template", request.request_id)
        return _template_explanation(request, profile, result)

    explanation = str(cached.get("explanation", "")).strip()
    if not explanation:
        logger.warning("Empty model explanation for %s; using template", request.request_id)
        return _template_explanation(request, profile, result)

    if not _is_grounded(explanation, _supported_tokens(fact_lines)):
        logger.warning(
            "Model explanation for %s contained unsupported figures; using template",
            request.request_id,
        )
        return _template_explanation(request, profile, result)

    if _makes_unsupported_claims(explanation, request, result):
        logger.warning(
            "Model explanation for %s made a claim the result contradicts; using template",
            request.request_id,
        )
        return _template_explanation(request, profile, result)

    return explanation[:MAX_EXPLANATION_CHARS]
