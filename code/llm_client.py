"""Thin wrapper around the Groq client with sane defaults, bounded retries,
and per-model token accounting for the required usage report.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env from code/ directory or repo root
_THIS_DIR = Path(__file__).resolve().parent
load_dotenv(_THIS_DIR / ".env")
load_dotenv(_THIS_DIR.parent / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class UsageTracker:
    """Accumulates per-model token usage across the run for the usage report."""

    def __init__(self) -> None:
        self.calls_by_model: Dict[str, int] = {}
        self.input_tokens_by_model: Dict[str, int] = {}
        self.output_tokens_by_model: Dict[str, int] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
        self.input_tokens_by_model[model] = (
            self.input_tokens_by_model.get(model, 0) + input_tokens
        )
        self.output_tokens_by_model[model] = (
            self.output_tokens_by_model.get(model, 0) + output_tokens
        )

    @property
    def total_calls(self) -> int:
        return sum(self.calls_by_model.values())


USAGE = UsageTracker()


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        load_dotenv(_THIS_DIR / ".env")
        api_key = os.environ.get("GROQ_API_KEY")
    return Groq(api_key=api_key)


def _extract_json(text: str) -> Any:
    """Extracts a JSON object from model output, handling markdown fences."""
    fenced = _JSON_FENCE_RE.search(text)
    payload = fenced.group(1) if fenced else text
    return json.loads(payload)


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    USAGE.record(
        getattr(response, "model", "unknown"),
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def call_text_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 900,
    max_retries: int = 3,
) -> Optional[dict]:
    """Calls the text model and parses a single JSON object from the response."""
    client = _client()
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            _record_usage(response)
            content = response.choices[0].message.content
            return _extract_json(content)
        except Exception as exc:
            last_error = exc
            logger.warning("Text call attempt %d/%d failed: %s", attempt, max_retries, exc)
            time.sleep(1.5 * attempt)
    logger.error("Text call exhausted retries: %s", last_error)
    return None


def call_vision_json(
    prompt: str,
    image_base64: str,
    max_tokens: int = 250,
    max_retries: int = 3,
) -> Optional[dict]:
    """Calls the vision model on one image and parses a single JSON object."""
    client = _client()
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                }],
            )
            _record_usage(response)
            content = response.choices[0].message.content
            return _extract_json(content)
        except Exception as exc:
            last_error = exc
            logger.warning("Vision call attempt %d/%d failed: %s", attempt, max_retries, exc)
            time.sleep(1.5 * attempt)
    logger.error("Vision call exhausted retries: %s", last_error)
    return None
