"""
AI Output Guard.
Validates AI-generated email drafts against contracts/ai-schemas/draft_output.json,
retries once with the specific validation error, and returns status "needs_review"
if the second attempt also fails. Never sends emails and never lets malformed
model output raise an unhandled exception.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NEEDS_REVIEW = "needs_review"
MAX_RETRIES = 1  # one retry means two attempts in total
SUBJECT_MAX_LENGTH = 80

REQUIRED_FIELDS: List[str] = [
    "subject",
    "body",
    "personalization_summary",
    "claims_used",
    "missing_information_warnings",
    "confidence_note",
]
STRING_FIELDS: List[str] = ["subject", "body", "personalization_summary", "confidence_note"]


class OutputValidationError(ValueError):
    """Raised when AI-generated output fails validation."""


def _strip_code_fences(text: str) -> str:
    """Removes ```json ... ``` fences that models often wrap around JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse(raw: Any) -> Dict[str, Any]:
    """Turns raw model output (JSON text or an already-parsed dict) into a dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise OutputValidationError("Output is empty or not text.")
    try:
        data = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"Output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OutputValidationError("Output must be a JSON object.")
    return data


def validate_output(raw: Any) -> Dict[str, Any]:
    """
    Validates raw model output against the AIDraftOutput contract.
    Returns the parsed dict, or raises OutputValidationError listing every problem found.
    """
    data = _parse(raw)
    errors: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field '{field}'.")

    extra = sorted(str(key) for key in data if key not in REQUIRED_FIELDS)
    if extra:
        errors.append(f"Unexpected field(s) not allowed by the schema: {', '.join(extra)}.")

    for field in STRING_FIELDS:
        if field in data and not isinstance(data[field], str):
            errors.append(f"Field '{field}' must be a string.")

    for field in ("subject", "body"):
        value = data.get(field)
        if isinstance(value, str) and not value.strip():
            errors.append(f"Field '{field}' must not be empty.")

    subject = data.get("subject")
    if isinstance(subject, str) and len(subject) > SUBJECT_MAX_LENGTH:
        errors.append(
            f"Field 'subject' is {len(subject)} characters; the maximum is {SUBJECT_MAX_LENGTH}."
        )

    if "claims_used" in data:
        claims = data["claims_used"]
        if not isinstance(claims, list):
            errors.append("Field 'claims_used' must be a list.")
        else:
            for i, item in enumerate(claims):
                if not isinstance(item, dict):
                    errors.append(f"claims_used[{i}] must be an object with 'fact_id' and 'claim'.")
                    continue
                for key in ("fact_id", "claim"):
                    if not isinstance(item.get(key), str):
                        errors.append(f"claims_used[{i}].{key} is missing or not a string.")

    if "missing_information_warnings" in data:
        warnings = data["missing_information_warnings"]
        if not isinstance(warnings, list):
            errors.append("Field 'missing_information_warnings' must be a list of strings.")
        else:
            for i, item in enumerate(warnings):
                if not isinstance(item, str):
                    errors.append(f"missing_information_warnings[{i}] must be a string.")

    if errors:
        raise OutputValidationError(" ".join(errors))
    return data


def guarded_generate(
    generate: Callable[[Optional[str]], Any],
    validate: Callable[[Any], Dict[str, Any]] = validate_output,
) -> Dict[str, Any]:
    """
    Runs generate() and validates its output, with at most one retry.

    generate(error) is called with None on the first attempt, and with the
    validation error message on the retry so it can be added to the prompt.

    Returns: {"status": "ok" | "needs_review", "output": dict | None,
              "error": str | None, "attempts": int}
    """
    error: Optional[str] = None
    attempts = 0

    for _ in range(1 + MAX_RETRIES):
        attempts += 1
        try:
            raw = generate(error)
            data = validate(raw)
            return {"status": STATUS_OK, "output": data, "error": None, "attempts": attempts}
        except Exception as exc:  # malformed output must never escape this guard
            error = str(exc)
            logger.warning("AI output invalid (attempt %d): %s", attempts, error)

    return {"status": STATUS_NEEDS_REVIEW, "output": None, "error": error, "attempts": attempts}