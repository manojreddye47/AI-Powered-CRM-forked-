"""Tests for ai/validators/output_guard.py (AI-12)."""

import json
from types import SimpleNamespace

import pytest

from ai.validators.output_guard import (
    MAX_RETRIES,
    STATUS_NEEDS_REVIEW,
    STATUS_OK,
    SUBJECT_MAX_LENGTH,
    OutputValidationError,
    guarded_generate,
    validate_output,
)

VALID = {
    "subject": "Backend internship at Acme",
    "body": "Hello, we noticed your interest in backend work and wanted to share a role.",
    "personalization_summary": "Matches the recipient's backend focus.",
    "claims_used": [{"fact_id": "f1", "claim": "Acme is hiring backend interns"}],
    "missing_information_warnings": [],
    "confidence_note": "High alignment with verified facts.",
}
VALID_JSON = json.dumps(VALID)


def with_overrides(**overrides):
    return {**VALID, **overrides}


def without(field):
    return {k: v for k, v in VALID.items() if k != field}


class ScriptedGenerate:
    """Fake generate(): returns scripted replies and records the error it receives."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.errors_seen = []

    def __call__(self, error):
        self.errors_seen.append(error)
        reply = self.replies[len(self.errors_seen) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def calls(self):
        return len(self.errors_seen)


# ---------- valid output ----------

def test_valid_dict_passes():
    assert validate_output(VALID) == VALID


def test_valid_json_string_passes():
    assert validate_output(VALID_JSON) == VALID


def test_valid_json_in_code_fence_passes():
    assert validate_output(f"```json\n{VALID_JSON}\n```") == VALID


def test_empty_claims_and_warnings_are_allowed():
    data = with_overrides(claims_used=[], missing_information_warnings=[])
    assert validate_output(data) == data


def test_subject_at_max_length_passes():
    data = with_overrides(subject="x" * SUBJECT_MAX_LENGTH)
    assert validate_output(data) == data


# ---------- invalid / malformed output ----------

@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "not json", "{broken", "[1, 2, 3]", '"just a string"', 42],
    ids=["none", "empty", "blank", "text", "broken-json", "list", "json-string", "number"],
)
def test_malformed_output_raises_validation_error(raw):
    with pytest.raises(OutputValidationError):
        validate_output(raw)


def test_invalid_json_message_is_specific():
    with pytest.raises(OutputValidationError, match="not valid JSON"):
        validate_output("not json")


def test_subject_too_long_is_rejected():
    with pytest.raises(OutputValidationError, match="81 characters"):
        validate_output(with_overrides(subject="x" * (SUBJECT_MAX_LENGTH + 1)))


@pytest.mark.parametrize("field", ["subject", "body"])
def test_blank_subject_or_body_is_rejected(field):
    with pytest.raises(OutputValidationError, match=f"'{field}' must not be empty"):
        validate_output(with_overrides(**{field: "   "}))


@pytest.mark.parametrize(
    "field", ["subject", "body", "personalization_summary", "confidence_note"]
)
def test_non_string_fields_are_rejected(field):
    with pytest.raises(OutputValidationError, match=f"'{field}' must be a string"):
        validate_output(with_overrides(**{field: 123}))


def test_claims_used_must_be_a_list():
    with pytest.raises(OutputValidationError, match="'claims_used' must be a list"):
        validate_output(with_overrides(claims_used="nope"))


def test_claims_used_items_need_fact_id_and_claim():
    with pytest.raises(OutputValidationError, match=r"claims_used\[0\]\.claim"):
        validate_output(with_overrides(claims_used=[{"fact_id": "f1"}]))


def test_missing_information_warnings_must_be_strings():
    with pytest.raises(OutputValidationError, match=r"missing_information_warnings\[0\]"):
        validate_output(with_overrides(missing_information_warnings=[1]))


# ---------- missing and extra fields ----------

@pytest.mark.parametrize("field", list(VALID))
def test_each_missing_required_field_is_reported(field):
    with pytest.raises(OutputValidationError, match=f"Missing required field '{field}'"):
        validate_output(without(field))


def test_all_missing_fields_are_reported_together():
    with pytest.raises(OutputValidationError) as exc:
        validate_output({"subject": "Hi"})
    message = str(exc.value)
    for field in ("body", "personalization_summary", "claims_used",
                  "missing_information_warnings", "confidence_note"):
        assert f"'{field}'" in message


def test_extra_field_is_rejected():
    with pytest.raises(OutputValidationError, match="Unexpected field.*extra"):
        validate_output(with_overrides(extra="x"))


# ---------- single-retry behaviour ----------

def test_max_retries_is_one():
    assert MAX_RETRIES == 1


def test_valid_output_makes_one_call_and_no_retry():
    generate = ScriptedGenerate(VALID_JSON)
    result = guarded_generate(generate)
    assert result["status"] == STATUS_OK
    assert result["output"] == VALID
    assert result["error"] is None
    assert result["attempts"] == 1
    assert generate.calls == 1
    assert generate.errors_seen == [None]


def test_invalid_then_valid_succeeds_on_the_retry():
    generate = ScriptedGenerate("not json", VALID_JSON)
    result = guarded_generate(generate)
    assert result["status"] == STATUS_OK
    assert result["output"] == VALID
    assert result["attempts"] == 2
    assert generate.calls == 2


def test_retry_receives_the_specific_validation_error():
    generate = ScriptedGenerate(json.dumps(without("body")), VALID_JSON)
    guarded_generate(generate)
    assert generate.errors_seen[0] is None
    assert "Missing required field 'body'" in generate.errors_seen[1]


def test_never_more_than_two_attempts():
    generate = ScriptedGenerate("bad", "bad", "bad", "bad")
    guarded_generate(generate)
    assert generate.calls == 2


# ---------- final needs_review response ----------

def test_two_failures_return_needs_review():
    generate = ScriptedGenerate("not json", json.dumps(with_overrides(extra="x")))
    result = guarded_generate(generate)
    assert result["status"] == STATUS_NEEDS_REVIEW
    assert result["output"] is None
    assert result["attempts"] == 2
    assert "extra" in result["error"]  # reports the final attempt's error


def test_exceptions_from_generate_do_not_escape():
    generate = ScriptedGenerate(RuntimeError("boom"), RuntimeError("boom again"))
    result = guarded_generate(generate)
    assert result["status"] == STATUS_NEEDS_REVIEW
    assert result["attempts"] == 2
    assert "boom again" in result["error"]


def test_exception_then_valid_output_succeeds():
    generate = ScriptedGenerate(RuntimeError("timeout"), VALID_JSON)
    result = guarded_generate(generate)
    assert result["status"] == STATUS_OK
    assert "timeout" in generate.errors_seen[1]


def test_unexpected_validator_error_is_contained():
    def broken_validate(raw):
        raise KeyError("surprise")

    result = guarded_generate(ScriptedGenerate(VALID_JSON, VALID_JSON), validate=broken_validate)
    assert result["status"] == STATUS_NEEDS_REVIEW


@pytest.mark.parametrize(
    "bad",
    [None, "", "not json", "[]", 42, {}],
    ids=["none", "empty", "text", "list", "number", "empty-dict"],
)
def test_malformed_output_never_raises_through_the_guard(bad):
    result = guarded_generate(ScriptedGenerate(bad, bad))
    assert result["status"] == STATUS_NEEDS_REVIEW
    assert result["output"] is None


# ---------- intended generation flow (GeminiClient -> guard) ----------

def make_generate(client, base_prompt="BASE PROMPT"):
    """Adapter mirroring the intended flow.

    GeminiClient.generate() returns (parsed_json, usage), so unpack it. On the
    retry, the specific validation error is appended to the prompt.
    """

    def generate(error):
        prompt = base_prompt
        if error:
            prompt += (
                f"\n\nYour previous reply was invalid: {error}\n"
                "Return only a valid JSON object matching the schema."
            )
        parsed, _usage = client.generate(prompt)
        return parsed

    return generate


class FakeClient:
    """Stands in for GeminiClient: same generate() signature and (parsed, usage) return."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, schema=None, temperature=0.3):
        self.prompts.append(prompt)
        reply = self.replies[len(self.prompts) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply, {"total_tokens": 1}


def test_flow_gemini_client_mock_mode_output_passes_guard():
    gemini_client = pytest.importorskip("ai.gemini_client")
    config = SimpleNamespace(mock_mode=True, gemini_api_key=None)
    client = gemini_client.GeminiClient(config=config)
    result = guarded_generate(make_generate(client))
    assert result["status"] == STATUS_OK
    assert result["attempts"] == 1


def test_flow_retry_prompt_contains_the_validation_error():
    client = FakeClient(without("body"), VALID)
    result = guarded_generate(make_generate(client))
    assert result["status"] == STATUS_OK
    assert len(client.prompts) == 2
    assert "Missing required field 'body'" not in client.prompts[0]
    assert "Missing required field 'body'" in client.prompts[1]


def test_flow_invalid_twice_returns_needs_review_after_two_client_calls():
    client = FakeClient(with_overrides(extra="x"), with_overrides(extra="x"))
    result = guarded_generate(make_generate(client))
    assert result["status"] == STATUS_NEEDS_REVIEW
    assert len(client.prompts) == 2


def test_flow_client_json_decode_error_is_handled_and_retried():
    # The real client raises JSONDecodeError when the model returns non-JSON text.
    client = FakeClient(json.JSONDecodeError("Expecting value", "", 0), VALID)
    result = guarded_generate(make_generate(client))
    assert result["status"] == STATUS_OK
    assert "Expecting value" in client.prompts[1]


def test_flow_client_failing_twice_returns_needs_review_without_raising():
    client = FakeClient(RuntimeError("Gemini down"), RuntimeError("Gemini down"))
    result = guarded_generate(make_generate(client))
    assert result["status"] == STATUS_NEEDS_REVIEW
    assert result["attempts"] == 2


def test_flow_guarded_output_also_matches_the_pydantic_schema():
    pytest.importorskip("pydantic")
    schemas = pytest.importorskip("ai.schemas.draft_output")
    result = guarded_generate(make_generate(FakeClient(VALID)))
    assert result["status"] == STATUS_OK
    model = schemas.AIDraftOutput(**result["output"])
    assert model.subject == VALID["subject"]