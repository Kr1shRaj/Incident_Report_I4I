import pytest
from pydantic import ValidationError

from incident_pipeline.extract import inline_refs
from incident_pipeline.template import (
    SchemaError,
    build_model,
    describe_fields,
    merge_records,
    safe_field_name,
    validate_schema,
)
from incident_pipeline.utils import canonical_key, estimate_tokens, snake_case, split_into_chunks

SCHEMA = {
    "fields": [
        {"name": "incident_id", "type": "string", "description": "Report number"},
        {"name": "incident_date", "type": "date", "description": "When it happened"},
        {"name": "days_lost", "type": "integer", "description": "Days off work"},
        {"name": "people", "type": "list_of_strings", "description": "Names"},
        {"name": "severity", "type": "category", "options": ["minor", "major"], "description": "Severity"},
    ]
}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Incident ID", "incident_id"),
        ("incidentDate", "incident_date"),
        ("  people--involved ", "people_involved"),
        ("2nd witness", "f_2nd_witness"),
        ("!!!", "field"),
    ],
)
def test_snake_case(raw, expected):
    assert snake_case(raw) == expected


def test_canonical_key_merges_reordered_and_plural_names():
    assert canonical_key("severity_of_injury") == canonical_key("Injury Severity")
    assert canonical_key("recommendations") == canonical_key("recommendation")
    assert canonical_key("incident_date") != canonical_key("injury_date")


def test_safe_field_name_avoids_pydantic_attributes():
    assert safe_field_name("schema") == "schema_value"
    assert safe_field_name("JSON") == "json_value"
    assert safe_field_name("class") == "class_value"
    assert safe_field_name("Model Number") == "model_number"


def test_split_into_chunks_respects_budget_and_keeps_text():
    paras = [f"Paragraph {i} " + "word " * 50 for i in range(20)]
    text = "\n\n".join(paras)
    chunks = split_into_chunks(text, max_tokens=200)
    assert len(chunks) > 1
    assert all(estimate_tokens(c) <= 200 + 1 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_split_into_chunks_short_text_unchanged():
    assert split_into_chunks("short", 100) == ["short"]


@pytest.mark.parametrize(
    "bad,message",
    [
        ({"fields": []}, "non-empty"),
        ({"fields": [{"name": "Bad Name", "type": "string"}]}, "snake_case"),
        ({"fields": [{"name": "json", "type": "string"}]}, "reserved"),
        ({"fields": [{"name": "a", "type": "string"}, {"name": "a", "type": "string"}]}, "duplicate"),
        ({"fields": [{"name": "a", "type": "category"}]}, "options"),
        ({"fields": [{"name": "a", "type": "matrix"}]}, "unknown type"),
    ],
)
def test_validate_schema_rejects(bad, message):
    with pytest.raises(SchemaError, match=message):
        validate_schema(bad)


def test_validate_schema_adds_other_option_and_accepts_bare_list():
    s = validate_schema(SCHEMA["fields"])
    sev = next(f for f in s["fields"] if f["name"] == "severity")
    assert sev["options"] == ["minor", "major", "other"]


def test_build_model_fields_are_optional_so_null_is_allowed():
    Model = build_model(SCHEMA)
    js = Model.model_json_schema()
    assert "required" not in js  # nothing forces the model to invent a value
    for prop in js["properties"].values():
        assert {"type": "null"} in prop["anyOf"]
    empty = Model.model_validate({})
    assert empty.model_dump() == {f["name"]: None for f in SCHEMA["fields"]}


def test_build_model_validates_types_and_categories():
    Model = build_model(SCHEMA)
    ok = Model.model_validate({"days_lost": 3, "people": ["A"], "severity": "minor", "unknown": 1})
    assert ok.days_lost == 3 and ok.severity == "minor"
    assert not hasattr(ok, "unknown")
    with pytest.raises(ValidationError):
        Model.model_validate({"severity": "catastrophic"})
    with pytest.raises(ValidationError):
        Model.model_validate({"days_lost": "many"})


def test_object_list_builds_nested_model_and_refs_inline():
    schema = {
        "fields": [
            {
                "name": "people",
                "type": "object_list",
                "description": "Injured people",
                "fields": [
                    {"name": "name", "type": "string"},
                    {"name": "severity", "type": "category", "options": ["minor"]},
                ],
            }
        ]
    }
    Model = build_model(schema)
    rec = Model.model_validate({"people": [{"name": "Ana", "severity": "minor"}]}).model_dump()
    assert rec == {"people": [{"name": "Ana", "severity": "minor"}]}
    flat = inline_refs(Model.model_json_schema())
    assert "$ref" not in str(flat) and "$defs" not in flat
    assert "people" in describe_fields(validate_schema(schema))


def test_merge_records_first_scalar_and_list_union():
    schema = validate_schema(SCHEMA)
    merged = merge_records(
        [
            {"incident_id": None, "people": ["A"], "severity": "other"},
            {"incident_id": "IR-1", "people": ["A", "B"], "severity": "major"},
            {"incident_id": "IR-2", "people": None, "severity": None},
        ],
        schema,
    )
    assert merged["incident_id"] == "IR-1"
    assert merged["people"] == ["A", "B"]
    assert merged["severity"] == "major"  # a real option beats "other"
    assert merged["days_lost"] is None


def test_describe_fields_includes_descriptions_and_options():
    text = describe_fields(validate_schema(SCHEMA))
    assert "incident_date (date): When it happened (format YYYY-MM-DD)" in text
    assert "one of: minor, major, other" in text
