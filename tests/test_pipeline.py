"""End-to-end tests of steps 1-5 and the evaluation, with a fake model."""

import json

import pandas as pd
import pytest

from incident_pipeline import evaluate, extract, parse, schema_discovery, table
from incident_pipeline.cli import main
from incident_pipeline.parse import html_table_to_markdown
from incident_pipeline.schema_discovery import merge_proposals
from incident_pipeline.utils import read_json, read_jsonl, write_json

from .conftest import answer_from_prompt
from .fake_llm import FakeLLM


# ---------------------------------------------------------------- step 1
def test_parse_writes_clean_text_and_index(workdir):
    index = parse.run(workdir)
    names = {e["source_file"] for e in index}
    assert "IR-2025-001.md" in names and "IR-2025-002.txt" in names
    text = parse.load_text(workdir, next(e for e in index if e["source_file"] == "IR-2025-002.txt"))
    # .txt keeps its line structure
    assert "Report number: IR-2025-002\nDate of incident: 2025-02-03" in text
    assert (workdir.parsed_dir / "index.json").exists()


def test_parse_keeps_tables_as_markdown(workdir):
    if workdir.parser != "unstructured":
        pytest.skip("needs unstructured")
    index = parse.run(workdir)
    md = parse.load_text(workdir, next(e for e in index if e["source_file"] == "IR-2025-001.md"))
    assert "| Site | Warehouse B, loading dock 3 |" in md
    assert md.startswith("# Incident Report IR-2025-001")
    docx = parse.load_text(workdir, next(e for e in index if e["source_file"] == "IR-2025-005.docx"))
    assert "| Location | Warehouse A, conveyor line 2 |" in docx
    assert "- Use lock-out/tag-out before clearing any jam." in docx


def test_parse_skips_unchanged_files(workdir, caplog):
    parse.run(workdir)
    caplog.set_level("INFO")
    parse.run(workdir)
    assert "Up to date" in caplog.text


def test_html_table_to_markdown():
    html = "<table><tr><th>a</th><th>b|c</th></tr><tr><td>1</td><td> 2 </td></tr></table>"
    assert html_table_to_markdown(html) == "| a | b\\|c |\n|---|---|\n| 1 | 2 |"
    assert html_table_to_markdown("<p>no table</p>") is None


# ---------------------------------------------------------------- step 2
def test_merge_proposals_counts_support_and_merges_aliases():
    per_doc = {
        "d1": [
            {"name": "Incident ID", "type": "string", "description": "id"},
            {"name": "severity_of_injury", "type": "category", "options": ["Minor", "Major"], "description": "sev"},
            {"name": "weather", "type": "string", "description": "only once"},
            {"name": "schema", "type": "string", "description": "reserved word"},
        ],
        "d2": [
            {"name": "incident_id", "type": "string", "description": "id"},
            {"name": "injury_severity", "type": "category", "options": ["minor", "fatal"], "description": "sev"},
            {"name": "witnesses", "type": "list_of_strings", "description": "w"},
        ],
        "d3": [
            {"name": "incident_id", "type": "integer", "description": "id"},
            {"name": "witness", "type": "list_of_strings", "description": "w"},
        ],
    }
    kept, candidates = merge_proposals(per_doc, min_support=0.5)
    by_name = {c["name"]: c for c in candidates}
    assert by_name["incident_id"]["support"] == 1.0
    assert by_name["incident_id"]["type"] == "string"  # majority vote
    sev = next(c for c in candidates if "severity" in c["name"])
    assert sev["docs"] == 2 and set(sev["options"]) == {"minor", "major", "fatal"}
    assert sev["aliases"]  # the other spelling is recorded
    assert any(c["name"].startswith("witness") and c["docs"] == 2 for c in candidates)
    assert by_name["weather"]["kept"] is False
    assert "schema_value" in by_name
    assert {f["name"] for f in kept} == {"incident_id", sev["name"], next(c["name"] for c in candidates if c["name"].startswith("witness"))}


def test_discover_end_to_end_writes_draft_and_candidates(workdir):
    parse.run(workdir)
    workdir.sample_size = 3

    def propose(messages, schema):
        assert "REPORT:" in messages[1]["content"]
        return {
            "fields": [
                {"name": "Incident ID", "type": "string", "description": "Report number", "example": "IR"},
                {"name": "location", "type": "string", "description": "Where", "example": "x"},
                {"name": "severity", "type": "category", "description": "How bad", "options": ["minor", "major"]},
            ]
        }

    llm = FakeLLM(propose)
    draft = schema_discovery.run(workdir, llm)
    assert len(llm.calls) == 3  # one call per sample report, not one giant prompt
    assert [f["name"] for f in draft["fields"]] == ["incident_id", "location", "severity"]
    assert len(draft["generated"]["sample"]) == 3
    assert workdir.schema_draft_file.exists() and workdir.schema_candidates_file.exists()
    # proposals are cached per report
    schema_discovery.run(workdir, llm)
    assert len(llm.calls) == 3


def test_discover_retries_invalid_proposal(workdir):
    parse.run(workdir)
    workdir.sample_size = 1
    good = {"fields": [{"name": "incident_id", "type": "string", "description": "id"}]}
    llm = FakeLLM(["not json at all", {"fields": [{"name": "x", "type": "weird"}]}, good])
    draft = schema_discovery.run(workdir, llm)
    assert draft["fields"][0]["name"] == "incident_id"
    assert "invalid" in llm.calls[1]["messages"][-1]["content"]


# ---------------------------------------------------------------- steps 3 + 4
def test_extract_requires_reviewed_schema(workdir):
    parse.run(workdir)
    workdir.schema_file.unlink()
    with pytest.raises(RuntimeError, match="schema.json not found"):
        extract.run(workdir, FakeLLM(answer_from_prompt))


def test_extract_sends_descriptions_and_nullable_schema(workdir):
    parse.run(workdir)
    llm = FakeLLM(answer_from_prompt)
    records = extract.run(workdir, llm)
    assert len(records) == len(parse.load_index(workdir))
    call = llm.calls[0]
    prompt = call["messages"][1]["content"]
    assert "injury_severity (one of: none, first_aid" in prompt
    assert "Never guess" in prompt
    assert "$ref" not in json.dumps(call["schema"])
    assert {"type": "null"} in call["schema"]["properties"]["days_lost"]["anyOf"]
    rec = next(r for r in records if r["source_file"] == "IR-2025-001.md")
    assert rec["data"]["people_involved"] == ["Tom Reyes", "Ana Costa"]
    assert rec["meta"]["valid"] and rec["meta"]["model"] == "fake-model"
    assert len(list(read_jsonl(workdir.extractions_file))) == len(records)
    assert workdir.template_schema_file.exists()


def test_extract_caches_and_reruns_when_schema_changes(workdir):
    parse.run(workdir)
    llm = FakeLLM(answer_from_prompt)
    extract.run(workdir, llm)
    n = len(llm.calls)
    extract.run(workdir, llm)
    assert len(llm.calls) == n  # all cached
    schema = read_json(workdir.schema_file)
    schema["fields"][0]["description"] = "changed"
    write_json(workdir.schema_file, schema)
    extract.run(workdir, llm)
    assert len(llm.calls) == 2 * n


def test_extract_retries_then_salvages_valid_fields(workdir):
    parse.run(workdir)
    bad = {"incident_id": "IR-2025-002", "days_lost": "many", "injury_severity": "awful"}
    llm = FakeLLM([bad, bad, bad])
    records = extract.run(workdir, llm, only=["IR-2025-002.txt"])
    assert len(llm.calls) == 3  # first try + 2 retries
    assert "days_lost" in llm.calls[1]["messages"][-1]["content"]
    rec = records[0]
    assert rec["data"]["incident_id"] == "IR-2025-002"
    assert rec["data"]["days_lost"] is None and rec["data"]["injury_severity"] is None
    assert not rec["meta"]["valid"] and len(rec["meta"]["errors"]) == 2


def test_extract_splits_long_reports_into_chunks(workdir):
    parse.run(workdir)
    long_text = "\n\n".join(f"Paragraph {i}: " + "detail " * 300 for i in range(12))
    entry = next(e for e in parse.load_index(workdir) if e["source_file"] == "IR-2025-002.txt")
    (workdir.parsed_dir / entry["md_file"]).write_text(long_text)
    replies = iter([{"incident_id": "IR-2025-002", "recommendations": ["a"]}] + [{"recommendations": ["b"]}] * 50)
    llm = FakeLLM(lambda m, s: next(replies), num_ctx=4096)
    rec = extract.run(workdir, llm, only=["IR-2025-002.txt"])[0]
    assert rec["meta"]["chunks"] > 1 and len(llm.calls) == rec["meta"]["chunks"]
    assert rec["data"]["incident_id"] == "IR-2025-002"
    assert rec["data"]["recommendations"] == ["a", "b"]
    assert "part 1 of" in llm.calls[0]["messages"][1]["content"]


# ---------------------------------------------------------------- step 5 + evaluation
def test_table_and_evaluate(workdir):
    parse.run(workdir)
    extract.run(workdir, FakeLLM(answer_from_prompt))
    df = table.run(workdir)
    assert list(df.columns[:3]) == ["source_file", "incident_id", "incident_date"]
    assert len(df) == len(parse.load_index(workdir))
    row = df[df.source_file == "IR-2025-001.md"].iloc[0]
    assert row["people_involved"] == "Tom Reyes; Ana Costa"
    rates = pd.read_csv(workdir.fill_rate_file)
    assert set(rates.columns) == {"field", "type", "filled", "total", "fill_rate"}

    summary = evaluate.run(workdir).set_index("field")
    details = pd.read_csv(workdir.eval_details_file, keep_default_na=False)
    outcome = lambda src, f: details[(details.source_file == src) & (details.field == f)].outcome.iloc[0]  # noqa: E731
    assert outcome("IR-2025-004.txt", "reported_by") == "hallucinated"
    assert outcome("IR-2025-004.txt", "incident_date") == "missed"
    assert outcome("IR-2025-003.md", "location") == "correct"  # case/space/punctuation ignored
    assert outcome("IR-2025-002.txt", "days_lost") == "both_empty"
    assert outcome("IR-2025-001.md", "people_involved") == "correct"
    assert summary.loc["incident_id", "accuracy"] == 1.0
    assert summary.loc["ALL", "hallucinated"] >= 1
    assert 0 < summary.loc["ALL", "accuracy"] < 1


def test_table_skips_records_from_older_schema(workdir, caplog):
    parse.run(workdir)
    extract.run(workdir, FakeLLM(answer_from_prompt), only=["IR-2025-001.md"])
    schema = read_json(workdir.schema_file)
    schema["fields"] = schema["fields"][:3]
    write_json(workdir.schema_file, schema)
    with pytest.raises(RuntimeError, match="No records match"):
        table.run(workdir)


def test_score_cell_cases():
    s = evaluate.score_cell
    assert s("21", 21, "integer") == "correct"
    assert s("Warehouse B, loading dock 3", "Warehouse B loading dock 3", "string") == "close"
    assert s("A; B", ["b", "a"], "list_of_strings") == "correct"
    assert s("A; B", ["A"], "list_of_strings") == "wrong"
    assert s("", None, "string") == "both_empty"
    assert s("x", "", "string") == "missed"
    assert s("", "x", "string") == "hallucinated"


# ---------------------------------------------------------------- CLI
def test_cli_parse_and_table_without_model(workdir, tmp_path):
    args = ["--reports", str(workdir.reports_dir), "--output", str(workdir.output_dir),
            "--schema", str(workdir.schema_file), "--parser", workdir.parser]
    assert main(["parse", *args]) == 0
    assert main(["table", *args]) == 1  # nothing extracted yet -> clear error, exit code 1
