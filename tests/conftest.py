import re
import shutil
from pathlib import Path

import pytest

from incident_pipeline.config import Config

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# What a perfect model would extract from each example report (used by the fake model).
ANSWERS = {
    "IR-2025-001": {
        "incident_id": "IR-2025-001", "incident_date": "2025-01-14",
        "location": "Warehouse B, loading dock 3", "reported_by": "Priya Nair",
        "people_involved": ["Tom Reyes", "Ana Costa"], "injury_severity": "lost_time",
        "days_lost": 21, "equipment_involved": ["forklift FL-07"], "cause_category": "vehicle",
        "cause_description": "Faulty reversing alarm; pedestrians in the forklift lane.",
        "recommendations": ["Repair and test reversing alarms weekly", "Mark pedestrian walkways"],
    },
    "IR-2025-002": {
        "incident_id": "IR-2025-002", "incident_date": "2025-02-03",
        "location": "Warehouse A, aisle 12", "reported_by": "Marcus Webb",
        "people_involved": ["Liam Okafor"], "injury_severity": "first_aid", "days_lost": None,
        "cause_category": "slip_trip_fall",
    },
    "IR-2025-003": {
        "incident_id": "IR-2025-003", "incident_date": "2025-02-21",
        "location": "warehouse B,  returns area.", "reported_by": "Priya Nair",
        "people_involved": ["Jordan Lee"], "injury_severity": "lost_time", "days_lost": 4,
        "cause_category": "manual_handling",
    },
    # a "bad" answer: invents a reporter and misses the date
    "IR-2025-004": {
        "incident_id": "IR-2025-004", "incident_date": None, "location": "Yard, gate 2",
        "reported_by": "Agency driver", "injury_severity": "none", "cause_category": "vehicle",
    },
    "IR-2025-005": {
        "incident_id": "IR-2025-005", "incident_date": "2025-03-18",
        "location": "Warehouse A, conveyor line 2", "reported_by": "Marcus Webb",
        "people_involved": ["Grace Mensah"], "injury_severity": "lost_time", "days_lost": 10,
        "cause_category": "machinery",
    },
}


def answer_from_prompt(messages, schema):
    """Fake model: find the report id in the prompt and return the canned answer."""
    report = messages[1]["content"].split("REPORT:", 1)[-1]
    match = re.search(r"IR-2025-\d{3}", report)
    return ANSWERS.get(match.group(0), {}) if match else {}


@pytest.fixture
def workdir(tmp_path):
    """A copy of the examples with a Config pointing at it."""
    shutil.copytree(EXAMPLES / "reports", tmp_path / "reports")
    shutil.copy(EXAMPLES / "schema.json", tmp_path / "schema.json")
    shutil.copy(EXAMPLES / "gold.csv", tmp_path / "gold.csv")
    try:
        import unstructured.partition.auto  # noqa: F401

        parser = "unstructured"
    except ImportError:  # light install: test with .txt/.md only
        parser = "plain"
        (tmp_path / "reports" / "IR-2025-005.docx").unlink()
    cfg = Config(
        reports_dir=tmp_path / "reports",
        output_dir=tmp_path / "output",
        schema_file=tmp_path / "schema.json",
        gold_file=tmp_path / "gold.csv",
        parser=parser,
        topic="warehouse workplace incident reports",
    )
    return cfg
