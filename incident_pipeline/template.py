"""Step 3 - turn the schema into a Pydantic template.

``schema.json`` format::

    {
      "topic": "workplace incident reports",
      "fields": [
        {"name": "incident_id", "type": "string", "description": "Report number"},
        {"name": "severity", "type": "category", "options": ["minor", "serious", "fatal"],
         "description": "Worst injury outcome"},
        {"name": "people", "type": "object_list", "description": "Everyone injured",
         "fields": [{"name": "name", "type": "string", "description": "..."},
                    {"name": "injury", "type": "string", "description": "..."}]}
      ]
    }

Every field is Optional with a default of None. That matters: Ollama forces the
answer to match the JSON Schema, so a required string field *must* get a
string, and the model will invent one. Optional fields let it answer null.
"""

from __future__ import annotations

import keyword
import re
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

TYPES = ("string", "integer", "number", "boolean", "date", "list_of_strings", "category", "object_list")
TYPE_ALIASES = {
    "str": "string",
    "text": "string",
    "int": "integer",
    "float": "number",
    "decimal": "number",
    "bool": "boolean",
    "datetime": "date",
    "list": "list_of_strings",
    "list[str]": "list_of_strings",
    "array": "list_of_strings",
    "list of strings": "list_of_strings",
    "enum": "category",
    "list_of_objects": "object_list",
    "list of objects": "object_list",
}
OTHER = "other"
RESERVED = set(dir(BaseModel)) | {"source_file"}
_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


class SchemaError(ValueError):
    pass


def safe_field_name(name: str) -> str:
    """snake_case a proposed name and steer it away from reserved words."""
    from .utils import snake_case

    name = snake_case(name)
    if name in RESERVED or keyword.iskeyword(name):
        name += "_value"
    return name


def normalize_type(value: str | None) -> str:
    t = (value or "string").strip().lower()
    t = TYPE_ALIASES.get(t, t)
    return t if t in TYPES else "string"


def validate_schema(schema: dict | list, _path: str = "") -> dict:
    """Check a schema dict and return it in canonical form. Raises SchemaError."""
    if isinstance(schema, list):
        schema = {"fields": schema}
    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise SchemaError(f"{_path or 'schema'}: 'fields' must be a non-empty list")
    seen: set[str] = set()
    clean: list[dict] = []
    for i, f in enumerate(fields):
        where = f"{_path}fields[{i}]"
        if not isinstance(f, dict) or "name" not in f:
            raise SchemaError(f"{where}: every field needs a 'name'")
        name = f["name"]
        if not _IDENT.match(name) or keyword.iskeyword(name):
            raise SchemaError(f"{where}: {name!r} must be snake_case (letters, digits, _)")
        if name in RESERVED:
            raise SchemaError(f"{where}: {name!r} is reserved; choose another name")
        if name in seen:
            raise SchemaError(f"{where}: duplicate field {name!r}")
        seen.add(name)
        ftype = f.get("type", "string")
        if ftype not in TYPES:
            raise SchemaError(f"{where}: unknown type {ftype!r}; use one of {', '.join(TYPES)}")
        item = {
            "name": name,
            "type": ftype,
            "description": str(f.get("description", "")).strip(),
        }
        if ftype == "category":
            options = [str(o).strip() for o in f.get("options", []) if str(o).strip()]
            if not options:
                raise SchemaError(f"{where}: category field {name!r} needs 'options'")
            if OTHER not in options:
                options.append(OTHER)
            item["options"] = list(dict.fromkeys(options))
        if ftype == "object_list":
            sub = validate_schema({"fields": f.get("fields", [])}, _path=f"{where}.")
            if any(s["type"] == "object_list" for s in sub["fields"]):
                raise SchemaError(f"{where}: object_list fields cannot be nested twice")
            item["fields"] = sub["fields"]
        clean.append(item)
    out = dict(schema)
    out["fields"] = clean
    return out


def _python_type(f: dict, model_name: str) -> Any:
    t = f["type"]
    if t == "string":
        return str
    if t == "date":
        return str  # format is asked for in the description and normalised later
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "list_of_strings":
        return List[str]
    if t == "category":
        return Literal[tuple(f["options"])]  # type: ignore[misc]
    if t == "object_list":
        sub = build_model({"fields": f["fields"]}, name=f"{model_name}_{f['name']}")
        return List[sub]  # type: ignore[valid-type]
    raise SchemaError(f"unknown type {t!r}")


def field_description(f: dict) -> str:
    desc = f.get("description", "")
    if f["type"] == "date":
        desc = (desc + " (format YYYY-MM-DD)").strip()
    return desc


def build_model(schema: dict, name: str = "IncidentReport") -> type[BaseModel]:
    """Build the Pydantic template at runtime with ``create_model``."""
    schema = validate_schema(schema)
    definitions: dict[str, Any] = {}
    for f in schema["fields"]:
        py_type = _python_type(f, name)
        definitions[f["name"]] = (
            Optional[py_type],
            Field(default=None, description=field_description(f)),
        )
    return create_model(  # type: ignore[call-overload]
        name,
        __config__=ConfigDict(extra="ignore", protected_namespaces=()),
        **definitions,
    )


def describe_fields(schema: dict, indent: str = "") -> str:
    """Field list for prompts. The model only sees descriptions if we put them in the prompt."""
    lines = []
    for f in schema["fields"]:
        kind = f["type"]
        if kind == "category":
            kind = "one of: " + ", ".join(f["options"])
        elif kind == "object_list":
            kind = "list of objects"
        lines.append(f"{indent}- {f['name']} ({kind}): {field_description(f)}")
        if f["type"] == "object_list":
            lines.append(describe_fields({"fields": f["fields"]}, indent + "    "))
    return "\n".join(lines)


def is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def merge_records(records: list[dict], schema: dict) -> dict:
    """Combine extractions of several chunks of one report.

    Scalars: first non-empty value wins. Lists: union in order of appearance.
    """
    merged: dict[str, Any] = {}
    for f in schema["fields"]:
        name = f["name"]
        values = [r.get(name) for r in records if not is_empty(r.get(name))]
        if not values:
            merged[name] = None
        elif f["type"] in ("list_of_strings", "object_list"):
            out: list = []
            for v in values:
                for item in v if isinstance(v, list) else [v]:
                    if item not in out:
                        out.append(item)
            merged[name] = out
        else:
            non_other = [v for v in values if v != OTHER]
            merged[name] = non_other[0] if non_other else values[0]
    return merged
