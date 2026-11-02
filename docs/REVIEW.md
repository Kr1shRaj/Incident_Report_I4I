# Review of the project plan

This reviews the plan document "Incident Report Project: How the Workflow Works" and
lists what the code in this repository does differently, and why.

## Verdict

The plan is sound and can be built with free software only. Parsing, schema discovery
from a sample, a runtime Pydantic template, extraction with Ollama's `format` option and
a pandas table is a standard and workable design. One schema per topic is the right call.

The points below are corrections. The first three would silently produce wrong data.

## Corrections

1. **Context window.** Ollama's default context is a few thousand tokens (2048 or 4096
   depending on version) and anything longer is dropped without an error. Six reports in
   one prompt usually overflow it. The code always sets `num_ctx` (default 16384), checks
   prompt size before sending, splits long reports into chunks and warns if Ollama reports
   a full context.

2. **"Leave the field empty" needs nullable fields.** With `format`, the model must produce
   JSON that matches the schema. A required string field must get a string, so the model
   invents one whatever the prompt says. Every template field is `Optional[...] = None`.

3. **The model does not read schema descriptions through `format`.** `format` only
   constrains the shape. The prompt now contains the field list with types, options and
   descriptions.

4. **Schema discovery.** "Ignore details that appear in only one report" is a counting job,
   which small models do badly. The code asks for fields one report at a time, then merges
   and counts in code: it cleans names, merges near-duplicates such as `severity_of_injury`
   and `injury_severity`, and keeps fields seen in at least `min_support` of the sample. The
   result is a draft that a person must review and save as `schema.json`.

5. **Knowledge graph.** With a flat schema (`people_involved` as a list of names,
   `injury_severity` as one value) the data cannot say *which* person had which injury, so
   "a person has a severity" is not recoverable. Graph links need either `object_list`
   fields (one record per person with name, injury, severity) or a separate relation
   extraction pass. Free-text values (`cause_description`) make poor graph nodes because
   they never match across reports; use `category` fields for anything you want to link or
   compare.

6. **Tables in parsed text.** Joining parsed elements as plain text flattens tables into a
   line of words. The code renders tables as Markdown from `text_as_html`. For PDFs this
   needs `pdf_strategy = "hi_res"` (slower, needs Poppler and Tesseract). Docling is
   supported as an alternative parser (`--parser docling`) and is often better on PDF
   tables. `.txt` files are read as-is, because partitioning merges their short
   `Key: value` lines.

7. **No accuracy measurement.** The plan says small models make mistakes but has no way to
   count them. `evaluate` scores extractions against a hand-filled `gold.csv` and counts
   hallucinated values (filled in although the report does not say) separately from
   missed ones.

8. **Keep the raw records.** Lists do not fit CSV cells. `extractions.jsonl` keeps the
   structured records; the table and the future graph are both built from it.

## Free-software check

| Item | Free? | Note |
|---|---|---|
| `unstructured` | Yes (open-source library, local) | The hosted Unstructured.io platform is the paid product. PDFs need Poppler, Tesseract, libmagic |
| Ollama + llama3.1:8b / qwen2.5:7b | Yes | ~6 GB RAM/VRAM; qwen2.5:14b ~10 GB |
| Pydantic, pandas, networkx | Yes | |
| Docling (optional) | Yes (MIT) | Large download (PyTorch) |
| Neo4j | Community Edition is free | |
| Google Docs | Yes | Download as `.docx` or Markdown |

## Build order

1. Collect 20–30 real reports; fill in about 10 by hand as `gold.csv`.
2. `parse`, and read the output, especially tables.
3. `discover`, then review and save `schema.json`. Include `category` fields.
4. `extract --only-gold` and `evaluate`; adjust schema descriptions, prompts or model until
   accuracy is acceptable and hallucinations are rare.
5. `extract` and `table` on the whole batch.
6. Next: normalise values (dates, names; `rapidfuzz` for fuzzy matching).
7. Next: knowledge graph with networkx and a pyvis HTML view; Neo4j only if you need queries.
