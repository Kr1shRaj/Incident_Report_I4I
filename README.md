# Incident Report Pipeline

Turns a batch of reports about one topic into structured, comparable data, using only
free software and a model that runs on your own computer (nothing leaves your machine).

```
reports/ (PDF, DOCX, TXT, MD)
   │ 1 parse ───────────► output/parsed/*.md            clean text, tables kept
   │ 2 discover ────────► output/schema.draft.json      proposed fields
   │      you review ───► schema.json                   the fields you approved
   │ 3+4 extract ───────► output/extractions.jsonl      one record per report
   │ 5 table ───────────► output/comparison.csv         one row per report
   └ evaluate ──────────► output/eval.csv               accuracy vs. your hand-filled gold.csv
```

Steps 6 (normalising values) and 7 (knowledge graph) build on `extractions.jsonl` and are
not built yet. See [docs/REVIEW.md](docs/REVIEW.md) for why the workflow looks like this.

## Setup

1. **Ollama**: install from <https://ollama.com>, then pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
   `llama3.1:8b` or `qwen2.5:7b` need about 6 GB of RAM/VRAM. `qwen2.5:14b` follows
   instructions better and needs about 10 GB. CPU-only works, just slowly.

2. **Python 3.11 or newer**, in a virtual environment:
   ```bash
   python -m venv .venv
   ```
   ```bash
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```
   ```bash
   pip install -e ".[parse,dev]"
   ```
   The `parse` extra installs the open-source `unstructured` library. It is free and runs
   locally; only the hosted Unstructured.io platform is paid.

3. **System packages for PDFs** (not needed for DOCX/TXT/MD):
   - Ubuntu/Debian: `sudo apt install poppler-utils tesseract-ocr libmagic1`
   - macOS: `brew install poppler tesseract libmagic`
   - Windows: install Poppler and Tesseract and add both to `PATH` (or use WSL).

4. **Google Docs**: download each as `.docx` (File → Download → Microsoft Word), or as
   Markdown, and put it in `reports/`.

Check that everything is in place:

```bash
python -m incident_pipeline check
```

## Try it on the examples

`examples/` has eight fictional reports, a reviewed `schema.json` and a hand-filled
`gold.csv` for five of them.

```bash
python -m incident_pipeline run --reports examples/reports --schema examples/schema.json --gold examples/gold.csv --output output/examples
```

## Use it on your reports

Put your reports in `reports/`. Settings can go in a file
(`cp pipeline.example.toml pipeline.toml`, then add `--config pipeline.toml`) or be given
as flags. At least set the topic.

**1. Parse**, then open a few files in `output/parsed/` and check the text, especially tables.

```bash
python -m incident_pipeline parse
```

**2. Discover the schema** from a sample (6 random reports by default; `--sample a.pdf b.pdf`
picks them yourself).

```bash
python -m incident_pipeline discover --topic "construction site accident reports"
```

**Review the draft. This is the most important manual step.** Open
`output/schema.draft.json` and `output/schema_candidates.csv` (every candidate with the
share of reports it appeared in). Then:
- rename fields, fix types, write clear descriptions (the model sees them when extracting);
- add back useful fields that missed the cut, remove useless ones;
- for values you want to compare or link later (severity, cause type), use `"type": "category"`
  with a fixed list of `options`;
- save the result as `schema.json` in the project folder.

Field types: `string`, `integer`, `number`, `boolean`, `date`, `list_of_strings`,
`category` (needs `options`; `other` is added automatically) and `object_list`
(a list of small records with their own `fields`, e.g. one entry per injured person with
name, injury and severity; you need this for person-level links in the graph later).

**3. Make a gold set.** Copy `output/comparison.csv`'s header into `gold.csv` (or start
from `examples/gold.csv`) and fill in about 10 reports by hand. Leave a cell empty when
the report does not say. Separate list items with `;`.

**4. Extract and score the gold reports first**, and improve prompts, schema descriptions or
the model until the scores are good enough:

```bash
python -m incident_pipeline extract --only-gold
```
```bash
python -m incident_pipeline evaluate
```

**5. Extract everything and build the table.**

```bash
python -m incident_pipeline extract
```
```bash
python -m incident_pipeline table
```

`run` does parse → (discover) → extract → table → (evaluate) in one go. Everything is
cached: re-running only redoes reports whose file, schema or model changed. Use `--force`
to redo everything.

## Output files

| File | Contents |
|---|---|
| `parsed/<doc>.md` | Clean text of each report: headings as `#`, lists as `-`, tables as Markdown tables |
| `parsed/<doc>.json` | The labelled pieces (Title, NarrativeText, ListItem, Table…), page and depth; for section splitting later |
| `parsed/index.json` | All parsed reports and their sizes |
| `schema_runs/<doc>.json` | The fields the model proposed for each sample report |
| `schema_candidates.csv` | Every candidate field: support, aliases merged into it, type votes, kept or not |
| `schema.draft.json` | The kept fields, for you to review and save as `schema.json` |
| `template.schema.json` | The JSON Schema sent to Ollama (all fields nullable) |
| `extracted/<doc>.json` | One record: `data` (the fields) and `meta` (model, chunks, retries, validation errors) |
| `extractions.jsonl` | All records; the input for the table, and later the graph |
| `comparison.csv` | One row per report, one column per field, plus `_valid` and `_errors` |
| `fill_rate.csv` | Share of reports with a value for each field. A low rate means a weak field or lost input |
| `eval.csv` | Per field: correct, close, wrong, missed, **hallucinated**, both_empty, accuracy |
| `eval_details.csv` | Every scored cell with the gold and predicted values side by side |

## How the main problems are handled

| Problem | What the code does |
|---|---|
| Ollama silently cuts prompts that exceed its small default context | `num_ctx` is always set (16384); prompts are size-checked before sending, long reports are split into chunks and merged, and a warning is logged if Ollama reports a full context |
| Forced JSON makes the model invent values for required fields | Every template field is `Optional` with default `null` |
| The model does not see field descriptions through `format` | The field list with types, options and descriptions is in the prompt |
| A one-shot "find the common fields" prompt is unreliable | One call per sample report; counting, merging of near-duplicate names and the support threshold are done in code |
| Invalid answers | Pydantic validation, up to 2 retries with the error message, then valid fields are kept and the rest set to null and flagged |
| LLM field names like `"Incident ID"` or `schema` | Names are cleaned to snake_case and moved away from reserved words |
| No way to know if results are right | `evaluate` against your gold set, with hallucinations counted separately |

## Tests

```bash
pytest
```

The tests use a fake model and a stand-in Ollama server, so they run without Ollama.

## Layout

```
incident_pipeline/
  cli.py               command line
  config.py            settings and defaults
  parse.py             step 1
  schema_discovery.py  step 2
  template.py          step 3 (schema.json -> Pydantic model)
  extract.py           step 4
  table.py             step 5
  evaluate.py          scoring against gold.csv
  llm.py               Ollama client
examples/              fictional reports, schema and gold file
tests/
```
