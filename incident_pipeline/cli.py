"""Command line: python -m incident_pipeline <step> [options]

Steps
  check      check that Ollama is running and the model is pulled
  parse      1  read and clean the reports           -> output/parsed/
  discover   2  propose a schema from a sample        -> output/schema.draft.json
  extract    3+4 build the template, extract reports  -> output/extractions.jsonl
  table      5  comparison table and fill rates       -> output/comparison.csv
  evaluate      score against gold.csv                -> output/eval.csv
  run        parse, (discover), extract, table, (evaluate) in one go
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from . import evaluate, extract, parse, schema_discovery, table
from .config import Config
from .utils import log


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="TOML file with settings (see pipeline.example.toml)")
    p.add_argument("--reports", dest="reports_dir", help="folder with the reports (default: reports)")
    p.add_argument("--output", dest="output_dir", help="folder for results (default: output)")
    p.add_argument("--schema", dest="schema_file", help="reviewed schema file (default: schema.json)")
    p.add_argument("--gold", dest="gold_file", help="hand-filled reports (default: gold.csv)")
    p.add_argument("--model", help="Ollama model, e.g. llama3.1:8b or qwen2.5:14b")
    p.add_argument("--num-ctx", dest="num_ctx", type=int, help="context window in tokens (default 16384)")
    p.add_argument("--host", dest="ollama_host", help="Ollama address (default http://localhost:11434)")
    p.add_argument("--topic", help='what the reports are about, e.g. "construction site accident reports"')
    p.add_argument("--parser", choices=["unstructured", "docling", "plain"])
    p.add_argument("--pdf-strategy", dest="pdf_strategy", choices=["fast", "hi_res", "auto", "ocr_only"])
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m incident_pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="check Ollama and the model")
    _common(p)

    p = sub.add_parser("parse", help="step 1: read and clean the reports")
    _common(p)
    p.add_argument("--force", action="store_true", help="re-parse even if up to date")

    p = sub.add_parser("discover", help="step 2: propose a schema from a sample")
    _common(p)
    p.add_argument("--sample", nargs="+", help="use these reports as the sample (file names)")
    p.add_argument("--sample-size", dest="sample_size", type=int)
    p.add_argument("--min-support", dest="min_support", type=float, help="keep fields seen in this share of the sample")
    p.add_argument("--force", action="store_true", help="ask the model again for every sample report")
    p.add_argument("--accept", action="store_true", help="copy the draft to schema.json without review")

    for name, helptext in (("extract", "steps 3+4: extract every report"), ("run", "all steps")):
        p = sub.add_parser(name, help=helptext)
        _common(p)
        p.add_argument("--accept-draft", action="store_true", help="use output/schema.draft.json if schema.json is missing")
        p.add_argument("--force", action="store_true", help="redo cached work")
        if name == "extract":
            p.add_argument("--only", nargs="+", help="extract only these reports")
            p.add_argument("--only-gold", action="store_true", help="extract only the reports listed in gold.csv")

    p = sub.add_parser("table", help="step 5: comparison table")
    _common(p)
    p.add_argument("--accept-draft", action="store_true")

    p = sub.add_parser("evaluate", help="score extractions against gold.csv")
    _common(p)
    p.add_argument("--accept-draft", action="store_true")
    return ap


def make_llm(cfg: Config):
    from .llm import OllamaLLM

    llm = OllamaLLM(cfg)
    llm.ping()
    return llm


SETTING_KEYS = (
    "reports_dir", "output_dir", "schema_file", "gold_file", "model", "num_ctx", "ollama_host",
    "topic", "parser", "pdf_strategy", "sample_size", "min_support",
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    if not args.verbose:
        for noisy in ("httpx", "unstructured", "pdfminer", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    overrides = {k: getattr(args, k, None) for k in SETTING_KEYS}
    cfg = Config.load(args.config, overrides)

    try:
        if args.command == "check":
            llm = make_llm(cfg)
            log.info("Ollama is running and %s is available (num_ctx=%d)", llm.model, llm.num_ctx)
        elif args.command == "parse":
            parse.run(cfg, force=args.force)
        elif args.command == "discover":
            schema_discovery.run(cfg, make_llm(cfg), sample=args.sample, force=args.force)
            _after_discover(cfg, args.accept)
        elif args.command == "extract":
            only = args.only
            if args.only_gold:
                only = _gold_files(cfg)
            extract.run(cfg, make_llm(cfg), accept_draft=args.accept_draft, only=only, force=args.force)
        elif args.command == "table":
            table.run(cfg, accept_draft=args.accept_draft)
        elif args.command == "evaluate":
            evaluate.run(cfg, accept_draft=args.accept_draft)
        elif args.command == "run":
            parse.run(cfg, force=args.force)
            llm = make_llm(cfg)
            if not cfg.schema_file.exists():
                schema_discovery.run(cfg, llm, force=args.force)
                if not args.accept_draft:
                    _after_discover(cfg, accept=False)
                    return 0
            extract.run(cfg, llm, accept_draft=args.accept_draft, force=args.force)
            table.run(cfg, accept_draft=args.accept_draft)
            if cfg.gold_file.exists():
                evaluate.run(cfg, accept_draft=args.accept_draft)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    return 0


def _after_discover(cfg: Config, accept: bool) -> None:
    if accept:
        shutil.copyfile(cfg.schema_draft_file, cfg.schema_file)
        log.info("Copied the draft to %s", cfg.schema_file)
        return
    log.info(
        "Next: review %s (all candidates are in %s), edit names, types, descriptions and "
        "category options, then save it as %s and run 'extract'.",
        cfg.schema_draft_file, cfg.schema_candidates_file, cfg.schema_file,
    )


def _gold_files(cfg: Config) -> list[str]:
    import pandas as pd

    if not cfg.gold_file.exists():
        raise RuntimeError(f"--only-gold needs {cfg.gold_file}")
    return pd.read_csv(cfg.gold_file, dtype=str, keep_default_na=False)["source_file"].tolist()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
