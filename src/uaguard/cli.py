import argparse
import json

from .config import CLASSIFIERS, DATA, LLM_GUARDS, LLM_TRANSLATORS, MT_BACK, MT_PRIMARY, MT_SECOND, RESULTS
from .inputs import build_inputs
from .manifest import text_sha, write_manifest
from .sampling import read_jsonl, write_samples
from .translation import MachineTranslator, translate_all_llm, translate_mt, write_jsonl

ITEMS = DATA / "items.jsonl"
TRANSLATED = DATA / "items_translated.jsonl"
INPUTS = DATA / "inputs.jsonl"
SCORES = RESULTS / "scores.jsonl"


LLM_PILOT = DATA / "llm_translation_pilot.jsonl"


def translate() -> None:
    items = read_jsonl(ITEMS)
    main = [item for item in items if item["group"] == "main"]
    translate_mt(items, "en", "uk", MachineTranslator(*MT_PRIMARY), "uk")
    translate_mt(main, "en", "uk_alt", MachineTranslator(*MT_SECOND), "uk")
    translate_mt(main, "uk", "pivot", MachineTranslator(*MT_BACK), "en")
    write_jsonl(items, TRANSLATED)
    write_jsonl(build_inputs(items), INPUTS)
    manifest()


def translate_llm() -> None:
    items = read_jsonl(LLM_PILOT) if LLM_PILOT.exists() else read_jsonl(ITEMS)
    main = [item for item in items if item["group"] == "main"]
    translate_all_llm(items, "en", "uk", LLM_TRANSLATORS[0], "en", "uk")
    translate_all_llm(main, "en", "uk_alt", LLM_TRANSLATORS[1], "en", "uk")
    translate_all_llm(main, "uk", "pivot", LLM_TRANSLATORS[1], "uk", "en")
    write_jsonl(items, LLM_PILOT)
    if TRANSLATED.exists():
        manifest()


def manifest() -> None:
    """data/manifest.json from the local texts; the texts themselves are not part of the repository."""
    write_manifest(read_jsonl(TRANSLATED), read_jsonl(INPUTS), read_jsonl(LLM_PILOT) if LLM_PILOT.exists() else None)


def read_scores() -> list[dict]:
    rows = read_jsonl(SCORES) if SCORES.exists() else []
    if rows and not {"text_sha", "config_sha"} <= rows[0].keys():
        raise SystemExit(f"{SCORES} was written by an older version without text and configuration hashes; "
                         "move it aside and score again")
    return rows


def score(names: list[str]) -> None:
    from .detectors import Classifier, OllamaGuard, configuration

    inputs = read_jsonl(INPUTS)
    done = {(row["detector"], row["input_id"], row["text_sha"], row["config_sha"]) for row in read_scores()}
    RESULTS.mkdir(exist_ok=True)
    for name in names:
        config = text_sha(configuration(name))
        todo = [row for row in inputs if (name, row["input_id"], text_sha(row["text"]), config) not in done]
        if not todo:
            continue
        detector = Classifier(CLASSIFIERS[name]) if name in CLASSIFIERS else OllamaGuard(LLM_GUARDS[name])
        chunk = 64
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            with SCORES.open("a") as handle:
                for row, result in zip(batch, detector.score([r["text"] for r in batch])):
                    handle.write(json.dumps({"detector": name, "input_id": row["input_id"],
                                             "text_sha": text_sha(row["text"]), "config_sha": config, **result},
                                            ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="uaguard")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample")
    sub.add_parser("translate")
    sub.add_parser("translate-llm")
    sub.add_parser("manifest")
    run = sub.add_parser("score")
    run.add_argument("detectors", nargs="+", choices=[*CLASSIFIERS, *LLM_GUARDS])
    sub.add_parser("latency")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "sample":
        write_samples(ITEMS)
    elif args.command == "translate":
        translate()
    elif args.command == "translate-llm":
        translate_llm()
    elif args.command == "manifest":
        manifest()
    elif args.command == "score":
        score(args.detectors)
    elif args.command == "latency":
        from .latency import measure
        measure()
    else:
        from .report import report
        report()
