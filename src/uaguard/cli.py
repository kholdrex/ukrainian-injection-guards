import argparse
import json

from .config import CLASSIFIERS, DATA, LLM_GUARDS, RESULTS, SECOND_TRANSLATOR, TRANSLATOR
from .inputs import build_inputs
from .sampling import read_jsonl, write_samples
from .translation import translate_all, write_jsonl

ITEMS = DATA / "items.jsonl"
TRANSLATED = DATA / "items_translated.jsonl"
INPUTS = DATA / "inputs.jsonl"
SCORES = RESULTS / "scores.jsonl"


def translate() -> None:
    items = read_jsonl(TRANSLATED) if TRANSLATED.exists() else read_jsonl(ITEMS)
    translate_all(items, "en", "uk", TRANSLATOR, "en", "uk")
    main = [item for item in items if item["group"] == "main"]
    translate_all(main, "en", "uk_alt", SECOND_TRANSLATOR, "en", "uk")
    translate_all(main, "uk", "pivot", SECOND_TRANSLATOR, "uk", "en")
    write_jsonl(items, TRANSLATED)
    write_jsonl(build_inputs(items), INPUTS)


def score(names: list[str]) -> None:
    from .detectors import Classifier, OllamaGuard

    inputs = read_jsonl(INPUTS)
    done = set()
    if SCORES.exists():
        done = {(row["detector"], row["input_id"]) for row in read_jsonl(SCORES)}
    RESULTS.mkdir(exist_ok=True)
    for name in names:
        todo = [row for row in inputs if (name, row["input_id"]) not in done]
        if not todo:
            continue
        detector = Classifier(CLASSIFIERS[name]) if name in CLASSIFIERS else OllamaGuard(LLM_GUARDS[name])
        chunk = 64
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            with SCORES.open("a") as handle:
                for row, result in zip(batch, detector.score([r["text"] for r in batch])):
                    handle.write(json.dumps({"detector": name, "input_id": row["input_id"], **result},
                                            ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="uaguard")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample")
    sub.add_parser("translate")
    run = sub.add_parser("score")
    run.add_argument("detectors", nargs="+", choices=[*CLASSIFIERS, *LLM_GUARDS])
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "sample":
        write_samples(ITEMS)
    elif args.command == "translate":
        translate()
    elif args.command == "score":
        score(args.detectors)
    else:
        from .report import report
        report()
