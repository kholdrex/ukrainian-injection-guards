import argparse
import hashlib
import json

from .config import CLASSIFIERS, DATA, LLM_GUARDS, LLM_TRANSLATORS, MT_BACK, MT_PRIMARY, MT_SECOND, RESULTS
from .inputs import build_inputs
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


def translate_llm() -> None:
    items = read_jsonl(LLM_PILOT) if LLM_PILOT.exists() else read_jsonl(ITEMS)
    main = [item for item in items if item["group"] == "main"]
    translate_all_llm(items, "en", "uk", LLM_TRANSLATORS[0], "en", "uk")
    translate_all_llm(main, "en", "uk_alt", LLM_TRANSLATORS[1], "en", "uk")
    translate_all_llm(main, "uk", "pivot", LLM_TRANSLATORS[1], "uk", "en")
    write_jsonl(items, LLM_PILOT)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def score(names: list[str]) -> None:
    from .detectors import Classifier, OllamaGuard

    inputs = read_jsonl(INPUTS)
    done = set()
    if SCORES.exists():
        done = {(row["detector"], row["input_id"], row["text_sha"]) for row in read_jsonl(SCORES)}
    RESULTS.mkdir(exist_ok=True)
    for name in names:
        todo = [row for row in inputs if (name, row["input_id"], text_sha(row["text"])) not in done]
        if not todo:
            continue
        detector = Classifier(CLASSIFIERS[name]) if name in CLASSIFIERS else OllamaGuard(LLM_GUARDS[name])
        chunk = 64
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            with SCORES.open("a") as handle:
                for row, result in zip(batch, detector.score([r["text"] for r in batch])):
                    handle.write(json.dumps({"detector": name, "input_id": row["input_id"],
                                             "text_sha": text_sha(row["text"]), **result}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="uaguard")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample")
    sub.add_parser("translate")
    sub.add_parser("translate-llm")
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
    elif args.command == "score":
        score(args.detectors)
    elif args.command == "latency":
        from .latency import measure
        measure()
    else:
        from .report import report
        report()
