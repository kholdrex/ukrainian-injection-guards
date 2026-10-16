"""Text-free record of the evaluation data: everything `report` needs, without the third-party texts.

The prompts come from datasets whose terms do not all allow redistribution, so the repository keeps their
identifiers, labels, translation checks, the letter classes of every text used by the character screens and the hash
of every scored input; the texts themselves are rebuilt with `uaguard sample` and `uaguard translate`.
"""
import hashlib
import json
import re

from .config import DATA

MANIFEST = DATA / "manifest.json"
TEXT_FIELDS = ("en", "uk", "uk_alt", "pivot")
LETTERS = {"cyrillic": re.compile(r"[а-яА-Я]"), "russian_only": re.compile(r"[ыэёъЫЭЁЪ]"),
           "ukrainian_only": re.compile(r"[іїєґІЇЄҐ]")}


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _drop_texts(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in TEXT_FIELDS}


def _item(row: dict) -> dict:
    letters = {field: {name: bool(pattern.search(row[field])) for name, pattern in LETTERS.items()}
               for field in TEXT_FIELDS if field in row}
    return {**_drop_texts(row), "letters": letters}


def write_manifest(items: list[dict], inputs: list[dict], llm_pilot: list[dict] | None = None) -> None:
    """`llm_pilot` None keeps the LLM translation records of the current manifest."""
    if llm_pilot is None:
        pilot = read_manifest()["llm_pilot"] if MANIFEST.exists() else []
    else:
        pilot = [_drop_texts(row) for row in llm_pilot]
    manifest = {"items": [_item(item) for item in items],
                "inputs": {row["input_id"]: text_sha(row["text"]) for row in inputs},
                "llm_pilot": pilot}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=0) + "\n")


def read_manifest() -> dict:
    return json.loads(MANIFEST.read_text())
