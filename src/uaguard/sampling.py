"""Fixed samples of English source texts drawn from public datasets."""
import json
import random
import re

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import (
    CARRIER_MAX_TOKENS, DATA, GANDALF, LENGTH_TOKENIZER, MAX_TOKENS, N_ATTACK, N_BENIGN, N_CARRIERS,
    PROMPTSHIELD, REVISIONS, SEED,
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _rows(split, prefix, tokenizer, max_tokens):
    seen, rows = set(), []
    for index, record in enumerate(split):
        text = record.get("prompt") or record.get("text")
        key = _normalise(text)
        if not key or key in seen or len(tokenizer(text)["input_ids"]) > max_tokens:
            continue
        seen.add(key)
        rows.append({"source_id": f"{prefix}:{index}", "label": int(record.get("label", 1)), "en": text})
    return rows


def build_samples() -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(LENGTH_TOKENIZER, revision=REVISIONS[LENGTH_TOKENIZER])
    shield = load_dataset(PROMPTSHIELD, revision=REVISIONS[PROMPTSHIELD])
    gandalf = load_dataset(GANDALF, revision=REVISIONS[GANDALF])
    rng = random.Random(SEED)

    calibration = [r for r in _rows(shield["validation"], "promptshield/validation", tokenizer, MAX_TOKENS)
                   if r["label"] == 0]
    test = _rows(shield["test"], "promptshield/test", tokenizer, MAX_TOKENS)
    attacks = rng.sample([r for r in test if r["label"] == 1], N_ATTACK)
    benign_pool = [r for r in test if r["label"] == 0]
    rng.shuffle(benign_pool)
    benign = benign_pool[:N_BENIGN]
    short = {r["source_id"] for r in _rows(shield["test"], "promptshield/test", tokenizer, CARRIER_MAX_TOKENS)}
    carriers = [r for r in benign_pool[N_BENIGN:] if r["source_id"] in short][:N_CARRIERS]
    payload_pool = (_rows(gandalf["validation"], "gandalf/validation", tokenizer, 60)
                    + _rows(gandalf["test"], "gandalf/test", tokenizer, 60))
    payloads = rng.sample(payload_pool, N_CARRIERS)

    items = []
    for group, rows in (("calibration", calibration), ("main", attacks + benign),
                        ("carrier", carriers), ("payload", payloads)):
        for row in rows:
            items.append({"id": f"{group}-{len(items):04d}", "group": group, **row})
    return items


def write_samples(path=DATA / "items.jsonl") -> None:
    items = build_samples()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]
