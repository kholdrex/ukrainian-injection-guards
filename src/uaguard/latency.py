"""Per-request latency of each defence, one input at a time, measured on an otherwise idle machine."""
import json
import random
import time

import numpy as np

from .config import CLASSIFIERS, DATA, LLM_GUARDS, MT_BACK, RESULTS, SEED
from .detectors import Classifier, OllamaGuard
from .sampling import read_jsonl
from .translation import MachineTranslator

N_INPUTS = 100


def _timed(fn, texts):
    seconds = []
    for text in texts:
        began = time.perf_counter()
        fn([text])
        seconds.append(time.perf_counter() - began)
    return {"median_ms": 1000 * float(np.median(seconds)), "p95_ms": 1000 * float(np.percentile(seconds, 95))}


def measure() -> None:
    inputs = [row for row in read_jsonl(DATA / "inputs.jsonl") if row["group"] == "main"]
    sample = random.Random(SEED).sample([row for row in inputs if row["condition"] == "uk"], N_INPUTS)
    texts = [row["text"] for row in sample]
    out = {}
    for name, model_id in CLASSIFIERS.items():
        for device in ("cpu", "mps"):
            detector = Classifier(model_id, device=device)
            detector.score(texts[:5])
            out[f"{name}|{device}"] = _timed(detector.score, texts)
    for name, model in LLM_GUARDS.items():
        guard = OllamaGuard(model)
        guard.score(texts[:3])
        out[name] = _timed(guard.score, texts)
    translator = MachineTranslator(*MT_BACK)
    out["back_translation|cpu"] = _timed(translator._translate, texts)
    (RESULTS / "latency.json").write_text(json.dumps(out, indent=1))
