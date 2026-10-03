"""Tables and figures from data/inputs.jsonl and results/scores.jsonl."""
import json
import re
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from .config import BOOTSTRAP, CLASSIFIERS, DATA, LLM_GUARDS, RESULTS, SEED, TARGET_FPR
from .metrics import auroc, holm, mcnemar, paired_bootstrap, threshold_at_fpr, wilson
from .sampling import read_jsonl

DETECTORS = [*CLASSIFIERS, *LLM_GUARDS]
LABELS = {"protectai": "ProtectAI DeBERTa-v3", "mdeberta": "mDeBERTa-v3 (Proventra)",
          "testsavant": "TestSavant defender", "qwen7b": "Qwen2.5-7B guard", "gemma4b": "Gemma3-4B guard"}
SHORT = {"protectai": "ProtectAI", "mdeberta": "mDeBERTa", "testsavant": "TestSavant", "qwen7b": "Qwen2.5-7B",
         "gemma4b": "Gemma3-4B"}
MAIN_CONDITIONS = ("en", "uk", "uk_alt", "pivot")


def _load():
    from .cli import read_scores, text_sha
    from .detectors import configuration

    current = {row["input_id"]: text_sha(row["text"]) for row in read_jsonl(DATA / "inputs.jsonl")}
    scores = defaultdict(dict)
    for row in read_scores():
        if current.get(row["input_id"]) == row["text_sha"] and \
                row["config_sha"] == text_sha(configuration(row["detector"])):
            scores[row["detector"]][row["input_id"]] = row
    return {name: rows for name, rows in scores.items() if len(rows) == len(current)}


RUSSIAN_ONLY = re.compile(r"[ыэёъЫЭЁЪ]")
UKRAINIAN_ONLY = re.compile(r"[іїєґІЇЄҐ]")
CYRILLIC = re.compile(r"[а-яА-Я]")


def clean_ukrainian(text: str) -> bool:
    """No letters that exist only in Russian; used for the sensitivity analysis."""
    return not RUSSIAN_ONLY.search(text)


def _excluded(items):
    return {item["id"] for item in items if item["group"] == "main"
            and any(item[f"{field}_meta"]["flags"] for field in ("uk", "uk_alt", "pivot"))}


def _vector(scores, condition, ids):
    return np.array([scores[f"{i}/{condition}"]["score"] for i in ids])


def report() -> dict:
    scores = _load()
    items = read_jsonl(DATA / "items_translated.jsonl")
    excluded = _excluded(items)
    main_ids = [i["id"] for i in items if i["group"] == "main" and i["id"] not in excluded]
    all_main = [i["id"] for i in items if i["group"] == "main"]
    calib_ids = [i["id"] for i in items if i["group"] == "calibration"]
    primary = set(main_ids)
    clean_ids = [i["id"] for i in items
                 if i["id"] in primary and clean_ukrainian(i["uk"]) and clean_ukrainian(i["uk_alt"])]
    by_id = {i["id"]: i for i in items}
    embed_ids = [i["id"] for i in items if i["group"] == "carrier"]
    labels = {i["id"]: i["label"] for i in items}
    out = {"n_main": len(main_ids), "n_excluded": len(excluded), "n_clean_uk": len(clean_ids),
           "n_clean_uk_attacks": sum(labels[i] for i in clean_ids), "detectors": {}}

    for name in DETECTORS:
        if name not in scores:
            continue
        s = scores[name]
        d = {"thresholds": {}, "main": {}, "embedded": {}, "uk_calibrated": {}, "tokens": {}, "embedded_clean": {}}
        calib_en = _vector(s, "en", calib_ids)
        calib_uk = _vector(s, "uk", calib_ids)
        for target in TARGET_FPR:
            d["thresholds"][str(target)] = threshold_at_fpr(calib_en, target)
        for cohort, ids in (("primary", main_ids), ("all", all_main), ("clean_uk", clean_ids)):
            y = np.array([labels[i] for i in ids])
            block = {}
            for condition in MAIN_CONDITIONS:
                x = _vector(s, condition, ids)
                row = {"auroc": auroc(y, x)}
                for target, thr in d["thresholds"].items():
                    flagged = x > thr
                    tp, fp = int(flagged[y == 1].sum()), int(flagged[y == 0].sum())
                    row[target] = {"recall": tp / (y == 1).sum(), "recall_ci": wilson(tp, int((y == 1).sum())),
                                   "fpr": fp / (y == 0).sum(), "fpr_ci": wilson(fp, int((y == 0).sum()))}
                row["default"] = {"recall": float((x[y == 1] > 0.5).mean()), "fpr": float((x[y == 0] > 0.5).mean())}
                block[condition] = row
            en = _vector(s, "en", ids)
            for condition in ("uk", "uk_alt", "pivot"):
                other = _vector(s, condition, ids)
                for target, thr in d["thresholds"].items():
                    a, b = en > thr, other > thr
                    block[condition][target]["d_recall"] = float(b[y == 1].mean() - a[y == 1].mean())
                    block[condition][target]["d_recall_ci"] = paired_bootstrap(
                        a[y == 1].astype(float), b[y == 1].astype(float), BOOTSTRAP, SEED)
                    block[condition][target]["d_fpr"] = float(b[y == 0].mean() - a[y == 0].mean())
                    block[condition][target]["d_fpr_ci"] = paired_bootstrap(
                        a[y == 0].astype(float), b[y == 0].astype(float), BOOTSTRAP, SEED)
                    block[condition][target]["p_recall"] = mcnemar(a[y == 1], b[y == 1])
                    block[condition][target]["p_fpr"] = mcnemar(a[y == 0], b[y == 0])
            d["main"][cohort] = block

        thr = d["thresholds"]["0.05"]
        payloads = [i for i in items if i["group"] == "payload"]
        clean_pairs = [c for c, p in zip(embed_ids, payloads)
                       if clean_ukrainian(by_id[c]["uk"]) and clean_ukrainian(p["uk"]) and UKRAINIAN_ONLY.search(p["uk"])
                       and not CYRILLIC.search(p["en"]) and not CYRILLIC.search(by_id[c]["en"])]
        for condition in ("clean_en", "clean_uk", "en+en", "uk+en", "en+uk", "uk+uk"):
            d["embedded"][condition] = float((_vector(s, condition, embed_ids) > thr).mean())
            d["embedded_clean"][condition] = float((_vector(s, condition, clean_pairs) > thr).mean())
        d["embedded_clean"]["pairs"] = len(clean_pairs)
        thr_uk = threshold_at_fpr(calib_uk, 0.05)
        y = np.array([labels[i] for i in main_ids])
        x = _vector(s, "uk", main_ids)
        d["uk_calibrated"] = {"threshold": thr_uk, "recall": float((x[y == 1] > thr_uk).mean()),
                              "fpr": float((x[y == 0] > thr_uk).mean()),
                              "calibration_fpr_uk_at_en_threshold": float((calib_uk > thr).mean())}
        for condition in ("en", "uk"):
            rows = [s[f"{i}/{condition}"] for i in main_ids]
            d["tokens"][condition] = {"mean": float(np.mean([r["tokens"] for r in rows])),
                                      "truncated": int(sum(r["truncated"] for r in rows))}
        out["detectors"][name] = d

    names = [n for n in DETECTORS if n in out["detectors"]]
    for field in ("p_recall", "p_fpr"):
        raw = [out["detectors"][n]["main"]["primary"]["uk"]["0.05"][field] for n in names]
        for n, p in zip(names, holm(raw)):
            out["detectors"][n]["main"]["primary"]["uk"]["0.05"][field + "_holm"] = p
    out["llm_translation_pilot"] = _llm_pilot()
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    _figure(scores, main_ids, labels, {n: out['detectors'][n]['thresholds']['0.05'] for n in out['detectors']})
    return out


def _llm_pilot():
    path = DATA / "llm_translation_pilot.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    out = {}
    for field in ("uk", "uk_alt"):
        for label in (0, 1):
            subset = [r for r in rows if r["group"] == "main" and r["label"] == label and f"{field}_meta" in r]
            failed = sum(bool(r[f"{field}_meta"]["flags"]) for r in subset)
            out[f"{subset[0][f'{field}_meta']['model']}|label={label}"] = {"failed": failed, "n": len(subset)}
    return out


def _figure(scores, ids, labels, thresholds):
    from matplotlib.lines import Line2D

    names = [n for n in DETECTORS if n in scores]
    fig, axes = plt.subplots(1, len(names), figsize=(7.6, 2.3), sharey=True)
    y = np.array([labels[i] for i in ids])
    styles = {"en": ("-", "o", "English"), "uk": ("--", "^", "Ukrainian"), "pivot": (":", "s", "Back-translated")}
    for ax, name in zip(np.atleast_1d(axes), names):
        for condition, (style, marker, label) in styles.items():
            x = _vector(scores[name], condition, ids)
            fpr, tpr, _ = roc_curve(y, x)
            ax.plot(fpr, tpr, style, color="black", linewidth=1.1)
            flagged = x > thresholds[name]
            ax.plot(max(flagged[y == 0].mean(), 0.005), flagged[y == 1].mean(), marker, color="black",
                    markerfacecolor="white", markersize=5)
        ax.set_xscale("log")
        ax.set_xlim(0.005, 1)
        ax.set_title(SHORT[name], fontsize=9)
        ax.set_xlabel("FPR", fontsize=9)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Recall", fontsize=9)
    handles = [Line2D([], [], linestyle=style, marker=marker, color="black", markerfacecolor="white", label=label)
               for style, marker, label in styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTS / "roc.png", dpi=300)
    plt.close(fig)
