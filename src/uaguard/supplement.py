"""Analyses added after review: results/supplement.json from the same manifest and scores as summary.json."""
import json

import numpy as np

from .config import BOOTSTRAP, RESULTS, SEED
from .metrics import auroc, mcnemar, threshold_at_fpr, wilson

CONDITIONS = ("en", "uk", "uk_alt", "pivot")


def stratified_auroc(y, columns, resamples=BOOTSTRAP, seed=SEED):
    """AUROC of each column with percentile intervals; attacks and benign prompts are resampled separately and the
    same resample is used for every column, so differences between columns are paired."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = {name: [] for name in columns}
    for _ in range(resamples):
        index = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for name, x in columns.items():
            draws[name].append(auroc(y[index], x[index]))
    out = {}
    for name, x in columns.items():
        values = np.array(draws[name])
        out[name] = {"auroc": auroc(y, x), "ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}
        if name != "en":
            diff = values - np.array(draws["en"])
            out[name]["d_auroc"] = out[name]["auroc"] - auroc(y, columns["en"])
            out[name]["d_auroc_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    return out


def supplement(scores, items, main_ids, embed_ids, vector) -> dict:
    labels = {i["id"]: i["label"] for i in items}
    y = np.array([labels[i] for i in main_ids])
    payloads = [i for i in items if i["group"] == "payload"]
    out = {}
    for name, s in scores.items():
        d = {}
        columns = {c: vector(s, c, main_ids) for c in CONDITIONS}
        d["auroc"] = stratified_auroc(y, columns)
        d["recall_at_own_test_fpr_0.05"] = {}
        for c, x in columns.items():
            thr = threshold_at_fpr(x[y == 0], 0.05)
            d["recall_at_own_test_fpr_0.05"][c] = {"recall": float((x[y == 1] > thr).mean()),
                                                   "fpr": float((x[y == 0] > thr).mean())}
        calib = [i["id"] for i in items if i["group"] == "calibration"]
        thr_en = threshold_at_fpr(vector(s, "en", calib), 0.05)
        kept = [i for i in main_ids if labels[i] and not s[f"{i}/uk"]["truncated"]]
        d["untruncated_attacks"] = {"n": len(kept)}
        for c in ("en", "uk"):
            d["untruncated_attacks"][f"recall_{c}"] = float((vector(s, c, kept) > thr_en).mean())
        flagged = {c: vector(s, c, embed_ids) > thr_en for c in ("clean_en", "clean_uk", "en+en", "uk+en", "en+uk",
                                                                   "uk+uk")}
        d["embedded"] = {c: {"rate": float(f.mean()), "ci": list(wilson(int(f.sum()), len(f)))} for c, f in flagged.items()}
        d["embedded"]["p_en+en_vs_en+uk"] = mcnemar(flagged["en+en"], flagged["en+uk"])
        d["embedded"]["p_uk+en_vs_uk+uk"] = mcnemar(flagged["uk+en"], flagged["uk+uk"])
        d["score_ties"] = {c: {"zero": int((x == 0).sum()), "half": int((x == 0.5).sum())} for c, x in columns.items()}
        out[name] = d
    out["n_distinct_payloads"] = len({p["source_id"] for p in payloads})
    (RESULTS / "supplement.json").write_text(json.dumps(out, indent=1))
    return out
