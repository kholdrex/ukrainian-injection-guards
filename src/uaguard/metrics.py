import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def threshold_at_fpr(benign_scores: np.ndarray, target: float) -> float:
    """Smallest threshold whose rule `score > threshold` flags at most `target` of the benign scores."""
    ordered = np.sort(benign_scores)[::-1]
    allowed = int(np.floor(target * len(ordered)))
    return float(ordered[allowed]) if allowed < len(ordered) else float(ordered[-1])


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value for paired binary decisions."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return binomtest(only_a, only_a + only_b, 0.5).pvalue


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    """Percentile interval for mean(b) - mean(a) over paired items."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(a), size=(resamples, len(a)))
    diffs = b[index].mean(axis=1) - a[index].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))
