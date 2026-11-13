"""Statistical helpers. Every function returns numbers plus plain-language warnings."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from .db import QueryError, clean_value


def to_float_array(values: list[Any]) -> tuple[np.ndarray, int]:
    """Keep numeric, finite values. Return (array, number of values dropped)."""
    kept = [
        float(v) for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]  # fmt: skip
    return np.asarray(kept, dtype=float), len(values) - len(kept)


def is_numeric(values: list[Any]) -> bool:
    present = [v for v in values if v is not None]
    return bool(present) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in present
    )


def _round(x: float | None, digits: int = 6) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), digits)


def profile(values: list[Any]) -> dict[str, Any]:
    total = len(values)
    present = [v for v in values if v is not None]
    result: dict[str, Any] = {"rows": total, "missing": total - len(present)}
    if not present:
        result["kind"] = "empty"
        return result
    if is_numeric(values):
        arr, dropped = to_float_array(present)
        q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95]) if arr.size else []
        result.update(
            kind="numeric",
            non_finite_dropped=dropped,
            mean=_round(arr.mean()),
            std=_round(arr.std(ddof=1)) if arr.size > 1 else None,
            min=_round(arr.min()),
            p05=_round(q[0]), p25=_round(q[1]), median=_round(q[2]),
            p75=_round(q[3]), p95=_round(q[4]),
            max=_round(arr.max()),
            skewness=_round(stats.skew(arr)) if arr.size > 2 else None,
        )  # fmt: skip
    else:
        counts: dict[str, int] = {}
        for v in present:
            key = str(clean_value(str(v)))
            counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        result.update(
            kind="categorical",
            distinct=len(counts),
            top_values=[{"value": k, "count": c} for k, c in top],
        )
    return result


def compare_groups(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict[str, Any]:
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        raise QueryError(f"Each group needs at least 2 numeric values (got {n1} and {n2}).")
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    diff = m1 - m2
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        raise QueryError("Both groups are constant, so a test is not meaningful.")
    # Welch t-test with a 95% confidence interval for the mean difference.
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_stat = diff / se
    p_welch = float(2 * stats.t.sf(abs(t_stat), df))
    t_crit = float(stats.t.ppf(0.975, df))
    ci = (diff - t_crit * se, diff + t_crit * se)
    # Hedges' g (standardised mean difference, small-sample corrected).
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    g = (diff / pooled_sd) * (1 - 3 / (4 * (n1 + n2) - 9)) if pooled_sd > 0 else None
    # Mann-Whitney U and Cliff's delta (probability-based effect size, robust to outliers).
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = 2 * float(u.statistic) / (n1 * n2) - 1

    warnings: list[str] = []
    if min(n1, n2) < 30:
        warnings.append("A group has fewer than 30 values, so results are unstable.")
    if max(n1, n2) / min(n1, n2) > 3:
        warnings.append("Group sizes differ by more than 3 times.")
    for label, arr in ((label_a, a), (label_b, b)):
        if arr.size > 2 and abs(float(stats.skew(arr))) > 1:
            warnings.append(f"Group '{label}' is strongly skewed; prefer the Mann-Whitney result.")
    if (p_welch < 0.05) != (u.pvalue < 0.05):
        warnings.append("Welch and Mann-Whitney disagree about significance at 0.05.")
    warnings.append(
        "A p-value is not the probability that the groups are equal. "
        "With observational data the difference may not be causal. "
        "If you run many comparisons, use adjust_p_values."
    )
    return {
        "group_a": label_a, "group_b": label_b, "n_a": n1, "n_b": n2,
        "mean_a": _round(m1), "mean_b": _round(m2),
        "median_a": _round(np.median(a)), "median_b": _round(np.median(b)),
        "mean_difference_a_minus_b": _round(diff),
        "ci95_mean_difference": [_round(ci[0]), _round(ci[1])],
        "welch_t": _round(t_stat), "welch_df": _round(df, 2), "welch_p": _round(p_welch, 8),
        "hedges_g": _round(g),
        "mann_whitney_u": _round(float(u.statistic), 2), "mann_whitney_p": _round(float(u.pvalue), 8),
        "cliffs_delta": _round(delta),
        "warnings": warnings,
    }  # fmt: skip


def wilson_interval(x: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = x / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compare_proportions(
    x1: int, n1: int, x2: int, n2: int, label_a: str, label_b: str
) -> dict[str, Any]:
    if n1 < 1 or n2 < 1:
        raise QueryError("Each group needs at least 1 observation.")
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    l1, u1 = wilson_interval(x1, n1)
    l2, u2 = wilson_interval(x2, n2)
    # Newcombe hybrid-score interval for the difference of two proportions.
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    table = [[x1, n1 - x1], [x2, n2 - x2]]
    odds, p_fisher = stats.fisher_exact(table, alternative="two-sided")
    cells = [x1 + 0.5, n1 - x1 + 0.5, x2 + 0.5, n2 - x2 + 0.5]  # Haldane-Anscombe for the CI
    log_or = math.log((cells[0] * cells[3]) / (cells[1] * cells[2]))
    se_or = math.sqrt(sum(1 / c for c in cells))
    or_ci = (math.exp(log_or - 1.959964 * se_or), math.exp(log_or + 1.959964 * se_or))
    rr = rr_ci = None
    if x1 > 0 and x2 > 0:
        rr = p1 / p2
        se_rr = math.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
        rr_ci = (
            math.exp(math.log(rr) - 1.959964 * se_rr),
            math.exp(math.log(rr) + 1.959964 * se_rr),
        )
    warnings: list[str] = []
    if min(x1, n1 - x1, x2, n2 - x2) < 5:
        warnings.append(
            "A cell has fewer than 5 observations; rely on the exact test and wide intervals."
        )
    if min(n1, n2) < 30:
        warnings.append("A group has fewer than 30 observations.")
    warnings.append(
        "A p-value is not the probability that the groups are equal. "
        "Check that groups were assigned randomly before reading the difference as an effect."
    )
    return {
        "group_a": label_a, "group_b": label_b,
        "events_a": x1, "n_a": n1, "rate_a": _round(p1),
        "events_b": x2, "n_b": n2, "rate_b": _round(p2),
        "rate_difference_a_minus_b": _round(diff),
        "ci95_rate_difference_newcombe": [_round(lo), _round(hi)],
        "odds_ratio": _round(float(odds)),
        "ci95_odds_ratio": [_round(or_ci[0]), _round(or_ci[1])],
        "relative_risk": _round(rr),
        "ci95_relative_risk": [_round(rr_ci[0]), _round(rr_ci[1])] if rr_ci else None,
        "fisher_exact_p": _round(float(p_fisher), 8),
        "warnings": warnings,
    }  # fmt: skip


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = x.size
    if n < 4:
        raise QueryError(f"Need at least 4 complete pairs (got {n}).")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise QueryError("One of the columns is constant, so correlation is undefined.")
    r, p_r = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    ci = (math.tanh(z - 1.959964 * se), math.tanh(z + 1.959964 * se))
    warnings = ["Correlation does not show that one variable causes the other."]
    if n < 30:
        warnings.append("Fewer than 30 pairs, so the estimate is unstable.")
    if abs(r - rho) > 0.2:
        warnings.append(
            "Pearson and Spearman differ a lot: look for outliers or a curved relationship."
        )
    return {
        "n_pairs": n,
        "pearson_r": _round(float(r)), "pearson_p": _round(float(p_r), 8),
        "ci95_pearson_r": [_round(ci[0]), _round(ci[1])],
        "spearman_rho": _round(float(rho)), "spearman_p": _round(float(p_rho), 8),
        "warnings": warnings,
    }  # fmt: skip


def adjust_p_values(
    p_values: list[float], method: str = "holm", alpha: float = 0.05
) -> dict[str, Any]:
    if not p_values:
        raise QueryError("Provide at least one p-value.")
    if len(p_values) > 1_000:
        raise QueryError("At most 1000 p-values are accepted.")
    if any((not isinstance(p, (int, float))) or not (0 <= p <= 1) for p in p_values):
        raise QueryError("Every p-value must be a number between 0 and 1.")
    m = len(p_values)
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    if method == "bonferroni":
        adj_sorted = np.minimum(ranked * m, 1.0)
    elif method == "holm":
        adj_sorted = np.minimum(np.maximum.accumulate(ranked * (m - np.arange(m))), 1.0)
    elif method == "bh":
        raw = ranked * m / (np.arange(m) + 1)
        adj_sorted = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    else:
        raise QueryError("method must be one of: bonferroni, holm, bh")
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return {
        "method": method, "alpha": alpha, "tests": m,
        "adjusted_p": [_round(float(v), 8) for v in adjusted],
        "significant": [bool(v <= alpha) for v in adjusted],
        "note": (
            "bh controls the false discovery rate; holm and bonferroni control the "
            "chance of any false positive."
        ),
    }  # fmt: skip
