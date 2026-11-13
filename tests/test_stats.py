import numpy as np
import pytest
from scipy import stats as sps

from stats_analytics_mcp import stats as st
from stats_analytics_mcp.db import QueryError


def test_compare_groups_matches_scipy_and_ci_covers_truth():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 400)
    b = rng.normal(9, 2, 380)
    out = st.compare_groups(a, b, "A", "B")
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert out["welch_p"] == pytest.approx(ref.pvalue, rel=1e-4, abs=1e-8)
    lo, hi = out["ci95_mean_difference"]
    assert lo < 1.0 < hi  # true difference is 1
    assert 0.3 < out["hedges_g"] < 0.7
    assert out["mann_whitney_p"] < 0.001


def test_compare_groups_rejects_tiny_or_constant():
    with pytest.raises(QueryError):
        st.compare_groups(np.array([1.0]), np.array([1.0, 2.0]), "A", "B")
    with pytest.raises(QueryError):
        st.compare_groups(np.ones(5), np.ones(6), "A", "B")


def test_small_sample_warning_present():
    out = st.compare_groups(np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 4, 9]), "A", "B")
    assert any("fewer than 30" in w for w in out["warnings"])


def test_proportions_match_scipy():
    out = st.compare_proportions(150, 1500, 188, 1500, "A", "B")
    _, p = sps.fisher_exact([[150, 1350], [188, 1312]])
    assert out["fisher_exact_p"] == pytest.approx(p, rel=1e-6)
    lo, hi = out["ci95_rate_difference_newcombe"]
    assert lo < out["rate_difference_a_minus_b"] < hi
    assert out["relative_risk"] < 1


def test_proportions_zero_events_still_work():
    out = st.compare_proportions(0, 20, 3, 20, "A", "B")
    assert out["relative_risk"] is None
    assert any("fewer than 5" in w for w in out["warnings"])


def test_correlation_and_ci():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    y = 0.6 * x + rng.normal(scale=0.8, size=300)
    out = st.correlation(x, y)
    lo, hi = out["ci95_pearson_r"]
    assert lo < out["pearson_r"] < hi and out["pearson_r"] > 0.4


def test_correlation_constant_column_rejected():
    with pytest.raises(QueryError):
        st.correlation(np.ones(10), np.arange(10.0))


def test_adjust_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert st.adjust_p_values(p, "holm")["adjusted_p"] == [0.03, 0.06, 0.06, 0.02]
    assert st.adjust_p_values(p, "bh")["adjusted_p"] == [0.02, 0.04, 0.04, 0.02]
    assert st.adjust_p_values(p, "bonferroni")["adjusted_p"] == [0.04, 0.16, 0.12, 0.02]


@pytest.mark.parametrize("bad", [[], [1.5], [-0.1], ["a"]])
def test_adjust_rejects_bad_input(bad):
    with pytest.raises(QueryError):
        st.adjust_p_values(bad)


def test_profile_numeric_and_categorical():
    num = st.profile([1, 2, 3, None, 5])
    assert num["kind"] == "numeric" and num["missing"] == 1 and num["median"] == 2.5
    cat = st.profile(["a", "b", "a", None])
    assert cat["kind"] == "categorical" and cat["top_values"][0] == {"value": "a", "count": 2}
