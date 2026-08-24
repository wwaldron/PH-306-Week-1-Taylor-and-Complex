import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import series


def test_harmonic_sums_first_terms():
    """The harmonic series should sum the first n reciprocal terms."""
    assert series.harmonic(0) == pytest.approx(0.0)
    assert series.harmonic(1) == pytest.approx(1.0)
    assert series.harmonic(5) == pytest.approx(2.283333333333333)


def test_boas_1_13_4_matches_log1p():
    """The logarithm series should match ln(1 + x) for valid x values."""
    assert series.boas_1_13_4(0.0) == pytest.approx(0.0)
    assert series.boas_1_13_4(0.5) == pytest.approx(math.log(1.5), rel=1e-7, abs=1e-12)
    assert series.boas_1_13_4(-0.5) == pytest.approx(math.log(0.5), rel=1e-7, abs=1e-12)


def test_boas_1_13_22_matches_rational_exp_expression():
    """The function should evaluate the closed form exp(x) / (1 - x)."""
    assert series.boas_1_13_22(0.0) == pytest.approx(1.0)
    assert series.boas_1_13_22(0.5) == pytest.approx(math.exp(0.5) / (1.0 - 0.5), rel=1e-7, abs=1e-12)
    assert series.boas_1_13_22(-0.5) == pytest.approx(math.exp(-0.5) / (1.0 + 0.5), rel=1e-7, abs=1e-12)


def test_boas_1_13_22_plot_creates_a_plot():
    """The plotting helper should generate a figure with at least one plotted curve."""
    plt.close("all")
    result = series.boas_1_13_22_plot(0.5, 5)

    assert plt.gcf().axes
    assert len(plt.gca().lines) >= 1
    if result is not None:
        assert hasattr(result, "lines") or hasattr(result, "axes")

    plt.close("all")


def test_boas_1_16_1c_matches_harmonic_overhang_rule():
    """The stack problem uses the harmonic series to determine the minimum books."""
    assert int(series.boas_1_16_1c(0.5)) == 1
    assert int(series.boas_1_16_1c(0.75)) == 2
    assert int(series.boas_1_16_1c(1.2)) >= 3


def test_cos_apprx_matches_cosine_taylor_series():
    """The cosine approximation should converge to the expected value for standard inputs."""
    value, n_terms = series.cos_apprx(0.0, rel_tol=1e-12, max_iter=1000)
    assert value == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert n_terms >= 1

    value, n_terms = series.cos_apprx(math.pi / 3.0, rel_tol=1e-12, max_iter=1000)
    assert value == pytest.approx(0.5, rel=1e-8, abs=1e-8)
    assert n_terms >= 1

    neg_value, _ = series.cos_apprx(-math.pi / 3.0, rel_tol=1e-12, max_iter=1000)
    assert neg_value == pytest.approx(value, rel=1e-8, abs=1e-8)
