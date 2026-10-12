"""1-X-2 aus Torverteilungen, Mehrverlust und die Gegenmodelle gegen Handwerte."""

import math

import numpy as np
import pytest

import trm_outcomes as O


def test_equal_rates_hand_values():
    # λ_Heim = λ_Gast = 1: P(Remis) = e⁻² Σ 1/(k!)² = e⁻² I₀(2) = 0,30851, P(Heim) = P(Gast) = (1 − 0,30851)/2 = 0,34575
    p = O.outcome_probabilities([1.0], [1.0])[0]
    i0_2 = sum(1.0 / math.factorial(k) ** 2 for k in range(30))
    assert p[1] == pytest.approx(math.exp(-2) * i0_2, abs=1e-9) and p[1] == pytest.approx(0.30851, abs=2e-5)
    assert p[0] == pytest.approx(0.34575, abs=2e-5) and p[0] == pytest.approx(p[2], abs=1e-12)


def test_probabilities_sum_to_one_and_are_symmetric_under_swapping_sides():
    lh, la = np.array([0.7, 1.9, 4.0]), np.array([1.2, 0.5, 4.5])
    for r in (None, 2.0):
        p = O.outcome_probabilities(lh, la, r)
        q = O.outcome_probabilities(la, lh, r)
        assert p.sum(axis=1) == pytest.approx(np.ones(3))
        assert p[:, 0] == pytest.approx(q[:, 2]) and p[:, 1] == pytest.approx(q[:, 1])


def test_count_pmf_has_the_right_mean_and_variance():
    for r, var in ((None, 3.0), (2.0, 3.0 + 9.0 / 2.0)):
        pmf = O.count_pmf([3.0], r, cap=300)[0]
        k = np.arange(300)
        mean = (k * pmf).sum()
        assert pmf.sum() == pytest.approx(1.0, abs=1e-9) and mean == pytest.approx(3.0, abs=1e-6)
        assert ((k - mean) ** 2 * pmf).sum() == pytest.approx(var, abs=1e-5)


def test_negative_binomial_approaches_poisson_for_a_large_form():
    p = O.outcome_probabilities([1.5], [1.1])
    q = O.outcome_probabilities([1.5], [1.1], 1e7)
    assert q == pytest.approx(p, abs=1e-5)


def test_ordered_pairs_and_true_probabilities_shape_and_home_advantage():
    I, J = O.ordered_pairs(5)
    assert len(I) == 20 and (I != J).all()
    flat = O.true_probabilities(np.zeros(5), 1.4, 0.0)
    home = O.true_probabilities(np.zeros(5), 1.4, 0.4)
    assert flat.shape == (20, 3) and flat[:, 0] == pytest.approx(flat[:, 2])     # ohne Heimvorteil und gleichstark: symmetrisch
    assert (home[:, 0] > home[:, 2]).all()


def test_kl_hand_value_and_zero_for_the_truth():
    t = np.array([[0.5, 0.5, 0.0]])
    q = np.array([[0.25, 0.5, 0.25]])
    assert O.kl_to_truth(t, t) == pytest.approx(0.0, abs=1e-12)
    assert O.kl_to_truth(t, q) == pytest.approx(0.5 * math.log(2), abs=1e-9)
    impossible = O.kl_to_truth(t, np.array([[0.0, 1.0, 0.0]]))            # Heimsieg als unmöglich vorhergesagt: auf 10⁻¹² geschnitten, nicht unendlich
    assert math.isfinite(impossible)
    assert impossible == pytest.approx(0.5 * (math.log(0.5) - math.log(1.0)) + 0.5 * (math.log(0.5) - math.log(1e-12)), abs=1e-9)


def test_bt_draw_rate_and_davidson_model_probabilities_hand_values():
    p = O.bt_draw_rate_probabilities(0.0, np.array([0.0, 0.0, 0.0]), 0.25)
    assert p.shape == (6, 3) and p[0] == pytest.approx([0.375, 0.25, 0.375])
    p = O.davidson_model_probabilities(0.0, 0.0, np.zeros(3))
    assert p[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    p = O.poisson_model_probabilities(0.0, 0.0, np.zeros(3))            # μ = 0: λ = 1 auf beiden Seiten
    assert p[0] == pytest.approx([0.34575, 0.30851, 0.34575], abs=2e-5)
