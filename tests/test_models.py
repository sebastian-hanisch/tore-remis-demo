"""Schätzer gegen die von Hand gerechneten Mini-Instanzen und die Likelihood-Gleichungen."""

import math
import random

import numpy as np
import pytest

import trm_constants as C
import trm_goals as G
import trm_models as M
import turnier_generator as T


def test_poisson_mini_instance_ratio_of_goals():
    # A (daheim) schlägt B 2:0, B (daheim) und A 1:1; h fest 0, λ → 0: Tore A 3, Tore B 1 über zwei Partien
    fit = M.fit_poisson(2, [0, 1], [1, 0], [2, 1], [0, 1], 1e-9, with_h=False)
    assert fit.converged
    assert fit.theta[0] - fit.theta[1] == pytest.approx(math.log(3), abs=1e-4)
    d = fit.theta[0] - fit.theta[1]
    assert math.exp(fit.mu + d / 2) == pytest.approx(1.5, abs=1e-3) and math.exp(fit.mu - d / 2) == pytest.approx(0.5, abs=1e-3)
    assert fit.h == 0.0


def test_poisson_home_advantage_closed_form_in_a_balanced_plan():
    gm = G.simulate_games(6, 0.3, 4, 1.4, 0.3, "alternierend", None, 5)
    fit = M.fit_poisson(6, gm.home, gm.away, gm.goals_home, gm.goals_away, 1e-9, with_h=True)
    assert fit.h == pytest.approx(math.log(gm.goals_home.sum() / gm.goals_away.sum()), abs=1e-3)


def _poisson_score(n, H, A, gh, ga, fit, lam):
    """Likelihood-Gleichungen aus der Partienliste (unabhängig vom Newton-Code): ∂/∂θ_i, ∂/∂μ, ∂/∂h."""
    lh = np.exp(fit.mu + fit.h + (fit.theta[H] - fit.theta[A]) / 2)
    la = np.exp(fit.mu - (fit.theta[H] - fit.theta[A]) / 2)
    eh, ea = gh - lh, ga - la
    g_theta = np.zeros(n)
    np.add.at(g_theta, H, 0.5 * (eh - ea))
    np.add.at(g_theta, A, -0.5 * (eh - ea))
    return g_theta - lam * fit.theta, eh.sum() + ea.sum(), eh.sum()


def test_poisson_solution_satisfies_the_score_equations():
    rnd = random.Random(3)
    for seed in range(8):
        gm = G.simulate_games(rnd.randint(4, 8), 0.3, rnd.randint(1, 4), 1.4, 0.3, "zufaellig", None, seed)
        n, lam = gm.n, 0.7
        fit = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, with_h=True)
        g_theta, g_mu, g_h = _poisson_score(n, gm.home, gm.away, gm.goals_home, gm.goals_away, fit, lam)
        assert np.abs(g_theta).max() < 1e-6
        assert abs(g_mu - C.MU_RIDGE * fit.mu) < 1e-6 and abs(g_h - C.MU_RIDGE * fit.h) < 1e-6


def test_newton_steps_are_few_and_the_solution_is_centered_even_for_a_vanishing_prior():
    for seed in range(15):
        gm = G.simulate_games(8, 0.3, 4, 1.4, 0.3, "alternierend", None, seed)
        fit = M.fit_poisson(8, gm.home, gm.away, gm.goals_home, gm.goals_away, 1e-8)
        assert fit.converged and fit.iterations <= 12
        assert fit.theta.sum() == pytest.approx(0.0, abs=1e-9)


def test_davidson_mini_instance_gives_nu_one():
    # A schlägt B, Remis, B schlägt A: jedes Ergebnis einmal, Stärken gleich, P(Remis) = 1/3 = ν/(2 + ν) → ν = 1
    fit = M.fit_davidson(2, [0, 0, 1], [1, 1, 0], np.array([1, 0, 1]), np.array([0, 0, 0]), 1e-9, with_h=False)
    assert fit.converged
    assert math.exp(fit.log_nu) == pytest.approx(1.0, abs=1e-6)
    assert fit.theta == pytest.approx([0.0, 0.0], abs=1e-6)


def test_davidson_probabilities_hand_values():
    p = M.davidson_probabilities(0.0, 0.0, np.array([0.0]), np.array([0.0]))
    assert p[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    p = M.davidson_probabilities(math.log(2.0), 0.0, np.array([0.0]), np.array([0.0]))     # ν = 2: Remis 2/4
    assert p[0] == pytest.approx([0.25, 0.5, 0.25])
    p = M.davidson_probabilities(0.0, math.log(3.0), np.array([0.0]), np.array([0.0]))     # e^h = 3: π_H = 3, π_G = 1
    assert p[0].sum() == pytest.approx(1.0) and p[0][0] / p[0][2] == pytest.approx(3.0)


def test_davidson_solution_satisfies_the_score_equations():
    gm = G.simulate_games(5, 0.3, 3, 1.4, 0.3, "zufaellig", None, 2)
    lam = 0.8
    fit = M.fit_davidson(5, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, with_h=True)
    p = M.davidson_probabilities(fit.log_nu, fit.h, fit.theta[gm.home], fit.theta[gm.away])
    out = np.where(gm.goals_home > gm.goals_away, 0, np.where(gm.goals_home == gm.goals_away, 1, 2))
    da = (out == 0) + 0.5 * (out == 1) - p[:, 0] - 0.5 * p[:, 1]
    db = (out == 2) + 0.5 * (out == 1) - p[:, 2] - 0.5 * p[:, 1]
    g = np.zeros(5)
    np.add.at(g, gm.home, da)
    np.add.at(g, gm.away, db)
    assert np.abs(g - lam * fit.theta).max() < 1e-6
    assert abs(((out == 1) - p[:, 1]).sum() - C.MU_RIDGE * fit.log_nu) < 1e-6


def test_bradley_terry_half_draws_and_drop():
    H, A = [0, 0, 1, 1], [1, 1, 0, 0]
    gh, ga = np.array([1, 2, 0, 0]), np.array([0, 2, 1, 0])        # A siegt daheim, Remis, A siegt auswärts, Remis
    half = M.fit_bradley_terry(2, H, A, gh, ga, 1e-9, "half")
    drop = M.fit_bradley_terry(2, H, A, gh, ga, 1e-9, "drop")
    assert half.theta[0] > half.theta[1]
    # Remis = ½: A hat 3 von 4 Punkten, also θ_A − θ_B = ln 3; verworfen: 2 von 2, die Stärke läuft weg (hier durch die Vorannahme begrenzt)
    assert half.theta[0] - half.theta[1] == pytest.approx(math.log(3), abs=1e-4)
    assert drop.theta[0] - drop.theta[1] > half.theta[0] - half.theta[1]
    with pytest.raises(ValueError, match="Remis-Behandlung"):
        M.fit_bradley_terry(2, H, A, gh, ga, 1.0, "mittel")
    zeros = M.fit_bradley_terry(2, [0], [1], np.array([1]), np.array([1]), 1.0, "drop")    # nur ein Remis, verworfen: keine Daten
    assert zeros.theta == pytest.approx([0.0, 0.0])


def test_goal_difference_table_and_massey_hand_values():
    H, A, gh, ga = [0, 1], [1, 0], np.array([2, 1]), np.array([0, 1])
    tab = M.goal_difference_table(2, H, A, gh, ga)
    assert tab["wins"].tolist() == [1, 0] and tab["draws"].tolist() == [1, 1] and tab["losses"].tolist() == [0, 1]
    assert tab["gf"].tolist() == [3, 1] and tab["ga"].tolist() == [1, 3] and tab["diff"].tolist() == [2, -2] and tab["games"].tolist() == [2, 2]
    r = M.massey_goal_difference(2, H, A, gh, ga)
    assert r[0] - r[1] == pytest.approx(1.0) and r.sum() == pytest.approx(0.0)       # mittlere Tordifferenz je Partie 1


def test_poisson_order_is_the_goal_difference_order_in_a_balanced_plan():
    seen = 0
    for seed in range(60):
        gm = G.simulate_games(8, 0.3, 2, 1.4, 0.0, "alternierend", None, seed)
        tab = M.goal_difference_table(8, gm.home, gm.away, gm.goals_home, gm.goals_away)
        if len(set(tab["diff"])) != 8:
            continue
        fit = M.fit_poisson(8, gm.home, gm.away, gm.goals_home, gm.goals_away, 0.5, with_h=False)
        assert list(np.argsort(-fit.theta)) == list(np.argsort(-tab["diff"]))
        seen += 1
    assert seen > 25


def test_rank_with_tolerance_and_kendall_tau_hand_instances():
    for s in range(30):
        assert M.rank_with_tolerance([1.0, 1.0 + 1e-13, 0.5], T.SplitMix64(s))[2] == 2
    assert {M.rank_with_tolerance([1.0, 1.0 + 1e-13, 0.5], T.SplitMix64(s))[0] for s in range(40)} == {0, 1}
    assert {M.rank_with_tolerance([1.0, 1.0 + 1e-6, 0.5], T.SplitMix64(s))[1] for s in range(40)} == {0}
    assert M.kendall_tau((0, 2, 1), (3, 1, 2)) == 1.0 and M.kendall_tau((0, 1, 2), (1, 2, 3)) == -1.0
    assert math.isnan(M.kendall_tau((0, 1), (5.0, 5.0)))


def test_oracle_lambda_formula():
    assert M.oracle_lambda(8, 0.3) == pytest.approx(12 / (0.09 * 63))
