"""Eigenschaftstests tore-remis-demo (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für Poisson-Tore, Davidson und die 1-X-2-Wahrscheinlichkeiten gelten
müssen, ohne dass das richtige Ergebnis bekannt ist. Instanzen: 24 simulierte Spielpläne (`trm_goals.simulate_games`, feste Seeds, n = 3 bis 6, 1 bis 4
Partien je Paar, alle vier Heimrecht-Verteilungen, mit und ohne Überstreuung) plus Randfälle (n = 2, ein Spiel, gleiche Stärken, nie ein Tor).

1. Umnummerieren der Mannschaften permutiert die Stärken (Poisson, Davidson, Bradley-Terry mit Remis = 1/2, Massey auf der Tordifferenz).
2. Heim und Auswärts vertauschen (jede Partie mit umgekehrten Rollen, die Tore bleiben bei ihren Mannschaften): Stärken gleich, Heimvorteil h -> -h
   (Poisson: zusätzlich μ -> μ + h; Davidson: ν gleich; Bradley-Terry: h -> -h). Aus der Definition der Modelle (Umparametrisieren), bis auf den winzigen Zuschlag auf μ, h, ln ν.
   Ohne Heimvorteil im Modell sind die Stärken beim Vertauschen exakt gleich; Massey und Tabelle ändern sich nicht.
3. Poisson-Verteilung: Zählwahrscheinlichkeiten summieren sich auf 1 (Poisson und negativ binomial); 1-X-2 spiegelt sich beim Vertauschen der Raten, hat bei gleichen
   Raten gleiche Heim- und Gastwahrscheinlichkeit, und eine höhere Heimrate hebt P(Heim) und senkt P(Gast).
4. Davidson: Summe 1, bei gleicher Stärke und ohne Heimvorteil P(Heim) = P(Gast), P(Remis) = ν sqrt(P(Heim) P(Gast)) (Definition), Rollentausch spiegelt,
   P(Remis) bei θ = (d, -d) gleich ν / (ν + 2 cosh d) und fällt mit |d|, ν -> 0 liefert Bradley-Terry.
5. Score-Gleichungen am Optimum: Poisson (Tordifferenz je Mannschaft = erwartete Tordifferenz + 2 λ θ, Gesamttore = erwartete Gesamttore, Heimtore = erwartete Heimtore,
   jeweils bis auf den Zuschlag 1e-4) und Davidson (beobachtete Remis = erwartete Remis).
6. Tabellen: Summe der Siege = Summe der Niederlagen, Summe der Tordifferenzen 0, Massey summiert sich zu 0.
7. Inversion der Verteilungsfunktion ist monoton in der Gleichverteilten und in der Rate (Poisson und negativ binomial); alternierendes Heimrecht ist bei geraden k
   ausgeglichen; gleicher Seed gibt gleiche Spiele.
"""

import math

import numpy as np
import pytest

import trm_constants as C
import trm_goals as G
import trm_models as M
import trm_outcomes as O
import turnier_generator as T

N_INSTANCES = 24
MODES = ("alternierend", "zufaellig", "setzliste", "umgekehrt")


def instance(seed):
    n = 3 + seed % 4
    k = 1 + seed % 4
    r = 2.0 if seed % 6 == 5 else None
    return G.simulate_games(n, 0.4, k, 1.6, 0.3, MODES[seed % 4], r, 500 + seed)


def instances():
    return [instance(s) for s in range(N_INSTANCES)]


def lam_of(g):
    return M.oracle_lambda(g.n, 0.4)


def relabel(g, perm):
    """Neu nummerieren: neue Mannschaft a ist die alte perm[a]."""
    inv = np.argsort(perm)
    return inv[g.home], inv[g.away], g.goals_home, g.goals_away


def swapped(g):
    """Jede Partie mit umgekehrten Rollen: neuer Heimklub = alter Gast, die Tore bleiben bei ihrer Mannschaft."""
    return g.away, g.home, g.goals_away, g.goals_home


def fits(n, H, A, gh, ga, lam, with_h):
    return (M.fit_poisson(n, H, A, gh, ga, lam, with_h), M.fit_davidson(n, H, A, gh, ga, lam, with_h),
            M.fit_bradley_terry(n, H, A, gh, ga, lam, "half", with_h))


# --- 1. Umnummerieren -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_h", [True, False])
def test_relabelling_the_teams_permutes_all_strength_estimates(with_h):
    rng = T.SplitMix64(31)
    for g in instances():
        lam = lam_of(g)
        perm = rng.permutation(g.n)
        H2, A2, gh2, ga2 = relabel(g, perm)
        base = fits(g.n, g.home, g.away, g.goals_home, g.goals_away, lam, with_h)
        new = fits(g.n, H2, A2, gh2, ga2, lam, with_h)
        for a, b in zip(base, new):
            assert b.converged and a.converged
            assert np.allclose(b.theta, a.theta[perm], atol=1e-6)
            assert b.h == pytest.approx(a.h, abs=1e-6)
        massey = M.massey_goal_difference(g.n, g.home, g.away, g.goals_home, g.goals_away)
        massey2 = M.massey_goal_difference(g.n, H2, A2, gh2, ga2)
        assert np.allclose(massey2, massey[perm], atol=1e-9)


# --- 2. Heim und Auswärts vertauschen ----------------------------------------------------------------------------------------------------

def test_swapping_home_and_away_keeps_the_strengths_and_negates_the_home_advantage():
    for g in instances():
        lam = lam_of(g)
        H2, A2, gh2, ga2 = swapped(g)
        p, d, b = fits(g.n, g.home, g.away, g.goals_home, g.goals_away, lam, True)
        p2, d2, b2 = fits(g.n, H2, A2, gh2, ga2, lam, True)
        assert np.allclose(p2.theta, p.theta, atol=2e-4) and p2.h == pytest.approx(-p.h, abs=2e-4) and p2.mu == pytest.approx(p.mu + p.h, abs=2e-4)
        assert np.allclose(d2.theta, d.theta, atol=2e-4) and d2.h == pytest.approx(-d.h, abs=2e-4) and d2.log_nu == pytest.approx(d.log_nu, abs=2e-4)
        assert np.allclose(b2.theta, b.theta, atol=2e-4) and b2.h == pytest.approx(-b.h, abs=2e-4)


def test_without_a_home_advantage_in_the_model_swapping_changes_nothing():
    for g in instances():
        lam = lam_of(g)
        H2, A2, gh2, ga2 = swapped(g)
        for a, b in zip(fits(g.n, g.home, g.away, g.goals_home, g.goals_away, lam, False), fits(g.n, H2, A2, gh2, ga2, lam, False)):
            assert np.allclose(a.theta, b.theta, atol=1e-6)
        m1 = M.massey_goal_difference(g.n, g.home, g.away, g.goals_home, g.goals_away)
        m2 = M.massey_goal_difference(g.n, H2, A2, gh2, ga2)
        assert np.allclose(m1, m2, atol=1e-9)
        t1 = M.goal_difference_table(g.n, g.home, g.away, g.goals_home, g.goals_away)
        t2 = M.goal_difference_table(g.n, H2, A2, gh2, ga2)
        assert all(np.array_equal(t1[key], t2[key]) for key in t1)


# --- 3. Poisson-Verteilung und 1-X-2 ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", [None, 0.7, 2.0, 8.0])
def test_count_probabilities_add_up_to_one(r):
    rng = T.SplitMix64(41)
    lam = 0.1 + rng.uniform(N_INSTANCES) * 12
    pmf = O.count_pmf(lam, r)
    assert (pmf >= 0).all()
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-7)


@pytest.mark.parametrize("r", [None, 2.0])
def test_one_x_two_mirrors_when_the_rates_are_swapped_and_is_symmetric_for_equal_rates(r):
    rng = T.SplitMix64(43)
    for _ in range(N_INSTANCES):
        a, b = 0.2 + rng.uniform(2) * 5
        p = O.outcome_probabilities([a], [b], r)[0]
        q = O.outcome_probabilities([b], [a], r)[0]
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert q.tolist() == pytest.approx([p[2], p[1], p[0]], abs=1e-9)
        e = O.outcome_probabilities([a], [a], r)[0]
        assert e[0] == pytest.approx(e[2], abs=1e-9)


def test_a_higher_home_rate_raises_the_home_win_and_lowers_the_away_win():
    rng = T.SplitMix64(47)
    for k in range(N_INSTANCES):
        r = None if k % 2 else 3.0
        a, b = 0.3 + rng.uniform(2) * 4
        lo = O.outcome_probabilities([a], [b], r)[0]
        hi = O.outcome_probabilities([a * 1.3], [b], r)[0]
        assert hi[0] > lo[0] and hi[2] < lo[2]


def test_true_probabilities_swap_with_the_roles_and_the_advantage_helps_the_home_side():
    rng = T.SplitMix64(53)
    for k in range(10):
        s = (rng.uniform(4) - 0.5) * 2
        I, J = O.ordered_pairs(4)
        for h in (0.0, 0.3):
            P = O.true_probabilities(s, 1.4, h)
            assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
            row = {(int(i), int(j)): P[x] for x, (i, j) in enumerate(zip(I, J))}
            if h == 0.0:
                assert row[(0, 1)].tolist() == pytest.approx([row[(1, 0)][2], row[(1, 0)][1], row[(1, 0)][0]], abs=1e-9)
        equal = O.true_probabilities(np.zeros(3), 1.4, 0.3)
        assert (equal[:, 0] > equal[:, 2]).all()                        # gleiche Stärke: Heimvorteil hilft dem Heimteam
        no_adv = O.true_probabilities(np.zeros(3), 1.4, 0.0)
        assert np.allclose(no_adv[:, 0], no_adv[:, 2], atol=1e-9)


def test_the_poisson_model_with_the_true_parameters_equals_the_true_probabilities():
    rng = T.SplitMix64(59)
    for k in range(10):
        s = (rng.uniform(4) - 0.5) * 1.5
        g, h = 1.1 + 0.3 * k, 0.1 * (k % 4)
        assert np.allclose(O.poisson_model_probabilities(math.log(g) - h / 2, h, s), O.true_probabilities(s, g, h), atol=1e-12)


def test_the_draw_rate_model_keeps_its_constant_draw_share_and_sums_to_one():
    rng = T.SplitMix64(61)
    for k in range(10):
        theta = (rng.uniform(4) - 0.5) * 2
        P = O.bt_draw_rate_probabilities(0.2, theta, 0.27)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12) and np.allclose(P[:, 1], 0.27)


# --- 4. Davidson ----------------------------------------------------------------------------------------------------------------------------

def test_davidson_probabilities_sum_to_one_and_follow_the_definition():
    rng = T.SplitMix64(67)
    for k in range(N_INSTANCES):
        lnu, h, a, b = (rng.uniform(4) - 0.5) * 3
        p = M.davidson_probabilities(lnu, h, a, b)
        assert (p > 0).all() and p.sum() == pytest.approx(1.0, abs=1e-12)
        assert p[1] == pytest.approx(math.exp(lnu) * math.sqrt(p[0] * p[2]), abs=1e-12)
        mirror = M.davidson_probabilities(lnu, 0.0, b, a)
        straight = M.davidson_probabilities(lnu, 0.0, a, b)
        assert mirror.tolist() == pytest.approx([straight[2], straight[1], straight[0]], abs=1e-12)     # ohne Heimvorteil spiegelt der Rollentausch
        assert M.davidson_probabilities(lnu, h, a, a + h)[0] == pytest.approx(M.davidson_probabilities(lnu, h, a, a + h)[2], abs=1e-12)   # π_Heim = π_Gast


def test_the_davidson_draw_share_is_largest_at_equal_strength_and_has_the_closed_form():
    for lnu in (-1.0, 0.0, 0.8):
        nu = math.exp(lnu)
        previous = 2.0
        for d in (0.0, 0.2, 0.5, 1.0, 2.0, 4.0):
            p = M.davidson_probabilities(lnu, 0.0, d, -d)
            assert p[1] == pytest.approx(nu / (nu + 2.0 * math.cosh(d)), abs=1e-12)
            assert p[1] < previous
            previous = p[1]


def test_without_ties_davidson_is_bradley_terry():
    rng = T.SplitMix64(71)
    for k in range(N_INSTANCES):
        h, a, b = (rng.uniform(3) - 0.5) * 3
        p = M.davidson_probabilities(-60.0, h, a, b)
        assert p[1] < 1e-12
        assert p[0] / (p[0] + p[2]) == pytest.approx(1.0 / (1.0 + math.exp(-(a + h - b))), abs=1e-9)


# --- 5. Score-Gleichungen ---------------------------------------------------------------------------------------------------------------------

def test_poisson_score_equations_hold_at_the_optimum():
    for g in instances():
        lam = lam_of(g)
        fit = M.fit_poisson(g.n, g.home, g.away, g.goals_home, g.goals_away, lam, True)
        assert fit.converged
        eh = np.exp(fit.mu + fit.h + (fit.theta[g.home] - fit.theta[g.away]) / 2)
        ea = np.exp(fit.mu - (fit.theta[g.home] - fit.theta[g.away]) / 2)
        assert float(g.goals_home.sum() + g.goals_away.sum() - eh.sum() - ea.sum()) == pytest.approx(C.MU_RIDGE * fit.mu, abs=1e-5)
        assert float(g.goals_home.sum() - eh.sum()) == pytest.approx(C.MU_RIDGE * fit.h, abs=1e-5)
        resid = np.zeros(g.n)
        np.add.at(resid, g.home, 0.5 * ((g.goals_home - eh) - (g.goals_away - ea)))
        np.add.at(resid, g.away, 0.5 * ((g.goals_away - ea) - (g.goals_home - eh)))
        assert np.allclose(resid, lam * fit.theta, atol=1e-5)


def test_davidson_expected_draws_equal_observed_draws_at_the_optimum():
    for g in instances():
        lam = lam_of(g)
        fit = M.fit_davidson(g.n, g.home, g.away, g.goals_home, g.goals_away, lam, True)
        assert fit.converged
        p = M.davidson_probabilities(fit.log_nu, fit.h, fit.theta[g.home], fit.theta[g.away])
        draws = float((g.goals_home == g.goals_away).sum())
        assert draws - float(p[:, 1].sum()) == pytest.approx(C.MU_RIDGE * fit.log_nu, abs=1e-5)


# --- 6. Tabelle und Massey -----------------------------------------------------------------------------------------------------------------------

def test_table_totals_balance_and_massey_sums_to_zero():
    for g in instances():
        t = M.goal_difference_table(g.n, g.home, g.away, g.goals_home, g.goals_away)
        assert t["wins"].sum() == t["losses"].sum()
        assert t["diff"].sum() == 0 and t["gf"].sum() == t["ga"].sum() == g.goals_home.sum() + g.goals_away.sum()
        assert (t["wins"] + t["draws"] + t["losses"] == t["games"]).all()
        assert t["games"].sum() == 2 * len(g.home)
        assert M.massey_goal_difference(g.n, g.home, g.away, g.goals_home, g.goals_away).sum() == pytest.approx(0.0, abs=1e-9)


def test_edge_cases_two_teams_no_goals_and_a_single_game():
    H, A = np.array([0]), np.array([1])
    zero = np.array([0])
    fit = M.fit_poisson(2, H, A, zero, zero, 1.0, True)                     # nie ein Tor: endlich dank der Zuschläge
    assert fit.converged and np.isfinite([fit.mu, fit.h]).all() and fit.theta.sum() == pytest.approx(0.0, abs=1e-9)
    fit0 = M.fit_poisson(2, H, A, zero, zero, 1.0, False)                   # ohne h in beide Richtungen symmetrisch: gleiche Stärken
    assert fit0.converged and np.allclose(fit0.theta, 0.0, atol=1e-9)
    d = M.fit_davidson(2, H, A, zero, zero, 1.0, False)
    assert np.allclose(d.theta, 0.0, atol=1e-9)
    win = M.fit_poisson(2, H, A, np.array([3]), zero, 1.0, False)
    assert win.theta[0] > 0 > win.theta[1] and win.theta.sum() == pytest.approx(0.0, abs=1e-9)
    b = M.fit_bradley_terry(2, H, A, np.array([1]), np.array([1]), 1.0, "half")      # Remis = 1/2: gleiche Stärke
    assert np.allclose(b.theta, 0.0, atol=1e-9)


# --- 7. Inversion, Heimrecht, Zufall ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", [None, 2.0])
def test_the_inversion_is_monotone_in_the_uniform_and_in_the_rate(r):
    rng = T.SplitMix64(73)
    u = np.sort(rng.uniform(200))
    lam = 0.2 + rng.uniform(8) * 6
    for rate in lam:
        k = G.inverse_counts(u, np.full(u.shape, rate), r)
        assert (np.diff(k) >= 0).all()
    for x in u[::10]:
        k = G.inverse_counts(np.full(len(lam), x), np.sort(lam), r)
        assert (np.diff(k) >= 0).all()


def test_alternating_home_rights_are_balanced_for_even_k_and_the_same_seed_gives_the_same_games():
    g = G.simulate_games(5, 0.4, 4, 1.4, 0.3, "alternierend", None, 9)
    home_count = np.bincount(g.home, minlength=5)
    away_count = np.bincount(g.away, minlength=5)
    assert (home_count == away_count).all()
    again = G.simulate_games(5, 0.4, 4, 1.4, 0.3, "alternierend", None, 9)
    assert np.array_equal(g.goals_home, again.goals_home) and np.array_equal(g.home, again.home)
    other = G.simulate_games(5, 0.4, 4, 1.4, 0.3, "alternierend", None, 10)
    assert not np.array_equal(g.goals_home, other.goals_home)
