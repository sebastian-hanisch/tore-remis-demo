"""Orakel-Tests tore-remis-demo: Schätzer, Torverteilungen und Vorhersagen gegen ANDERE Rechenwege (DEMO-PLAYBOOK §12).

1. Poisson-Newton gegen scipy.optimize.minimize auf der negativen Log-Likelihood, aufgebaut aus der PARTIENLISTE (jede Partie einzeln).
2. Davidson-Newton gegen scipy.optimize auf der Partienliste.
3. Bradley-Terry mit Remis = ½ gegen scipy auf der Partienliste.
4. Inversion gegen scipy.stats.poisson.ppf und nbinom.ppf (Gleichheit); 1-X-2 gegen die Brute-Force-Doppelsumme mit scipy.stats-Wahrscheinlichkeiten.
5. Im vollständigen, ausgeglichenen Plan ohne Heimvorteil: Poisson-Ordnung = Ordnung der Tordifferenz; Davidson- und Bradley-Terry-Ordnung = Ordnung der Punkte.
6. Heimvorteil: ĥ = ln(Heimtore/Gasttore) in ausgeglichenen Plänen (geschlossene Form).
"""

import random

import numpy as np
import pytest

import trm_goals as G
import trm_models as M
import trm_outcomes as O
import turnier_generator as T

optimize = pytest.importorskip("scipy.optimize")
stats = pytest.importorskip("scipy.stats")


def random_games(n, rnd, k_max=3, g=1.4):
    """Zufällige Partienliste (Heim, Gast, Tore): jedes Paar mit Wahrscheinlichkeit 0,8 bis zu k_max-mal, Tore Poisson aus zufälligen Stärken."""
    s = np.array([rnd.uniform(-1, 1) for _ in range(n)])
    H, A, gh, ga = [], [], [], []
    gen = np.random.default_rng(rnd.randint(0, 10 ** 6))
    for i in range(n):
        for j in range(i + 1, n):
            if rnd.random() < 0.8:
                for _ in range(rnd.randint(1, k_max)):
                    h, a = (i, j) if rnd.random() < 0.5 else (j, i)
                    lh, la = G.rate_log(s[h], s[a], g, 0.2)
                    H.append(h)
                    A.append(a)
                    gh.append(int(gen.poisson(np.exp(lh))))
                    ga.append(int(gen.poisson(np.exp(la))))
    return np.array(H), np.array(A), np.array(gh), np.array(ga)


def neg_poisson(b, n, H, A, gh, ga, lam, with_h):
    mu, h, th = b[0], (b[1] if with_h else 0.0), b[(2 if with_h else 1):]
    total = 0.0
    for hi, ai, x, y in zip(H, A, gh, ga):                      # jede Partie einzeln
        d = (th[hi] - th[ai]) / 2
        total += np.exp(mu + h + d) - x * (mu + h + d) + np.exp(mu - d) - y * (mu - d)
    return total + 0.5 * lam * float(th @ th) + 0.5 * M.C.MU_RIDGE * (mu ** 2 + (h ** 2 if with_h else 0.0))


# --- 1. Poisson ---------------------------------------------------------------------------------------------------------------------------

def test_poisson_newton_matches_scipy_on_the_game_list():
    rnd = random.Random(2026)
    checked = 0
    for _ in range(25):
        n = rnd.randint(3, 5)
        H, A, gh, ga = random_games(n, rnd)
        if len(H) < n:
            continue
        for with_h in (True, False):
            lam = 0.6
            ours = M.fit_poisson(n, H, A, gh, ga, lam, with_h)
            x0 = np.zeros(n + 1 + int(with_h))
            res = optimize.minimize(neg_poisson, x0, args=(n, H, A, gh, ga, lam, with_h), method="BFGS", options={"gtol": 1e-9})
            th = res.x[(2 if with_h else 1):]
            assert ours.theta == pytest.approx(th - th.mean(), abs=1e-4)
            assert ours.mu == pytest.approx(res.x[0], abs=1e-4)
            if with_h:
                assert ours.h == pytest.approx(res.x[1], abs=1e-4)
        checked += 1
    assert checked > 15


# --- 2. Davidson und 3. Bradley-Terry ------------------------------------------------------------------------------------------------------

def neg_davidson(b, n, H, A, out, lam, with_h):
    lnu, h, th = b[0], (b[1] if with_h else 0.0), b[(2 if with_h else 1):]
    total = 0.0
    for hi, ai, o in zip(H, A, out):
        a, c = th[hi] + h, th[ai]
        mid = lnu + (a + c) / 2
        den = np.logaddexp(np.logaddexp(a, c), mid)
        total -= (a, mid, c)[o] - den
    return total + 0.5 * lam * float(th @ th) + 0.5 * M.C.MU_RIDGE * (lnu ** 2 + (h ** 2 if with_h else 0.0))


def test_davidson_newton_matches_scipy_on_the_game_list():
    rnd = random.Random(7)
    checked = 0
    for _ in range(20):
        n = rnd.randint(3, 5)
        H, A, gh, ga = random_games(n, rnd, k_max=4, g=1.0)
        if len(H) < n:
            continue
        out = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
        for with_h in (True, False):
            lam = 0.7
            ours = M.fit_davidson(n, H, A, gh, ga, lam, with_h)
            x0 = np.zeros(n + 1 + int(with_h))
            res = optimize.minimize(neg_davidson, x0, args=(n, H, A, out, lam, with_h), method="BFGS", options={"gtol": 1e-9})
            th = res.x[(2 if with_h else 1):]
            assert ours.theta == pytest.approx(th - th.mean(), abs=1e-4)
            assert ours.log_nu == pytest.approx(res.x[0], abs=1e-4)
        checked += 1
    assert checked > 12


def neg_bt(b, n, H, A, w, lam):
    th = b
    x = np.array([th[h] - th[a] for h, a in zip(H, A)])
    return float(-(w * -np.logaddexp(0.0, -x) + (1 - w) * -np.logaddexp(0.0, x)).sum() + 0.5 * lam * float(th @ th))


def test_bradley_terry_half_matches_scipy_on_the_game_list():
    rnd = random.Random(11)
    for _ in range(15):
        n = rnd.randint(3, 5)
        H, A, gh, ga = random_games(n, rnd, k_max=4, g=1.0)
        if len(H) < n:
            continue
        w = (gh > ga) + 0.5 * (gh == ga)
        ours = M.fit_bradley_terry(n, H, A, gh, ga, 0.9, "half", False)
        res = optimize.minimize(neg_bt, np.zeros(n), args=(n, H, A, w, 0.9), method="BFGS", options={"gtol": 1e-10})
        assert ours.theta == pytest.approx(res.x - res.x.mean(), abs=1e-5)


# --- 4. Verteilungen --------------------------------------------------------------------------------------------------------------------------

def test_inversion_equals_scipy_ppf_for_poisson_and_negative_binomial():
    gen = np.random.default_rng(4)
    u = gen.uniform(size=3000)
    lam = gen.uniform(0.2, 25, size=3000)
    assert np.array_equal(G.inverse_counts(u, lam), stats.poisson.ppf(u, lam).astype(int))
    for r in (0.7, 2.0, 8.0):
        assert np.array_equal(G.inverse_counts(u, lam, r), stats.nbinom.ppf(u, r, r / (r + lam)).astype(int))


def test_outcome_probabilities_match_the_double_sum_with_scipy_pmfs():
    for lh, la, r in ((1.3, 0.9, None), (0.6, 2.4, None), (2.0, 1.5, 2.0), (3.5, 0.8, 0.7)):
        ks = np.arange(80)
        if r is None:
            ph, pa = stats.poisson.pmf(ks, lh), stats.poisson.pmf(ks, la)
        else:
            ph, pa = stats.nbinom.pmf(ks, r, r / (r + lh)), stats.nbinom.pmf(ks, r, r / (r + la))
        joint = np.outer(ph, pa)
        want = np.array([sum(joint[i, j] for i in range(80) for j in range(80) if i > j), np.trace(joint),
                         sum(joint[i, j] for i in range(80) for j in range(80) if i < j)])
        want = want / want.sum()
        assert O.outcome_probabilities([lh], [la], r)[0] == pytest.approx(want, abs=1e-6)


# --- 5. Ordnungen im vollständigen, ausgeglichenen Plan -------------------------------------------------------------------------------------

def test_orders_are_the_goal_difference_and_points_orders_in_a_balanced_plan():
    seen_gd, seen_pts = 0, 0
    for seed in range(100000, 100120):
        gm = G.simulate_games(8, 0.3, 2, 1.4, 0.0, "alternierend", None, seed)
        n = 8
        tab = M.goal_difference_table(n, gm.home, gm.away, gm.goals_home, gm.goals_away)
        lam = 0.4
        if len(set(tab["diff"])) == n:
            po = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, with_h=False)
            assert list(np.argsort(-po.theta)) == list(np.argsort(-tab["diff"]))
            seen_gd += 1
        pts = 2 * tab["wins"] + tab["draws"]
        if len(set(pts)) == n:
            dv = M.fit_davidson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, with_h=False)
            bt = M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, "half", False)
            assert list(np.argsort(-dv.theta)) == list(np.argsort(-pts)) == list(np.argsort(-bt.theta))
            seen_pts += 1
    assert seen_gd > 40 and seen_pts > 40


# --- 6. Heimvorteil ---------------------------------------------------------------------------------------------------------------------

def test_home_advantage_matches_the_log_ratio_of_home_to_away_goals_in_balanced_plans():
    for seed in range(12):
        for k in (2, 4):
            gm = G.simulate_games(6, 0.3, k, 1.4, 0.3, "alternierend", None, seed)
            fit = M.fit_poisson(6, gm.home, gm.away, gm.goals_home, gm.goals_away, 1e-9, with_h=True)
            assert fit.h == pytest.approx(np.log(gm.goals_home.sum() / gm.goals_away.sum()), abs=2e-3)
