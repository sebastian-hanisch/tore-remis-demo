"""Tormodell: Inversion der Verteilungsfunktion, Torraten und Spielplan mit Heimrecht gegen von Hand gerechnete Werte."""

import math

import numpy as np
import pytest

import trm_goals as G
import turnier_generator as T


def test_inverse_counts_hand_values_poisson():
    # λ = 1: F(0) = 0,3679, F(1) = 0,7358, F(2) = 0,9197, F(3) = 0,9810, F(4) = 0,9963
    assert G.inverse_counts([0.30, 0.5, 0.80, 0.95, 0.99], [1.0] * 5).tolist() == [0, 1, 2, 3, 4]


def test_inverse_counts_is_monotone_in_u_and_in_the_rate():
    u = np.linspace(0.01, 0.99, 50)
    for r in (None, 2.0):
        k = G.inverse_counts(u, np.full(50, 3.0), r)
        assert (np.diff(k) >= 0).all()
        k_low, k_high = G.inverse_counts(u, np.full(50, 1.0), r), G.inverse_counts(u, np.full(50, 5.0), r)
        assert (k_high >= k_low).all()


def test_inverse_counts_reproduce_mean_and_variance():
    u = T.SplitMix64(5).uniform(200000)
    lam = np.full(200000, 2.5)
    pois = G.inverse_counts(u, lam)
    assert pois.mean() == pytest.approx(2.5, rel=0.01) and pois.var() == pytest.approx(2.5, rel=0.03)
    nb = G.inverse_counts(u, lam, 2.0)
    assert nb.mean() == pytest.approx(2.5, rel=0.015) and nb.var() == pytest.approx(2.5 + 2.5 ** 2 / 2.0, rel=0.04)


def test_rate_log_hand_values():
    lh, la = G.rate_log(0.0, 0.0, 1.4, 0.3)
    assert math.exp(lh) == pytest.approx(1.4 * math.exp(0.15)) and math.exp(la) == pytest.approx(1.4 * math.exp(-0.15))
    lh, la = G.rate_log(1.0, 0.0, 1.0, 0.0)             # Torverhältnis e^1
    assert math.exp(lh - la) == pytest.approx(math.e)


def test_schedule_every_pair_k_times_and_home_balance_for_alternating():
    n = 6
    for k, diff in ((4, 0), (3, 1)):
        gm = G.simulate_games(n, 0.3, k, 1.4, 0.3, "alternierend", None, 7)
        assert len(gm.home) == n * (n - 1) // 2 * k
        for i in range(n):
            for j in range(i + 1, n):
                mask = ((gm.home == i) & (gm.away == j)) | ((gm.home == j) & (gm.away == i))
                assert mask.sum() == k
                assert abs(((gm.home == i) & mask).sum() - ((gm.home == j) & mask).sum()) == diff      # gerades k: gleich oft daheim, ungerades: Unterschied 1


def test_home_modes_follow_their_description():
    n = 8
    for mode, expected in (("setzliste", 0.8), ("umgekehrt", 0.2), ("zufaellig", 0.5)):
        shares = []
        for seed in range(20):
            gm = G.simulate_games(n, 0.3, 10, 1.4, 0.3, mode, None, seed)
            shares.append(float((gm.home > gm.away).mean()))        # Index = Stärkerang: ein höherer Index ist stärker
        assert np.mean(shares) == pytest.approx(expected, abs=0.03), mode


def test_games_are_deterministic_and_seed_sensitive_and_goals_are_nonnegative():
    a = G.simulate_games(8, 0.3, 2, 1.4, 0.3, "zufaellig", 2.0, 11)
    b = G.simulate_games(8, 0.3, 2, 1.4, 0.3, "zufaellig", 2.0, 11)
    c = G.simulate_games(8, 0.3, 2, 1.4, 0.3, "zufaellig", 2.0, 12)
    assert np.array_equal(a.goals_home, b.goals_home) and np.array_equal(a.home, b.home)
    assert not np.array_equal(a.goals_home, c.goals_home)
    assert (a.goals_home >= 0).all() and (a.goals_away >= 0).all()
    assert a.strengths.mean() == pytest.approx(0.0, abs=1e-12) and len(set(a.names)) == 8


def test_unknown_home_mode_raises():
    with pytest.raises(ValueError, match="Heimrecht"):
        G.simulate_games(4, 0.3, 1, 1.4, 0.3, "nie", None, 1)


def test_mean_goals_match_the_rates():
    gm = G.simulate_games(8, 0.3, 40, 1.4, 0.3, "alternierend", None, 3)
    s = gm.strengths
    lh, la = G.rate_log(s[gm.home], s[gm.away], 1.4, 0.3)
    assert gm.goals_home.mean() == pytest.approx(np.exp(lh).mean(), rel=0.03)
    assert gm.goals_away.mean() == pytest.approx(np.exp(la).mean(), rel=0.03)
