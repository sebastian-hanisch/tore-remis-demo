"""Analyse, Szenario und Verdikte; Randfälle der Regler."""

import itertools
import math

import numpy as np
import pytest

import trm_constants as C
import trm_evaluation as ev
import trm_outcomes as O
from trm_scenario import Settings, make_games

CASES = [Settings(n=n, k=k, g=g, h=h, home=home, disp=disp, r=2.0, fit_h=fit_h, seed=seed)
         for n, k, g, h, home, disp, fit_h, seed in itertools.product((4, 8), (1, 3), (0.5, 3.0), (0.0, 0.3), ("alternierend", "setzliste"), (False, True),
                                                                       (False, True), (3,))]


def test_analysis_invariants_over_many_settings():
    for s in CASES:
        a = ev.analyse(s)
        n = s.n
        for m in C.METHODS:
            assert sorted(a.rank[m]) == list(range(n)), (s, m)
        assert a.poisson.theta.sum() == pytest.approx(0.0, abs=1e-8)
        assert a.table["games"].tolist() == [s.k * (n - 1)] * n
        assert (a.table["wins"] + a.table["draws"] + a.table["losses"] == a.table["games"]).all()
        assert a.table["diff"].sum() == 0
        for p in a.probs.values():
            assert p.shape == (n * (n - 1), 3) and p.sum(axis=1) == pytest.approx(np.ones(n * (n - 1)))
        assert all(v >= -1e-12 for v in a.kl.values())
        assert 0.0 <= a.draw_share <= 1.0 and a.oracle_lam > 0
        if not s.fit_h:
            assert a.poisson.h == 0.0 and a.davidson.h == 0.0 and a.bt.h == 0.0


def test_poisson_and_massey_order_agree_in_the_balanced_plan_without_home_advantage_estimation():
    seen = 0
    for seed in range(100000, 100100):
        a = ev.analyse(Settings(n=8, k=2, h=0.0, fit_h=False, seed=seed))
        if len(set(a.table["diff"])) == 8:
            assert a.rank["Poisson"] == a.rank["Massey"]
            seen += 1
    assert seen > 30


def test_analysis_is_cached_and_deterministic():
    s = Settings(n=6, seed=11)
    assert ev.analyse(s) is ev.analyse(s)
    assert np.array_equal(make_games(s).goals_home, make_games(Settings(n=6, seed=11)).goals_home)


def test_pair_probabilities_rows_sum_to_one_and_the_truth_matches_the_strengths():
    a = ev.analyse(Settings(n=6, g=1.4, h=0.3, seed=4))
    pp = ev.pair_probabilities(a, 5, 0)                  # Index 5 ist der Stärkste, 0 der Schwächste
    for m, p in pp.items():
        assert sum(p) == pytest.approx(1.0), m
    assert pp["Wahrheit"][0] > pp["Wahrheit"][2]
    truth = O.true_probabilities(a.games.strengths, 1.4, 0.3)
    I, J = O.ordered_pairs(6)
    assert pp["Wahrheit"] == pytest.approx(tuple(truth[int(np.nonzero((I == 5) & (J == 0))[0][0])]))


def test_overdispersion_is_the_only_switch_that_changes_the_truth_distribution():
    flat = ev.analyse(Settings(n=6, g=1.4, h=0.0, disp=False, seed=2))
    disp = ev.analyse(Settings(n=6, g=1.4, h=0.0, disp=True, r=1.0, seed=2))
    assert not np.allclose(flat.probs["Wahrheit"], disp.probs["Wahrheit"])
    assert Settings(disp=False, r=1.0).shape is None and Settings(disp=True, r=1.0).shape == 1.0


def test_verdict_cases():
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=8, k=2, h=0.0, fit_h=False, seed=3)))
    assert level == "info" and "Tordifferenz" in text and "Der Gewinn des Tormodells liegt in der Vorhersage" in text
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=8, k=4, home="setzliste", seed=3)))
    assert "geschätzter Heimvorteil" in text
    bad = next(a for a in (ev.analyse(Settings(n=8, k=4, disp=True, r=0.5, seed=s)) for s in range(100000, 100060)) if a.kl["Poisson"] > a.kl["Davidson"])
    level, text = ev.verdict_analysis(bad)
    assert level == "warning" and "überstreut" in text


def test_sweeps_have_consistent_shapes():
    seeds = C.SWEEP_SEEDS[:15]
    rows = ev.sweep_value(ks=(2,), seeds=seeds)
    assert rows[0]["same_order"] == rows[0]["tie_free"] and abs(rows[0]["Massey"] - rows[0]["Poisson"]) < 1e-12
    eq = ev.sweep_equivalent(ks=(2,), grid=(1, 2, 4, 8), seeds=seeds)
    assert eq[0]["k_bt"] >= 1.0 and eq[0]["factor"] == pytest.approx(eq[0]["k_bt"] / 2)
    lv = ev.sweep_level(gs=(0.5, 3.0), seeds=seeds)
    assert lv[0]["draw"] > lv[1]["draw"]
    dr = ev.sweep_draws(seeds=seeds)
    assert [r["model"] for r in dr][-1] == "Poisson" and dr[-1]["kl"] < dr[1]["kl"]
    pr = ev.sweep_predict(ks=(2,), seeds=seeds)
    assert pr[0]["Poisson"] < pr[0]["BT + Remisquote"]
    hm = ev.sweep_home(modes=("alternierend", "setzliste"), seeds=seeds)
    assert [r["mode"] for r in hm] == ["alternierend", "setzliste"] and hm[1]["strong_home"] > 0.6
    ds = ev.sweep_dispersion(rs=(None, 1.0), seeds=seeds)
    assert ds[1]["kl_po"] > ds[0]["kl_po"]


def test_decimal_comma_helpers_and_standardised_error():
    assert ev.de(0.5) == "0,50" and ev.de(3.14159, 3) == "3,142" and ev.pct1(0.0526) == "5,3" and ev.pct(0.405) in ("40", "41")
    assert ev.std_rmse([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(0.0, abs=1e-12)
    assert ev.std_rmse([3.0, 2.0, 1.0], [10.0, 20.0, 30.0]) == pytest.approx(2.0)
    assert math.isnan(ev.std_rmse([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
