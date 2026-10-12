"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import trm_evaluation as ev


@pytest.fixture(scope="module")
def value():
    return {r["k"]: r for r in ev.sweep_value()}


@pytest.fixture(scope="module")
def equivalent():
    return {r["k"]: r for r in ev.sweep_equivalent()}


@pytest.fixture(scope="module")
def level():
    return {r["g"]: r for r in ev.sweep_level()}


@pytest.fixture(scope="module")
def draws():
    return {r["model"]: r for r in ev.sweep_draws()}


@pytest.fixture(scope="module")
def predict():
    return {r["k"]: r for r in ev.sweep_predict()}


@pytest.fixture(scope="module")
def home():
    return {r["mode"]: r for r in ev.sweep_home()}


@pytest.fixture(scope="module")
def disp():
    return {r["r"]: r for r in ev.sweep_dispersion()}


# --- Experiment 1 ------------------------------------------------------------------------------------------------------------------------

def test_goals_order_better_than_win_loss_and_the_order_is_the_goal_difference_order(value):
    expected = {1: (0.684, 0.760), 2: (0.794, 0.849), 4: (0.880, 0.924), 8: (0.930, 0.964)}
    for k, (bt, po) in expected.items():
        assert value[k]["Bradley-Terry"] == pytest.approx(bt, abs=0.006) and value[k]["Poisson"] == pytest.approx(po, abs=0.006), k
        assert value[k]["Massey"] == value[k]["Poisson"]                         # dieselbe Ordnung, also dasselbe τ
        assert value[k]["Poisson"] > value[k]["Bradley-Terry"] + 0.03
    assert [(value[k]["same_order"], value[k]["tie_free"]) for k in (1, 2, 4, 8)] == [(150, 150), (216, 216), (265, 265), (290, 290)]


def test_davidson_orders_like_bradley_terry_with_draws_as_half_a_win(value):
    for k in (1, 2, 4, 8):
        assert value[k]["Davidson"] == value[k]["Bradley-Terry"]
        assert value[k]["same_points"] == value[k]["points_tie_free"]
    assert [value[k]["points_tie_free"] for k in (1, 2, 4, 8)] == [24, 107, 190, 243]
    assert value[2]["draw"] == pytest.approx(0.183, abs=0.006)


def test_goals_are_worth_about_one_and_a_half_to_two_games(equivalent):
    assert equivalent[1]["factor"] == pytest.approx(1.70, abs=0.08) and equivalent[2]["factor"] == pytest.approx(1.57, abs=0.08)
    assert equivalent[4]["factor"] == pytest.approx(1.84, abs=0.1)
    assert all(1.5 <= e["factor"] <= 1.9 for e in equivalent.values())
    assert equivalent[2]["k_bt"] == pytest.approx(3.1, abs=0.2)


# --- Experiment 2 ------------------------------------------------------------------------------------------------------------------------

def test_fewer_goals_mean_more_draws_and_goals_help_at_every_level(level):
    expected = {0.5: (0.405, 0.667, 0.707), 1.4: (0.183, 0.794, 0.849), 3.0: (0.092, 0.868, 0.921), 10.0: (0.027, 0.926, 0.988), 40.0: (0.006, 0.971, 1.0)}
    for g, (draw, bt, po) in expected.items():
        assert level[g]["draw"] == pytest.approx(draw, abs=0.006), g
        assert level[g]["Bradley-Terry"] == pytest.approx(bt, abs=0.006) and level[g]["Poisson"] == pytest.approx(po, abs=0.006), g
        assert level[g]["Poisson"] > level[g]["Bradley-Terry"]
    assert level[40.0]["Poisson"] == 1.0


# --- Experiment 3 ------------------------------------------------------------------------------------------------------------------------

def test_the_draw_treatment_does_not_change_the_order_but_the_prediction(draws):
    assert draws["BT, Remis verworfen"]["tau"] == pytest.approx(0.790, abs=0.006)
    assert draws["BT, Remis = ½"]["tau"] == draws["Davidson"]["tau"] == pytest.approx(0.794, abs=0.006)
    assert draws["Poisson"]["tau"] == pytest.approx(0.849, abs=0.006)
    assert draws["BT, Remis verworfen"]["kl"] == pytest.approx(0.082, abs=0.004) and draws["BT, Remis = ½"]["kl"] == pytest.approx(0.088, abs=0.004)
    assert draws["Davidson"]["kl"] == pytest.approx(0.075, abs=0.004) and draws["Poisson"]["kl"] == pytest.approx(0.034, abs=0.003)
    assert draws["Poisson"]["kl"] < 0.5 * draws["Davidson"]["kl"]


def test_the_goal_model_predicts_better_at_every_number_of_games(predict):
    expected = {1: (0.174, 0.150, 0.071), 2: (0.094, 0.082, 0.038), 4: (0.064, 0.046, 0.019), 10: (0.041, 0.020, 0.008)}
    for k, (bt, dv, po) in expected.items():
        assert predict[k]["BT + Remisquote"] == pytest.approx(bt, abs=0.005), k
        assert predict[k]["Davidson"] == pytest.approx(dv, abs=0.005), k
        assert predict[k]["Poisson"] == pytest.approx(po, abs=0.003), k
        assert predict[k]["Poisson"] < predict[k]["Davidson"] < predict[k]["BT + Remisquote"]


# --- Experiment 4 ------------------------------------------------------------------------------------------------------------------------

def test_with_balanced_home_rights_ignoring_the_advantage_costs_prediction_not_rank(home):
    r = home["alternierend"]
    assert r["tau0"] == r["tau1"] == r["tau_m"] and r["tau0"] == pytest.approx(0.847, abs=0.006)
    assert r["h_hat"] == pytest.approx(0.306, abs=0.02)
    assert r["kl0"] == pytest.approx(0.053, abs=0.004) and r["kl1"] == pytest.approx(0.038, abs=0.004)
    assert home["zufaellig"]["h_hat"] == pytest.approx(0.297, abs=0.02) and home["zufaellig"]["strong_home"] == pytest.approx(0.496, abs=0.01)


def test_when_home_rights_follow_strength_the_estimated_advantage_moves(home):
    s, u = home["setzliste"], home["umgekehrt"]
    assert s["strong_home"] == pytest.approx(0.669, abs=0.01) and u["strong_home"] == pytest.approx(0.322, abs=0.01)
    assert s["h_hat"] == pytest.approx(0.413, abs=0.025) and u["h_hat"] == pytest.approx(0.198, abs=0.025)
    assert s["h_hat"] > 0.35 and u["h_hat"] < 0.25                                   # nach oben bzw. unten vom wahren 0,30 weg
    assert s["tau0"] == pytest.approx(0.886, abs=0.006) and s["tau1"] == pytest.approx(0.852, abs=0.006)
    assert s["tau0"] > s["tau1"] + 0.02                                              # mit Heimvorteil schlechter
    assert u["tau0"] == pytest.approx(0.820, abs=0.006) and u["tau1"] == pytest.approx(0.843, abs=0.006)
    assert u["tau1"] > u["tau0"] + 0.015                                             # mit Heimvorteil besser
    assert s["err0"] < s["err1"] and u["err1"] < u["err0"]


# --- Experiment 5 ------------------------------------------------------------------------------------------------------------------------

def test_overdispersion_breaks_the_probabilities_but_not_the_rank(disp):
    expected = {None: (0.880, 0.924, 0.042, 0.017), 8.0: (0.855, 0.904, 0.036, 0.020), 2.0: (0.807, 0.856, 0.029, 0.044), 0.7: (0.700, 0.774, 0.025, 0.123)}
    for r, (tb, tp, kd, kp) in expected.items():
        assert disp[r]["tau_bt"] == pytest.approx(tb, abs=0.007), r
        assert disp[r]["tau_po"] == pytest.approx(tp, abs=0.007), r
        assert disp[r]["kl_dv"] == pytest.approx(kd, abs=0.004), r
        assert disp[r]["kl_po"] == pytest.approx(kp, abs=0.004), r
        assert disp[r]["tau_po"] > disp[r]["tau_bt"]
    assert disp[8.0]["kl_po"] < disp[8.0]["kl_dv"] and disp[2.0]["kl_po"] > disp[2.0]["kl_dv"]      # die Kreuzung liegt zwischen Form 8 und 2


# --- Texte -------------------------------------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_values(value, equivalent, level, draws, predict, home, disp):
    level_, text = ev.verdict_value(list(value.values()), list(equivalent.values()))
    assert ev.de(value[2]["Poisson"], 3) in text and "216 von 216" in text and ev.de(min(e["factor"] for e in equivalent.values()), 1) in text
    level_, text = ev.verdict_level(list(level.values()))
    assert "41 %" in text and "18 %" in text and "0,6 %" in text
    level_, text = ev.verdict_draws(list(draws.values()), list(predict.values()))
    assert ev.de(draws["Poisson"]["kl"], 3) in text and ev.de(draws["Davidson"]["kl"], 3) in text
    level_, text = ev.verdict_home(list(home.values()))
    assert ev.de(home["setzliste"]["h_hat"]) in text and ev.de(home["umgekehrt"]["h_hat"]) in text
    level_, text = ev.verdict_dispersion(list(disp.values()))
    assert level_ == "warning" and ev.de(disp[2.0]["kl_po"], 3) in text and ev.de(disp[0.7]["kl_dv"], 3) in text
