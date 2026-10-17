"""Kennzahlen, Verdikt-Logik und Experimente (über die festen Instanzen SWEEP_SEEDS)."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import trm_constants as C
import trm_goals as G
import trm_models as M
import trm_outcomes as O
import turnier_generator as T
from trm_scenario import Settings, make_games

_LOT_SALT = 0xC0FFEE


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    games: G.Games
    table: dict                # Siege, Remis, Niederlagen, Tore dafür/dagegen, Differenz, Partien je Mannschaft
    poisson: M.PoissonFit
    davidson: M.DavidsonFit
    bt: M.BradleyTerryFit
    scores: dict               # Verfahren -> Wertung je Mannschaft
    rank: dict                 # Verfahren -> Platz je Mannschaft (0 = bester)
    tau_s: dict                # Verfahren -> τ gegen die Generator-Stärke
    draw_share: float
    kl: dict                   # Modell -> Mehrverlust der 1-X-2-Vorhersage gegen die Wahrheit (nat je Partie)
    probs: dict                # Modell -> (m, 3) Wahrscheinlichkeiten aller geordneten Paare ("Wahrheit" eingeschlossen)
    oracle_lam: float


def _tau(score, strengths, seed) -> float:
    return M.kendall_tau(M.rank_with_tolerance(score, T.SplitMix64(seed ^ _LOT_SALT), C.TOL), strengths)


@functools.lru_cache(maxsize=4096)
def analyse(settings: Settings) -> Analysis:
    """Gleiche Reglerwerte, gleiche Analyse (Settings ist unveränderlich)."""
    s = settings
    gm = make_games(s)
    n = s.n
    lam = M.oracle_lambda(n, s.delta)
    pf = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, with_h=s.fit_h)
    df = M.fit_davidson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, with_h=s.fit_h)
    bf = M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, draws="half", with_h=s.fit_h)
    scores = {"Bradley-Terry": bf.theta, "Davidson": df.theta, "Poisson": pf.theta,
              "Massey": M.massey_goal_difference(n, gm.home, gm.away, gm.goals_home, gm.goals_away)}
    rank = {m: M.rank_with_tolerance(scores[m], T.SplitMix64(s.seed ^ _LOT_SALT), C.TOL) for m in C.METHODS}
    draw = float((gm.goals_home == gm.goals_away).mean())
    truth = O.true_probabilities(gm.strengths, s.g, s.h, s.shape)
    probs = {"Wahrheit": truth,
             "BT + Remisquote": O.bt_draw_rate_probabilities(bf.h, bf.theta, draw),
             "Davidson": O.davidson_model_probabilities(df.log_nu, df.h, df.theta),
             "Poisson": O.poisson_model_probabilities(pf.mu, pf.h, pf.theta)}
    return Analysis(
        settings=s, games=gm, table=M.goal_difference_table(n, gm.home, gm.away, gm.goals_home, gm.goals_away),
        poisson=pf, davidson=df, bt=bf, scores=scores, rank=rank,
        tau_s={m: M.kendall_tau(rank[m], gm.strengths) for m in C.METHODS}, draw_share=draw,
        kl={m: O.kl_to_truth(truth, probs[m]) for m in C.MODEL_LABELS}, probs=probs, oracle_lam=lam,
    )


def pair_probabilities(a: Analysis, home: int, away: int) -> dict:
    """1-X-2 (Heim, Remis, Gast) für die Paarung (home gegen away) aus Wahrheit und den drei Modellen."""
    I, J = O.ordered_pairs(a.settings.n)
    row = int(np.nonzero((I == home) & (J == away))[0][0])
    return {m: tuple(float(x) for x in p[row]) for m, p in a.probs.items()}


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def pct1(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",")


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else float("nan")


def std_rmse(theta, strengths) -> float:
    """Skalenfreier Fehler der Stärken: RMSE zwischen den standardisierten θ̂ und s (die Skalen der Modelle unterscheiden sich)."""
    t, s = np.asarray(theta, dtype=float), np.asarray(strengths, dtype=float)
    if t.std() == 0 or s.std() == 0:
        return float("nan")
    return float(np.sqrt(np.mean(((t - t.mean()) / t.std() - (s - s.mean()) / s.std()) ** 2)))


def _games(n, delta, k, g, h, home, r, seed):
    return G.simulate_games(n, delta, k, g, h, home, r, seed)


# --- Experiment 1: was Tore wert sind -------------------------------------------------------------------------------------------------------

def sweep_value(ks=C.VALUE_KS, n=C.EXP_N, delta=C.EXP_DELTA, g=C.EXP_G, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """τ gegen die Stärke nach Partien je Paar (kein Heimvorteil, vollständig, alternierend): Bradley-Terry (Remis = ½), Davidson, Poisson, Massey auf der
    Tordifferenz; dazu, wie oft die Poisson-Ordnung die der Tordifferenz ist (nur Turniere ohne Gleichstand der Tordifferenz)."""
    lam = M.oracle_lambda(n, delta)
    rows = []
    for k in ks:
        acc = {m: [] for m in C.METHODS}
        draws, same, free, same_pts, free_pts = [], 0, 0, 0, 0
        for sd in seeds:
            gm = _games(n, delta, k, g, 0.0, "alternierend", None, sd)
            draws.append(float((gm.goals_home == gm.goals_away).mean()))
            bt = M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, "half", False)
            dv = M.fit_davidson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False)
            po = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False)
            ms = M.massey_goal_difference(n, gm.home, gm.away, gm.goals_home, gm.goals_away)
            for m, v in (("Bradley-Terry", bt.theta), ("Davidson", dv.theta), ("Poisson", po.theta), ("Massey", ms)):
                acc[m].append(_tau(v, gm.strengths, sd))
            tab = M.goal_difference_table(n, gm.home, gm.away, gm.goals_home, gm.goals_away)
            if len(set(tab["diff"])) == n:
                free += 1
                same += int(list(np.argsort(-po.theta)) == list(np.argsort(-tab["diff"])))
            points = 2 * tab["wins"] + tab["draws"]            # zwei Punkte je Sieg, einer je Remis (ganzzahlig, damit Gleichstände exakt sind)
            if len(set(points)) == n:
                free_pts += 1
                same_pts += int(list(np.argsort(-dv.theta)) == list(np.argsort(-points)) and list(np.argsort(-bt.theta)) == list(np.argsort(-points)))
        rows.append({"k": k, "draw": _mean(draws), **{m: _mean(v) for m, v in acc.items()}, "same_order": same, "tie_free": free,
                     "same_points": same_pts, "points_tie_free": free_pts})
    return rows


def sweep_equivalent(ks=(1, 2, 4), grid=C.VALUE_EQ_KS, n=C.EXP_N, delta=C.EXP_DELTA, g=C.EXP_G, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Wie viele Bradley-Terry-Partien je Paar braucht es für die Rangtreue von Poisson bei k Partien (lineare Interpolation über das Raster)?"""
    lam = M.oracle_lambda(n, delta)
    bt_tau = []
    for k in grid:
        vals = []
        for sd in seeds:
            gm = _games(n, delta, k, g, 0.0, "alternierend", None, sd)
            vals.append(_tau(M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, "half", False).theta, gm.strengths, sd))
        bt_tau.append(_mean(vals))
    rows = []
    for k in ks:
        vals = []
        for sd in seeds:
            gm = _games(n, delta, k, g, 0.0, "alternierend", None, sd)
            vals.append(_tau(M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False).theta, gm.strengths, sd))
        po = _mean(vals)
        k_eq = float(np.interp(po, bt_tau, grid))
        rows.append({"k": k, "tau_poisson": po, "k_bt": k_eq, "factor": k_eq / k})
    return rows


# --- Experiment 2: das Torniveau --------------------------------------------------------------------------------------------------------------

def sweep_level(gs=C.LEVEL_GS, k=C.LEVEL_K, n=C.EXP_N, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    lam = M.oracle_lambda(n, delta)
    rows = []
    for g in gs:
        acc = {m: [] for m in ("Bradley-Terry", "Poisson", "Massey")}
        draws = []
        for sd in seeds:
            gm = _games(n, delta, k, g, 0.0, "alternierend", None, sd)
            draws.append(float((gm.goals_home == gm.goals_away).mean()))
            acc["Bradley-Terry"].append(_tau(M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, "half", False).theta, gm.strengths, sd))
            acc["Poisson"].append(_tau(M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False).theta, gm.strengths, sd))
            acc["Massey"].append(_tau(M.massey_goal_difference(n, gm.home, gm.away, gm.goals_home, gm.goals_away), gm.strengths, sd))
        rows.append({"g": g, "draw": _mean(draws), **{m: _mean(v) for m, v in acc.items()}})
    return rows


# --- Experiment 3: Remis und die Vorhersage ----------------------------------------------------------------------------------------------------

def sweep_draws(k=C.DRAW_K, n=C.EXP_N, delta=C.EXP_DELTA, g=C.EXP_G, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Remis behandeln (kein Heimvorteil): τ und Mehrverlust der 1-X-2-Vorhersage für Bradley-Terry mit verworfenem Remis, mit Remis = ½, Davidson, Poisson."""
    lam = M.oracle_lambda(n, delta)
    truth = None
    acc = {m: ([], []) for m in ("BT, Remis verworfen", "BT, Remis = ½", "Davidson", "Poisson")}
    for sd in seeds:
        gm = _games(n, delta, k, g, 0.0, "alternierend", None, sd)
        if truth is None:
            truth = O.true_probabilities(gm.strengths, g, 0.0)
        d_hat = float((gm.goals_home == gm.goals_away).mean())
        for label, draws in (("BT, Remis verworfen", "drop"), ("BT, Remis = ½", "half")):
            bt = M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, draws, False)
            acc[label][0].append(_tau(bt.theta, gm.strengths, sd))
            acc[label][1].append(O.kl_to_truth(truth, O.bt_draw_rate_probabilities(0.0, bt.theta, d_hat)))
        dv = M.fit_davidson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False)
        acc["Davidson"][0].append(_tau(dv.theta, gm.strengths, sd))
        acc["Davidson"][1].append(O.kl_to_truth(truth, O.davidson_model_probabilities(dv.log_nu, 0.0, dv.theta)))
        po = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False)
        acc["Poisson"][0].append(_tau(po.theta, gm.strengths, sd))
        acc["Poisson"][1].append(O.kl_to_truth(truth, O.poisson_model_probabilities(po.mu, 0.0, po.theta)))
    return [{"model": m, "tau": _mean(t), "kl": _mean(k_)} for m, (t, k_) in acc.items()]


def sweep_predict(ks=C.PREDICT_KS, n=C.EXP_N, delta=C.EXP_DELTA, g=C.EXP_G, h=C.HOME_H, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Mehrverlust der 1-X-2-Vorhersage nach Partien je Paar (Heimvorteil geschätzt, alternierend): BT + Remisquote, Davidson, Poisson."""
    lam = M.oracle_lambda(n, delta)
    truth = None
    rows = []
    for k in ks:
        acc = {m: [] for m in C.MODEL_LABELS}
        for sd in seeds:
            gm = _games(n, delta, k, g, h, "alternierend", None, sd)
            if truth is None:
                truth = O.true_probabilities(gm.strengths, g, h)
            d_hat = float((gm.goals_home == gm.goals_away).mean())
            bt = M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, "half", True)
            dv = M.fit_davidson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, True)
            po = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, True)
            acc["BT + Remisquote"].append(O.kl_to_truth(truth, O.bt_draw_rate_probabilities(bt.h, bt.theta, d_hat)))
            acc["Davidson"].append(O.kl_to_truth(truth, O.davidson_model_probabilities(dv.log_nu, dv.h, dv.theta)))
            acc["Poisson"].append(O.kl_to_truth(truth, O.poisson_model_probabilities(po.mu, po.h, po.theta)))
        rows.append({"k": k, **{m: _mean(v) for m, v in acc.items()}})
    return rows


# --- Experiment 4: Heimvorteil und wer Heimrecht hat ------------------------------------------------------------------------------------------

def sweep_home(modes=C.HOME_MODES, k=C.HOME_K, n=C.EXP_N, delta=C.EXP_DELTA, g=C.EXP_G, h=C.HOME_H, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Je Heimrecht-Verteilung: Poisson ohne und mit geschätztem Heimvorteil, Massey auf der Tordifferenz: τ, standardisierter Skalenfehler, ĥ, Mehrverlust."""
    lam = M.oracle_lambda(n, delta)
    rows = []
    for mode in modes:
        acc = {key: [] for key in ("tau0", "tau1", "tau_m", "err0", "err1", "err_m", "h_hat", "kl0", "kl1", "strong_home")}
        truth = None
        for sd in seeds:
            gm = _games(n, delta, k, g, h, mode, None, sd)
            if truth is None:
                truth = O.true_probabilities(gm.strengths, g, h)
            p0 = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False)
            p1 = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, True)
            ms = M.massey_goal_difference(n, gm.home, gm.away, gm.goals_home, gm.goals_away)
            acc["tau0"].append(_tau(p0.theta, gm.strengths, sd))
            acc["tau1"].append(_tau(p1.theta, gm.strengths, sd))
            acc["tau_m"].append(_tau(ms, gm.strengths, sd))
            acc["err0"].append(std_rmse(p0.theta, gm.strengths))
            acc["err1"].append(std_rmse(p1.theta, gm.strengths))
            acc["err_m"].append(std_rmse(ms, gm.strengths))
            acc["h_hat"].append(p1.h)
            acc["kl0"].append(O.kl_to_truth(truth, O.poisson_model_probabilities(p0.mu, 0.0, p0.theta)))
            acc["kl1"].append(O.kl_to_truth(truth, O.poisson_model_probabilities(p1.mu, p1.h, p1.theta)))
            acc["strong_home"].append(float((gm.home >= n // 2).mean()))
        rows.append({"mode": mode, **{key: _mean(v) for key, v in acc.items()}})
    return rows


# --- Experiment 5: Überstreuung ----------------------------------------------------------------------------------------------------------------

def sweep_dispersion(rs=C.DISP_RS, k=C.DISP_K, n=C.EXP_N, delta=C.EXP_DELTA, g=C.EXP_G, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Negativ binomiale Tore mit Form r (kein Heimvorteil): τ von Bradley-Terry und Poisson, Mehrverlust von Davidson und Poisson gegen die wahre, überstreute
    Verteilung."""
    lam = M.oracle_lambda(n, delta)
    rows = []
    for r in rs:
        acc = {key: [] for key in ("tau_bt", "tau_po", "kl_dv", "kl_po")}
        truth = None
        for sd in seeds:
            gm = _games(n, delta, k, g, 0.0, "alternierend", r, sd)
            if truth is None:
                truth = O.true_probabilities(gm.strengths, g, 0.0, r)
            bt = M.fit_bradley_terry(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, "half", False)
            dv = M.fit_davidson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False)
            po = M.fit_poisson(n, gm.home, gm.away, gm.goals_home, gm.goals_away, lam, False)
            acc["tau_bt"].append(_tau(bt.theta, gm.strengths, sd))
            acc["tau_po"].append(_tau(po.theta, gm.strengths, sd))
            acc["kl_dv"].append(O.kl_to_truth(truth, O.davidson_model_probabilities(dv.log_nu, 0.0, dv.theta)))
            acc["kl_po"].append(O.kl_to_truth(truth, O.poisson_model_probabilities(po.mu, 0.0, po.theta)))
        rows.append({"r": r, **{key: _mean(v) for key, v in acc.items()}})
    return rows


# --- Verdikte (live aus den Messwerten) --------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Primäransicht: Was lesen die Tore, und wo tragen die Annahmen nicht?"""
    s = a.settings
    t, kl = a.tau_s, a.kl
    same = a.rank["Poisson"] == a.rank["Massey"]
    text = (f"τ gegen die Generator-Stärke: Bradley-Terry {de(t['Bradley-Terry'])}, Davidson {de(t['Davidson'])}, Poisson {de(t['Poisson'])}, Massey auf der "
            f"Tordifferenz {de(t['Massey'])}. Mehrverlust der 1-X-2-Vorhersage gegen die Wahrheit: BT + Remisquote {de(kl['BT + Remisquote'], 3)}, "
            f"Davidson {de(kl['Davidson'], 3)}, Poisson {de(kl['Poisson'], 3)} nat je Partie.")
    if same:
        text += " Die Poisson-Ordnung ist hier die der Tordifferenz (Massey): Der Gewinn des Tormodells liegt in der Vorhersage, nicht im Rang."
    else:
        text += " Poisson und Massey ordnen hier verschieden: Heimrecht oder Überstreuung stören die Gleichheit, die im ausgeglichenen Plan gilt."
    if s.fit_h and s.h > 0 and s.home in ("setzliste", "umgekehrt"):
        text += (f" Das Heimrecht hängt hier an der Stärke: geschätzter Heimvorteil {de(a.poisson.h)} statt {de(s.h)}; Stärke und Heimvorteil sind nur schwer "
                 "zu trennen.")
    if s.disp and kl["Poisson"] > kl["Davidson"]:
        return "warning", text + " **Die Tore sind überstreut:** Das Poisson-Modell ist überheblich und sagt schlechter voraus als Davidson."
    return "info", text


def verdict_value(rows, equivalent) -> tuple[str, str]:
    r2 = next(r for r in rows if r["k"] == 2)
    r4 = next(r for r in rows if r["k"] == 4)
    fac = [e["factor"] for e in equivalent]
    return "info", (f"Bei 2 Partien je Paar ist τ gegen die Stärke {de(r2['Bradley-Terry'], 3)} (Bradley-Terry, Remis = ½), {de(r2['Davidson'], 3)} (Davidson) und "
                    f"**{de(r2['Poisson'], 3)}** (Poisson); bei 4 Partien {de(r4['Bradley-Terry'], 3)} / {de(r4['Davidson'], 3)} / **{de(r4['Poisson'], 3)}**. "
                    f"Das Tormodell entspricht Bradley-Terry mit dem **{de(min(fac), 1)}- bis {de(max(fac), 1)}-fachen** an Partien je Paar. Aber: Die Poisson-Ordnung "
                    f"ist in {r2['same_order']} von {r2['tie_free']} gleichstandsfreien Turnieren (2 Partien) die der Tordifferenz; Massey hat sie schon "
                    "(τ gleich).")


def verdict_level(rows) -> tuple[str, str]:
    lo = min(rows, key=lambda r: r["g"])
    hi = max(rows, key=lambda r: r["g"])
    mid = next(r for r in rows if r["g"] == 1.4)
    return "info", (f"Je weniger Tore, desto mehr Remis: {pct(lo['draw'])} % bei {de(lo['g'], 1)} Toren je Mannschaft, {pct(mid['draw'])} % bei 1,4, "
                    f"{pct1(hi['draw'])} % bei {de(hi['g'], 0)}. Das Tormodell ordnet auf jedem Niveau besser als Bradley-Terry ({de(lo['Poisson'], 3)} gegen "
                    f"{de(lo['Bradley-Terry'], 3)} bei {de(lo['g'], 1)}, {de(hi['Poisson'], 3)} gegen {de(hi['Bradley-Terry'], 3)} bei {de(hi['g'], 0)}); "
                    "bei sehr vielen Toren ist der Sieg-oder-Niederlage-Rang schon fast perfekt.")


def verdict_draws(draw_rows, predict_rows) -> tuple[str, str]:
    d = {r["model"]: r for r in draw_rows}
    p = {r["k"]: r for r in predict_rows}
    taus = [d[m]["tau"] for m in ("BT, Remis verworfen", "BT, Remis = ½", "Davidson")]
    return "info", (f"Die Behandlung des Remis ändert die Rangordnung im Mittel kaum (τ {de(min(taus), 3)} bis {de(max(taus), 3)} für verwerfen, ½ und Davidson), nur die "
                    f"Vorhersage: Mehrverlust {de(d['BT, Remis = ½']['kl'], 3)} (BT + Remisquote), {de(d['Davidson']['kl'], 3)} (Davidson), "
                    f"**{de(d['Poisson']['kl'], 3)}** (Poisson). Mit mehr Partien bleibt das Tormodell vorn ({de(p[10]['Davidson'], 3)} gegen "
                    f"{de(p[10]['Poisson'], 3)} bei 10 Partien je Paar, Davidson gegen Poisson); Bradley-Terry mit fester Remisquote lernt nicht, dass Remis "
                    "zwischen Gleichstarken häufiger ist.")


def verdict_home(rows) -> tuple[str, str]:
    r = {x["mode"]: x for x in rows}
    return "warning", (f"Bei ausgeglichenem Heimrecht ist ĥ = {de(r['alternierend']['h_hat'])} nahe dem wahren 0,30 und das Ignorieren kostet nur Vorhersage "
                       f"(Mehrverlust {de(r['alternierend']['kl0'], 3)} gegen {de(r['alternierend']['kl1'], 3)}), nicht Rang. **Hängt das Heimrecht an der Stärke, "
                       f"verschiebt sich ĥ:** {de(r['setzliste']['h_hat'])} bei Setzliste, {de(r['umgekehrt']['h_hat'])} umgekehrt. Bei Setzliste ordnet das "
                       f"Modell ohne Heimvorteil sogar besser (τ {de(r['setzliste']['tau0'], 3)} gegen {de(r['setzliste']['tau1'], 3)} mit); umgekehrt hilft das "
                       f"Schätzen ({de(r['umgekehrt']['tau0'], 3)} gegen {de(r['umgekehrt']['tau1'], 3)}).")


def verdict_dispersion(rows) -> tuple[str, str]:
    r = {x["r"]: x for x in rows}
    return "warning", (f"Ohne Überstreuung ist der Mehrverlust von Poisson {de(r[None]['kl_po'], 3)}, von Davidson {de(r[None]['kl_dv'], 3)}. Mit Form 2 kehrt es "
                       f"sich um: Poisson {de(r[2.0]['kl_po'], 3)}, Davidson {de(r[2.0]['kl_dv'], 3)}; bei Form 0,7 sind es {de(r[0.7]['kl_po'], 3)} gegen "
                       f"{de(r[0.7]['kl_dv'], 3)}. Die Rangordnung bleibt vorn (τ {de(r[2.0]['tau_po'], 3)} gegen {de(r[2.0]['tau_bt'], 3)} bei Form 2): "
                       "Überstreuung bricht die Wahrscheinlichkeiten, nicht den Vorsprung im Rang.")
