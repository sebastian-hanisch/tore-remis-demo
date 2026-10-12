"""Schätzer (nur numpy): Poisson-Regression mit einer Stärke je Mannschaft, Davidson-Modell für Remis, Bradley-Terry mit Remis als halbem Sieg, Massey auf der
Tordifferenz; dazu Rangordnung mit Toleranz und Kendall-τ. Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b).

Alle Modelle: Ridge auf die Stärken θ (Vorannahme „Varianz 1/λ“, hier das Orakel-λ), ein winziger Zuschlag auf μ, h und ln ν. Newton-Verfahren mit
Schrittverkürzung; der Schritt wird in der Richtung „alle Stärken gleich“ zentriert (sonst verstärkt Rundungsrauschen die Schrittzahl, plattformabhängig)."""

from dataclasses import dataclass

import numpy as np

import trm_constants as C
import turnier_generator as T


def oracle_lambda(n: int, delta: float) -> float:
    """1/Varianz der Generator-Stärken: Var(s) = δ²(n² − 1)/12 (nur in der Simulation bekannt)."""
    return 12.0 / (delta ** 2 * (n ** 2 - 1))


def newton_ascent(f, grad, hess, x0, theta_idx, max_iter: int = 100, tol: float = 1e-10):
    """Maximiert eine konkave Funktion. `hess` liefert die Hesse-Matrix von f (negativ definit). Rückgabe (x, Schritte, konvergiert)."""
    x = np.asarray(x0, dtype=float).copy()
    fx = f(x)
    for it in range(1, max_iter + 1):
        g = grad(x)
        step = np.linalg.solve(-hess(x), g)
        step[theta_idx] -= step[theta_idx].mean()
        t = 1.0
        while True:
            cand = x + t * step
            fc = f(cand)
            if (np.isfinite(fc) and fc >= fx - 1e-12 * (1.0 + abs(fx))) or t < 1e-12:
                break
            t *= 0.5
        moved = float(np.abs(cand - x).max())
        x, fx = cand, fc
        if moved < tol:
            return x, it, True
    return x, max_iter, False


@dataclass(frozen=True)
class PoissonFit:
    mu: float
    h: float
    theta: np.ndarray
    iterations: int
    converged: bool


def fit_poisson(n: int, H, A, goals_home, goals_away, lam: float, with_h: bool = True) -> PoissonFit:
    """Poisson-Regression: log λ_Heim = μ + h + (θ_H − θ_G)/2, log λ_Gast = μ − (θ_H − θ_G)/2, Ridge λ auf θ. Ohne `with_h` ist h = 0 fest."""
    H, A = np.asarray(H), np.asarray(A)
    G = len(H)
    cols = 1 + (1 if with_h else 0) + n
    X = np.zeros((2 * G, cols))
    X[:, 0] = 1.0
    off = 2 if with_h else 1
    rows = np.arange(G)
    if with_h:
        X[rows, 1] = 1.0                                   # nur die Heimzeilen tragen h
    X[rows, off + H] += 0.5
    X[rows, off + A] -= 0.5
    X[G + rows, off + H] -= 0.5
    X[G + rows, off + A] += 0.5
    y = np.r_[np.asarray(goals_home, dtype=float), np.asarray(goals_away, dtype=float)]
    pen = np.r_[[C.MU_RIDGE] * (off), [lam] * n]
    pen[0] = C.MU_RIDGE

    def f(b):
        with np.errstate(over="ignore"):
            eta = X @ b
            return float((y * eta - np.exp(eta)).sum() - 0.5 * (pen * b * b).sum())

    def grad(b):
        with np.errstate(over="ignore"):
            return X.T @ (y - np.exp(X @ b)) - pen * b

    def hess(b):
        with np.errstate(over="ignore"):
            return -(X.T * np.exp(X @ b)) @ X - np.diag(pen)

    b0 = np.zeros(cols)
    b0[0] = np.log(max(float(y.mean()), 0.1))
    b, it, ok = newton_ascent(f, grad, hess, b0, np.arange(off, cols))
    theta = b[off:] - b[off:].mean()
    return PoissonFit(float(b[0]), float(b[1]) if with_h else 0.0, theta, it, ok)


@dataclass(frozen=True)
class DavidsonFit:
    log_nu: float
    h: float
    theta: np.ndarray
    iterations: int
    converged: bool


def davidson_probabilities(log_nu, h, theta_home, theta_away) -> np.ndarray:
    """(P Heim, P Remis, P Gast) als Matrix (m, 3): Logits (θ_H + h, ln ν + (θ_H + h + θ_G)/2, θ_G)."""
    a = np.asarray(theta_home, dtype=float) + h
    b = np.asarray(theta_away, dtype=float)
    mid = log_nu + (a + b) / 2.0
    den = np.logaddexp(np.logaddexp(a, b), mid)
    return np.exp(np.stack([a - den, mid - den, b - den], axis=-1))


def fit_davidson(n: int, H, A, goals_home, goals_away, lam: float, with_h: bool = True) -> DavidsonFit:
    """Davidson-Modell für Sieg/Remis/Niederlage: P(Remis) ∝ ν √(π_H π_G), Heimvorteil multipliziert π_H mit e^h. Tore gehen nur als Ausgang ein."""
    H, A = np.asarray(H), np.asarray(A)
    out = np.where(goals_home > goals_away, 0, np.where(goals_home == goals_away, 1, 2))
    is_h, is_d, is_a = (out == 0).astype(float), (out == 1).astype(float), (out == 2).astype(float)
    off = 2 if with_h else 1
    cols = off + n
    pen = np.r_[[C.MU_RIDGE] * off, [lam] * n]

    def unpack(b):
        return b[0], (b[1] if with_h else 0.0), b[off:]

    def f(b):
        lnu, h, th = unpack(b)
        p = davidson_probabilities(lnu, h, th[H], th[A])
        pick = np.where(out == 0, p[:, 0], np.where(out == 1, p[:, 1], p[:, 2]))
        with np.errstate(divide="ignore"):
            return float(np.log(pick).sum() - 0.5 * (pen * b * b).sum())

    def grad(b):
        lnu, h, th = unpack(b)
        p = davidson_probabilities(lnu, h, th[H], th[A])
        da = is_h + 0.5 * is_d - p[:, 0] - 0.5 * p[:, 1]
        db = is_a + 0.5 * is_d - p[:, 2] - 0.5 * p[:, 1]
        g = np.zeros(cols)
        g[0] = float((is_d - p[:, 1]).sum())
        if with_h:
            g[1] = float(da.sum())
        np.add.at(g, off + H, da)
        np.add.at(g, off + A, db)
        return g - pen * b

    def hess(b, eps=1e-5):
        Hm = np.zeros((cols, cols))
        for i in range(cols):
            e = np.zeros(cols)
            e[i] = eps
            Hm[:, i] = (grad(b + e) - grad(b - e)) / (2 * eps)
        return (Hm + Hm.T) / 2.0

    b, it, ok = newton_ascent(f, grad, hess, np.zeros(cols), np.arange(off, cols))
    lnu, h, th = unpack(b)
    return DavidsonFit(float(lnu), float(h), th - th.mean(), it, ok)


@dataclass(frozen=True)
class BradleyTerryFit:
    h: float
    theta: np.ndarray
    iterations: int
    converged: bool


def fit_bradley_terry(n: int, H, A, goals_home, goals_away, lam: float, draws: str = "half", with_h: bool = False) -> BradleyTerryFit:
    """Bradley-Terry mit Remis als halber Sieg ('half') oder verworfen ('drop'); optional Heimvorteil h (auf der Skala der Differenz)."""
    H, A = np.asarray(H), np.asarray(A)
    gh, ga = np.asarray(goals_home), np.asarray(goals_away)
    w = (gh > ga).astype(float) + 0.5 * (gh == ga)
    if draws == "drop":
        keep = gh != ga
        H, A, w = H[keep], A[keep], w[keep]
    elif draws != "half":
        raise ValueError(f"unbekannte Remis-Behandlung: {draws}")
    off = 1 if with_h else 0
    cols = off + n
    pen = np.r_[[C.MU_RIDGE] * off, [lam] * n]
    D = np.zeros((len(H), cols))
    if with_h:
        D[:, 0] = 1.0
    rows = np.arange(len(H))
    D[rows, off + H] += 1.0
    D[rows, off + A] -= 1.0

    def f(b):
        x = D @ b
        return float((w * -np.logaddexp(0.0, -x) + (1 - w) * -np.logaddexp(0.0, x)).sum() - 0.5 * (pen * b * b).sum())

    def grad(b):
        p = 0.5 * (1.0 + np.tanh(0.5 * (D @ b)))
        return D.T @ (w - p) - pen * b

    def hess(b):
        p = 0.5 * (1.0 + np.tanh(0.5 * (D @ b)))
        return -(D.T * (p * (1 - p))) @ D - np.diag(pen)

    b, it, ok = newton_ascent(f, grad, hess, np.zeros(cols), np.arange(off, cols))
    th = b[off:]
    return BradleyTerryFit(float(b[0]) if with_h else 0.0, th - th.mean(), it, ok)


def goal_difference_table(n: int, H, A, goals_home, goals_away) -> dict:
    """Tabelle je Mannschaft: Siege, Remis, Niederlagen, Tore dafür, Tore dagegen, Tordifferenz, Partien."""
    H, A = np.asarray(H), np.asarray(A)
    gh, ga = np.asarray(goals_home), np.asarray(goals_away)
    tab = {k: np.zeros(n, dtype=int) for k in ("wins", "draws", "losses", "gf", "ga", "games")}
    for idx, (own, opp) in enumerate(((H, A), (A, H))):
        gf, gg = (gh, ga) if idx == 0 else (ga, gh)
        np.add.at(tab["gf"], own, gf)
        np.add.at(tab["ga"], own, gg)
        np.add.at(tab["games"], own, 1)
        np.add.at(tab["wins"], own, (gf > gg).astype(int))
        np.add.at(tab["draws"], own, (gf == gg).astype(int))
        np.add.at(tab["losses"], own, (gf < gg).astype(int))
    tab["diff"] = tab["gf"] - tab["ga"]
    return tab


def massey_goal_difference(n: int, H, A, goals_home, goals_away) -> np.ndarray:
    """Massey: Kleinste Quadrate auf der Tordifferenz je Paar (Minimum-Norm). Im vollständigen, ausgeglichenen Plan ist die Ordnung die der Tordifferenz."""
    G = np.zeros((n, n))
    D = np.zeros((n, n))
    np.add.at(G, (H, A), 1.0)
    np.add.at(G, (A, H), 1.0)
    d = np.asarray(goals_home, dtype=float) - np.asarray(goals_away, dtype=float)
    np.add.at(D, (H, A), d)
    np.add.at(D, (A, H), -d)
    M = np.diag(G.sum(axis=1)) - G
    r, *_ = np.linalg.lstsq(M, D.sum(axis=1), rcond=None)
    return r


def rank_with_tolerance(score, rng: T.SplitMix64, tol: float = 1e-9) -> tuple:
    """Platz je Teilnehmer (0 = bester) nach absteigender Wertung; Wertungen, die sich um weniger als `tol` unterscheiden, sind gleich, das Los trennt sie."""
    score = np.asarray(score, dtype=float)
    key = np.round(score / tol).astype(np.int64)
    lots = rng.uniform(len(score))
    order = sorted(range(len(score)), key=lambda i: (-key[i], lots[i]))
    pos = [0] * len(score)
    for place, i in enumerate(order):
        pos[i] = place
    return tuple(pos)


def kendall_tau(rank, values) -> float:
    """τ zwischen den Plätzen (kleiner = besser) und einer Größe `values` (größer = besser); Paare mit gleichem Wert zählen nicht."""
    r = np.asarray(rank, dtype=float)
    v = np.asarray(values, dtype=float)
    iu = np.triu_indices(len(r), 1)
    sv = np.sign(v[iu[0]] - v[iu[1]])
    sr = np.sign(r[iu[1]] - r[iu[0]])
    counted = sv != 0
    if not counted.any():
        return float("nan")
    return float((sv[counted] * sr[counted]).mean())
