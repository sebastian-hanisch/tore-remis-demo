"""1-X-2 aus Torverteilungen (exakt über die Faltung) und der Mehrverlust eines Modells gegen die Wahrheit. Nur numpy."""

import numpy as np

import trm_constants as C
import trm_goals as G


def count_pmf(lam, r=None, cap: int | None = None) -> np.ndarray:
    """Wahrscheinlichkeiten 0..cap−1 je Rate (Zeilen): Poisson (r = None) oder negativ binomial mit Form r und Mittel λ; per Rekursion."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    if cap is None:
        var = lam if r is None else lam + lam ** 2 / r
        cap = int(np.ceil((lam + C.OUTCOME_TAIL * np.sqrt(var) + 30).max()))
    out = np.empty((len(lam), cap))
    if r is None:
        out[:, 0] = np.exp(-lam)
        for j in range(1, cap):
            out[:, j] = out[:, j - 1] * lam / j
    else:
        p = r / (r + lam)
        out[:, 0] = p ** r
        for j in range(1, cap):
            out[:, j] = out[:, j - 1] * (j - 1 + r) / j * (1.0 - p)
    return out


def outcome_probabilities(lam_home, lam_away, r=None) -> np.ndarray:
    """(P Heim, P Remis, P Gast) je Partie als Matrix (m, 3), unabhängige Tore; die abgeschnittene Randmasse wird durch die Summe geteilt."""
    lam_home = np.atleast_1d(np.asarray(lam_home, dtype=float))
    lam_away = np.atleast_1d(np.asarray(lam_away, dtype=float))
    top = float(max(lam_home.max(), lam_away.max()))
    var = top if r is None else top + top ** 2 / r
    cap = int(np.ceil(top + C.OUTCOME_TAIL * np.sqrt(var) + 30))
    ph, pa = count_pmf(lam_home, r, cap), count_pmf(lam_away, r, cap)
    cum_h, cum_a = np.cumsum(ph, axis=1), np.cumsum(pa, axis=1)
    below_a = np.concatenate([np.zeros((len(pa), 1)), cum_a[:, :-1]], axis=1)       # P(Gasttore < x)
    below_h = np.concatenate([np.zeros((len(ph), 1)), cum_h[:, :-1]], axis=1)
    p_home = (ph * below_a).sum(axis=1)
    p_away = (pa * below_h).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    out = np.stack([p_home, p_draw, p_away], axis=1)
    return out / out.sum(axis=1, keepdims=True)


def ordered_pairs(n: int) -> tuple:
    """Alle geordneten Paare (Heim i, Gast j), i ≠ j."""
    idx = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    return idx[:, 0], idx[:, 1]


def true_probabilities(strengths, g: float, h: float, r=None) -> np.ndarray:
    """Wahre 1-X-2-Wahrscheinlichkeiten aller geordneten Paare, Zeilen in der Reihenfolge von `ordered_pairs`."""
    s = np.asarray(strengths, dtype=float)
    I, J = ordered_pairs(len(s))
    lh, la = G.rate_log(s[I], s[J], g, h)
    return outcome_probabilities(np.exp(lh), np.exp(la), r)


def poisson_model_probabilities(mu: float, h: float, theta) -> np.ndarray:
    """1-X-2 des angepassten Poisson-Modells für alle geordneten Paare."""
    th = np.asarray(theta, dtype=float)
    I, J = ordered_pairs(len(th))
    d = (th[I] - th[J]) / 2.0
    return outcome_probabilities(np.exp(mu + h + d), np.exp(mu - d))


def davidson_model_probabilities(log_nu: float, h: float, theta) -> np.ndarray:
    from trm_models import davidson_probabilities
    th = np.asarray(theta, dtype=float)
    I, J = ordered_pairs(len(th))
    return davidson_probabilities(log_nu, h, th[I], th[J])


def bt_draw_rate_probabilities(h: float, theta, draw_rate: float) -> np.ndarray:
    """Bradley-Terry mit konstanter Remisquote d: (P Heim, P Remis, P Gast) = ((1 − d) σ(θ_i − θ_j + h), d, (1 − d)(1 − σ))."""
    th = np.asarray(theta, dtype=float)
    I, J = ordered_pairs(len(th))
    p = 0.5 * (1.0 + np.tanh(0.5 * (th[I] - th[J] + h)))
    return np.stack([(1 - draw_rate) * p, np.full_like(p, draw_rate), (1 - draw_rate) * (1 - p)], axis=1)


def kl_to_truth(true_p, model_p) -> float:
    """Mittlere KL-Divergenz wahr → Modell über die Zeilen (nat je Partie): der Mehrverlust des Modells gegen die Wahrheit."""
    t = np.asarray(true_p, dtype=float)
    q = np.clip(np.asarray(model_p, dtype=float), 1e-12, 1.0)
    return float((t * (np.log(np.clip(t, 1e-300, 1.0)) - np.log(q))).sum(axis=1).mean())
