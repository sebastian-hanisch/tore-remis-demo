"""Tormodell (lokal; der gemeinsame Generator der Linie liefert nur Sieg oder Niederlage und bleibt unverändert): Stärken s aus turnier_generator.make_strengths,

    log λ_Heim = m + h + (s_H − s_G)/2,      log λ_Gast = m − (s_H − s_G)/2,      m = ln g − h/2.

Das Verhältnis der Torraten ist exp(s_H − s_G + h): dieselbe Skala wie in Bradley-Terry. g ist die mittlere Torzahl je Mannschaft bei Gleichstand ohne
Heimvorteil, h der Heimvorteil. Tore sind Poisson oder, mit Form r, negativ binomial (Gamma-Mischung der Rate = Überstreuung). Zufall nur über SplitMix64,
Zählwerte per Inversion der Verteilungsfunktion aus EINER Gleichverteilten je Torzahl (monoton in der Rate)."""

from dataclasses import dataclass

import numpy as np

import turnier_generator as T

_SALT = 0x6A6A
MAX_COUNT = 5000        # Sicherung der Inversionsschleife


def rate_log(s_home, s_away, g: float, h: float) -> tuple:
    """(log λ_Heim, log λ_Gast) für Mannschaften mit den Stärken s_home und s_away."""
    m = np.log(g) - h / 2.0
    d = (np.asarray(s_home, dtype=float) - np.asarray(s_away, dtype=float)) / 2.0
    return m + h + d, m - d


def inverse_counts(u, lam, r=None) -> np.ndarray:
    """Kleinstes k mit F(k) ≥ u für Poisson(λ) (r = None) bzw. negativ binomial mit Form r und Mittel λ. Vektorisiert; u, λ gleicher Form."""
    u = np.asarray(u, dtype=float)
    lam = np.asarray(lam, dtype=float)
    if r is None:
        pmf = np.exp(-lam)
    else:
        p = r / (r + lam)
        pmf = p ** r
    cdf = pmf.copy()
    k = np.zeros(u.shape, dtype=int)
    todo = cdf < u
    j = 0
    while todo.any() and j < MAX_COUNT:
        j += 1
        if r is None:
            pmf = pmf * lam / j
        else:
            pmf = pmf * (j - 1 + r) / j * (1.0 - p)
        cdf = cdf + pmf
        done = todo & (cdf >= u)
        k = np.where(done, j, k)
        todo = todo & ~done
    k = np.where(todo, j, k)
    return k


@dataclass(frozen=True)
class Games:
    strengths: np.ndarray        # wahre Stärke je Mannschaft (Mittel 0)
    home: np.ndarray             # Index der Heimmannschaft je Partie
    away: np.ndarray             # Index der Gastmannschaft
    goals_home: np.ndarray
    goals_away: np.ndarray
    names: tuple

    @property
    def n(self) -> int:
        return len(self.strengths)


def simulate_games(n: int, delta: float, k: int, g: float, h: float, home_mode: str, r, seed: int) -> Games:
    """Vollständiger Plan: jedes Paar k-mal. Heimrecht: 'alternierend' (abwechselnd, wer beginnt, entscheidet eine Münze), 'zufaellig' (Münze je Partie),
    'setzliste' (der Stärkere in 80 % der Partien daheim), 'umgekehrt' (der Schwächere in 80 %)."""
    if home_mode not in ("alternierend", "zufaellig", "setzliste", "umgekehrt"):
        raise ValueError(f"unbekannte Heimrecht-Verteilung: {home_mode}")
    s = T.make_strengths(n, delta)
    s = s - s.mean()
    rng = T.SplitMix64(seed ^ _SALT)
    perm = rng.permutation(n)
    names = tuple(T.LABELS[int(i)] for i in perm)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    coins = rng.uniform(m * (1 + k))
    first_home_i = coins[:m] < 0.5
    per_game = coins[m:].reshape(m, k)
    t = np.arange(k)[None, :]
    if home_mode == "alternierend":
        i_home = np.where(t % 2 == 0, first_home_i[:, None], ~first_home_i[:, None])
    elif home_mode == "zufaellig":
        i_home = per_game < 0.5
    elif home_mode == "setzliste":                    # j > i ist der Stärkere
        i_home = per_game < 0.2
    else:
        i_home = per_game < 0.8
    H = np.where(i_home, iu[:, None], ju[:, None]).ravel()
    A = np.where(i_home, ju[:, None], iu[:, None]).ravel()
    lh, la = rate_log(s[H], s[A], g, h)
    u = rng.uniform(2 * len(H))
    gh = inverse_counts(u[:len(H)], np.exp(lh), r)
    ga = inverse_counts(u[len(H):], np.exp(la), r)
    return Games(s, H.astype(int), A.astype(int), gh, ga, names)
