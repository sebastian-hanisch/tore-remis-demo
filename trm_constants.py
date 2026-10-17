"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 4, 16, 8
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
K_MIN, K_MAX, DEFAULT_K = 1, 20, 2
G_MIN, G_MAX, G_STEP, DEFAULT_G = 0.5, 20.0, 0.1, 1.4         # mittlere Tore je Mannschaft bei Gleichstand
H_MIN, H_MAX, H_STEP, DEFAULT_H = 0.0, 0.6, 0.05, 0.3         # Heimvorteil auf der Log-Skala der Torrate (0,3 = 35 % mehr Heimtore)
R_MIN, R_MAX, R_STEP, DEFAULT_R = 0.5, 20.0, 0.5, 2.0         # Form der negativen Binomialverteilung (klein = stark überstreut)
SEED_MAX, DEFAULT_SEED = 999999, 43         # typischer Seed: ĥ 0,27, τ BT 0,79 gegen Poisson 0,86, Mehrverlust nahe den Mittelwerten der Experimente

HOME_LABELS = {
    "alternierend": "Alternierend (jedes Paar abwechselnd daheim)",
    "zufaellig": "Zufällig (Münzwurf je Partie)",
    "setzliste": "Setzliste (Stärkerer in 80 % der Partien daheim)",
    "umgekehrt": "Umgekehrt (Schwächerer in 80 % der Partien daheim)",
}
METHODS = ("Bradley-Terry", "Davidson", "Poisson", "Massey")
MODEL_LABELS = ("BT + Remisquote", "Davidson", "Poisson")      # Modelle mit 1-X-2-Vorhersage
TOL = 1e-9                                  # Wertungen, die sich um weniger unterscheiden, sind gleich (Gleitkomma-Rauschen)
MU_RIDGE = 1e-4                             # winziger Zuschlag auf μ und h: hält die Schätzung endlich, wenn nie ein Tor fällt
OUTCOME_TAIL = 10.0                         # Abschneiden der Torverteilung bei Mittel + 10 Standardabweichungen + 30

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100300))
EXP_N = 8
EXP_DELTA = 0.3
EXP_G = 1.4
VALUE_KS = (1, 2, 4, 8)
VALUE_EQ_KS = (1, 2, 3, 4, 6, 8, 12, 16)    # Raster für „wie viele BT-Partien entsprechen Poisson“
LEVEL_GS = (0.5, 1.4, 3.0, 10.0, 40.0)
LEVEL_K = 2
DRAW_K = 2
PREDICT_KS = (1, 2, 4, 10)
HOME_MODES = ("alternierend", "zufaellig", "setzliste", "umgekehrt")
HOME_K = 2
HOME_H = 0.3
DISP_RS = (None, 8.0, 2.0, 0.7)             # None = ohne Überstreuung (Poisson)
DISP_K = 4

COLOR_BT = "#8e6bb0"
COLOR_DAVIDSON = "#d98c1f"
COLOR_POISSON = "#2F6B65"
COLOR_MASSEY = "#c0392b"
COLOR_TRUTH = "#e67e22"
METHOD_COLORS = {"Bradley-Terry": COLOR_BT, "Davidson": COLOR_DAVIDSON, "Poisson": COLOR_POISSON, "Massey": COLOR_MASSEY}

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "k": DEFAULT_K, "g": DEFAULT_G, "h": DEFAULT_H, "home": "alternierend", "disp": False, "r": DEFAULT_R,
         "fit_h": True, "seed": DEFAULT_SEED}
PRESETS = {
    "Fußball-Niveau": dict(_BASE),
    "Torarmes Spiel": {**_BASE, "g": 0.5, "k": 4},
    "Heimrecht nach Setzliste": {**_BASE, "home": "setzliste", "k": 4},
    "Überstreute Tore": {**_BASE, "disp": True, "r": 2.0, "k": 4},
}
PRESET_HELP = {
    "Fußball-Niveau": "8 Mannschaften, 1,4 Tore je Mannschaft, 2 Partien je Paar, Heimvorteil 0,3 alternierend verteilt: das Tormodell ordnet treuer und sagt 1-X-2 besser voraus.",
    "Torarmes Spiel": "0,5 Tore je Mannschaft: 41 % Remis, Tore tragen wenig Information je Partie; mit 4 Partien je Paar.",
    "Heimrecht nach Setzliste": "Die Stärkeren haben in 80 % der Partien Heimrecht: Heimvorteil und Stärke sind bei wenigen Partien schwer zu trennen, der geschätzte Heimvorteil weicht vom wahren ab.",
    "Überstreute Tore": "Die Torzahlen streuen stärker als Poisson (Form 2): das Poisson-Modell wird überheblich, die Vorhersage fällt hinter Davidson.",
}
