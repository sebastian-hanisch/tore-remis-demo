"""Literatur-Referenzwerte tore-remis-demo (ORACLE-PLAYBOOK §5.1). Ehrliche Vorbemerkung: Zu den Modellen dieser Demo (Poisson-Tore mit einer Stärke je Mannschaft,
Davidson mit Heimvorteil) fand sich KEINE eingesehene Quelle mit gedruckten Schätzwerten eines Beispieldatensatzes (siehe unten "Nicht eingesehen"). Es gibt
deshalb nur zwei Tests, und beide prüfen die DEFINITION aus der Quelle, nicht Schätzwerte; sie sind in den Docstrings als Handrechnung bzw. Formelauswertung
gekennzeichnet. Quellen am 2026-10-17 eingesehen (WebFetch; das PDF lokal mit pdftotext gelesen).

1. Davidson-Modell für Remis, Definition: Bush, S. et al. (2011): The Optimal Design of Choice Experiments that Incorporate Ties. Proc. 58th World Statistical
   Congress, Dublin (ISI), Session STS035, S. 2825-2826, https://isi-web.org/sites/default/files/import/files-2011/950043.pdf, Abschnitt "The Generalised Davidson
   Ties Model" (dort nach Davidson 1970 und Bush et al. 2010): Die Güte, dass zwei Optionen als gleich gut gelten, ist proportional zum geometrischen Mittel der
   Güten (Konstante ν > 0). Für zwei Optionen: P(i) = π_i / D, P(Remis) = ν √(π_i π_j) / D, D = π_i + π_j + ν √(π_i π_j). Die Zahlen unten sind VON HAND mit
   dieser Formel gerechnet (π = (2, 1), ν = 1: D = 3 + √2 = 4,414214; 2/D = 0,453082, √2/D = 0,320377, 1/D = 0,226541; π = (1, 1), ν = 2: 0,25 / 0,5 / 0,25), sie
   stehen nicht in der Quelle. Die Demo schreibt π_Heim = e^(θ_Heim + h), π_Gast = e^θ_Gast, ν = e^(log_nu).
2. Skellam-Verteilung: Wikipedia, "Skellam distribution" (https://en.wikipedia.org/wiki/Skellam_distribution), Abschnitt "Derivation": die Differenz zweier unabhängiger
   Poisson-Variablen mit Mitteln μ1, μ2 hat p(k) = e^(-(μ1+μ2)) (μ1/μ2)^(k/2) I_|k|(2√(μ1 μ2)) (I = modifizierte Besselfunktion erster Art). Daraus folgen die
   1-X-2-Wahrscheinlichkeiten zweier unabhängiger Poisson-Tore: P(Heim) = Σ_{k>0} p(k), P(Remis) = p(0), P(Gast) = Σ_{k<0} p(k). Der Test wertet die Formel mit
   scipy.special.ive aus (Formelauswertung, keine gedruckten Zahlen; die Seite druckt keine Zahlenbeispiele).

Nicht eingesehen (kein Test, siehe Bericht "offen"): Davidson (1970) selbst; Maher (1982); Dixon/Coles (1997) (Original nicht erreichbar, ihr Modell hat
zudem Angriff/Abwehr getrennt und die Niedrigergebnis-Korrektur ρ, die diese Demo nicht kennt: Tabellenwerte wären nicht vergleichbar); die R-Pakete BradleyTerry2
(Vignette 1.1.4: "does not implement the Davidson model", nur GenDavidson ohne gedruckte Ergebnisse) und prefmod/PlackettLuce lieferten keine gedruckten
Davidson-Schätzwerte.
"""

import math

import numpy as np
import pytest

import trm_models as M
import trm_outcomes as O

special = pytest.importorskip("scipy.special")


def test_davidson_probabilities_follow_the_geometric_mean_definition():
    # π = (2, 1), ν = 1, kein Heimvorteil: von Hand gerechnet, D = 3 + sqrt(2)
    p = M.davidson_probabilities(0.0, 0.0, math.log(2.0), 0.0)
    assert p.tolist() == pytest.approx([0.453082, 0.320377, 0.226541], abs=1e-6)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    # π = (1, 1), ν = 2: Remis 2/(1+1+2) = 1/2
    q = M.davidson_probabilities(math.log(2.0), 0.0, 0.0, 0.0)
    assert q.tolist() == pytest.approx([0.25, 0.5, 0.25], abs=1e-12)
    # Heimvorteil multipliziert π_Heim: θ = (0, 0), h = ln 2 ist dasselbe wie π = (2, 1)
    r = M.davidson_probabilities(0.0, math.log(2.0), 0.0, 0.0)
    assert r.tolist() == pytest.approx(p.tolist(), abs=1e-12)
    # Definition: P(Remis) = ν * sqrt(P(Heim) * P(Gast))
    for lnu, h, a, b in [(0.3, 0.2, 0.5, -0.4), (-1.0, 0.0, 1.0, 1.0), (1.2, 0.5, -0.7, 0.3)]:
        pr = M.davidson_probabilities(lnu, h, a, b)
        assert pr[1] == pytest.approx(math.exp(lnu) * math.sqrt(pr[0] * pr[2]), abs=1e-12)


def skellam_pmf(k, mu1, mu2):
    """Wikipedia: p(k; μ1, μ2) = e^(-(μ1+μ2)) (μ1/μ2)^(k/2) I_|k|(2 sqrt(μ1 μ2)); ive(v, z) = iv(v, z) e^(-z)."""
    z = 2.0 * math.sqrt(mu1 * mu2)
    return math.exp(z - mu1 - mu2) * (mu1 / mu2) ** (k / 2.0) * special.ive(abs(k), z)


@pytest.mark.parametrize("mu1,mu2", [(1.0, 1.0), (2.0, 1.0), (0.5, 3.0), (1.4 * math.exp(0.3), 1.4), (6.0, 5.0), (0.2, 0.2)])
def test_one_x_two_probabilities_equal_the_skellam_sums(mu1, mu2):
    ks = range(-80, 81)
    home = sum(skellam_pmf(k, mu1, mu2) for k in ks if k > 0)
    draw = skellam_pmf(0, mu1, mu2)
    away = sum(skellam_pmf(k, mu1, mu2) for k in ks if k < 0)
    assert home + draw + away == pytest.approx(1.0, abs=1e-9)
    ours = O.outcome_probabilities([mu1], [mu2])[0]
    assert ours.tolist() == pytest.approx([home, draw, away], abs=1e-9)
