# Tore, Remis, Heimvorteil: mehr Information als Sieg oder Niederlage – Streamlit-Demo

Sechstes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `bradley-terry-demo` und `massey-colley-demo`: Bradley-Terry sieht nur Sieg oder
Niederlage. Hier liest das Modell auch die **Tore**, das **Remis** und den **Heimvorteil**: ein Poisson-Modell mit einer Stärke je Mannschaft, ein Davidson-Modell für
Sieg-Remis-Niederlage und der Heimvorteil als Faktor der Heim-Torrate. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel (Stärken aus dem Generator
Version 1, wortgleich; die Tore sind ein lokales Modell darauf); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Was gewinnt ein Modell, das Tore, Remis und Heimvorteil liest, gegenüber Sieg oder Niederlage? **Rang:** im vollständigen, ausgeglichenen Plan ordnet das
Poisson-Modell **exakt wie die Tordifferenz**, also wie Massey; es trifft die Stärke treuer als Bradley-Terry (rund das 1,6- bis 1,8-fache an Partien). **Vorhersage:**
der Mehrverlust der 1-X-2-Wahrscheinlichkeiten gegen die Wahrheit halbiert sich gegenüber Davidson. Das Remis anders zu behandeln ändert den Rang **nicht**. Der
Heimvorteil ist nur trennbar, wenn das Heimrecht nicht an der Stärke hängt. Überstreute Tore kippen die Vorhersage des Poisson-Modells, nicht seinen Rang.

## Modell

- **Tore:** je Mannschaft Poisson mit Rate λ; Heimrate exp(μ + h + (θ_H − θ_G)/2), Gastrate exp(μ − (θ_H − θ_G)/2). Das Torverhältnis ist exp(θ_H − θ_G + h), dieselbe
  Skala wie Bradley-Terry; μ = ln g − h/2 setzt das Torniveau (g = mittlere Tore je Mannschaft bei Gleichstand), h ist der Heimvorteil. Optional negativ binomial mit
  Form r (Gamma-Mischung der Rate = Überstreuung). Zählwerte per Inversion der Verteilungsfunktion aus einer Gleichverteilten (SplitMix64), monoton in der Rate.
- **Spielplan:** vollständig, jedes Paar k-mal; Heimrecht alternierend, zufällig, nach Setzliste (der Stärkere in 80 % der Partien daheim) oder umgekehrt.
- **Schätzer:** Poisson-Regression mit Ridge auf θ (Vorannahme „Varianz 1/λ“, hier das Orakel-λ = 12/(δ²(n²−1)) wie in Stück 5), Newton-Verfahren mit Schrittverkürzung und
  zentriertem Schritt; Davidson (ν, optional h); Bradley-Terry mit Remis = ½ oder verworfen; Massey auf der Tordifferenz.
- **1-X-2:** exakt über die Faltung der Torverteilungen (bei Mittel plus zehn Standardabweichungen abgeschnitten). **Mehrverlust:** mittlere KL-Divergenz von der wahren zur
  Modellverteilung über alle geordneten Paare (nat je Partie).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 300 feste Turniere (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanzen mit 8 Teilnehmern, unabhängig vom
  Regler.
- **Maße:** Kendall-τ gegen die Generator-Stärke, Remis-Anteil, geschätzter Heimvorteil, standardisierter Skalenfehler der Stärken, Mehrverlust der 1-X-2-Vorhersage.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit `siegtabelle-demo`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 300 feste Turniere, 8 Teilnehmer, Stärkeabstand 0,3, vollständiger Plan, wenn nicht anders genannt.

| Frage | Befund | Test |
|---|---|---|
| Rangtreue nach Partien je Paar (1,4 Tore, kein h) | τ gegen die Stärke bei 1, 2, 4, 8 Partien: **Bradley-Terry 0,684 / 0,794 / 0,880 / 0,930, Poisson 0,760 / 0,849 / 0,924 / 0,964**; Massey auf der Tordifferenz identisch mit Poisson. | `test_goals_order_better_than_win_loss_and_the_order_is_the_goal_difference_order` |
| Ist die Poisson-Ordnung die der Tordifferenz? | **Ja: 150 von 150, 216 von 216, 265 von 265, 290 von 290** gleichstandsfreien Turnieren (1, 2, 4, 8 Partien). | dasselbe |
| Und die von Davidson? | Davidson und Bradley-Terry (Remis = ½) ordnen in jedem Turnier ohne Punktgleichstand **nach den Punkten** (2 je Sieg, 1 je Remis): 24/24, 107/107, 190/190, 243/243; τ identisch. | `test_davidson_orders_like_bradley_terry_with_draws_as_half_a_win` |
| Wie viele Partien sind Tore wert? | Poisson mit 1, 2, 4 Partien ≈ Bradley-Terry mit 1,7 / 3,1 / 7,4 Partien: **Faktor 1,6 bis 1,8**. | `test_goals_are_worth_about_one_and_a_half_to_two_games` |
| Torniveau (2 Partien) | Remis-Anteil bei 0,5 / 1,4 / 3 / 10 / 40 Toren: **41 / 18 / 9 / 2,7 / 0,6 %**. τ Bradley-Terry gegen Poisson: 0,667 / 0,707; 0,794 / 0,849; 0,868 / 0,921; 0,926 / 0,988; 0,971 / 1,000. | `test_fewer_goals_mean_more_draws_and_goals_help_at_every_level` |
| Remis behandeln (2 Partien, 1,4 Tore) | τ: verworfen 0,790, Remis = ½ 0,794, Davidson 0,794, Poisson 0,849. **Mehrverlust 1-X-2:** 0,082 / 0,088 / 0,075 / **0,034** nat je Partie. | `test_the_draw_treatment_does_not_change_the_order_but_the_prediction` |
| Vorhersage nach Partien (h = 0,3 geschätzt) | Mehrverlust BT + Remisquote / Davidson / Poisson: 1 Partie 0,174 / 0,150 / **0,071**; 2: 0,094 / 0,082 / **0,038**; 4: 0,064 / 0,046 / **0,019**; 10: 0,041 / 0,020 / **0,008**. | `test_the_goal_model_predicts_better_at_every_number_of_games` |
| Heimvorteil bei ausgeglichenem Heimrecht | ĥ = 0,31 (wahr 0,30); τ mit und ohne h gleich (0,847), der **Mehrverlust** sinkt von 0,053 auf 0,038. | `test_with_balanced_home_rights_ignoring_the_advantage_costs_prediction_not_rank` |
| Heimrecht hängt an der Stärke | Setzliste: ĥ = **0,41**, τ ohne h 0,886, mit h 0,852; umgekehrt: ĥ = **0,20**, τ ohne h 0,820, mit h 0,843. Der Skalenfehler der Stärken wird bei Setzliste mit h schlechter, umgekehrt besser. | `test_when_home_rights_follow_strength_the_estimated_advantage_moves` |
| Überstreuung (4 Partien, Form r) | Mehrverlust Poisson / Davidson: r = ∞ 0,017 / 0,042; r = 8 0,020 / 0,036; **r = 2 0,044 / 0,029**; r = 0,7 0,123 / 0,025. τ Poisson bleibt vor Bradley-Terry (0,856 gegen 0,807 bei r = 2). | `test_overdispersion_breaks_the_probabilities_but_not_the_rank` |

## Befunde und Korrekturen gegenüber dem Plan

- **Faktor 1,6 bis 1,8 statt 1,5 bis 1,7:** andere feste Turniere und ein anderer Zufallsstrom der Spielplan-Erzeugung (vektorisiert statt Münzwürfen in der Schleife).
- **Neu gegenüber der Vorab-Messreihe:** Davidson und Bradley-Terry mit Remis = ½ ordnen identisch (nach Punkten), wie Poisson und Massey (nach Tordifferenz). Die
  Behandlung des Remis ändert den Rang also nicht nur „kaum“, sondern im vollständigen Plan ohne Heimvorteil gar nicht.
- **Handwerte:** Die 1-X-2-Wahrscheinlichkeiten bei λ = 1 gegen 1 sind 0,3457 / 0,3085 / 0,3457 (der Plan nannte 0,3458 durch Aufrunden).
- **Dateien:** `trm_goals.py`, `trm_models.py`, `trm_outcomes.py` statt der geplanten Aufteilung; die Bradley-Terry-Anpassung ist eine eigene, in dieses Repo kopierte
  Fassung (kein Import aus Stück 5).
- **Seed:** Der Standard-Seed 43 ist typisch (ĥ 0,27, Mehrverlust nahe den Mittelwerten); der erste Standard-Seed schätzte ĥ = 0,59 und hätte die Vorschau
  irreführend gemacht.
- **Messartefakt der Vorab-Messreihe (E6):** Der Mehrverlust muss gegen die überstreute, nicht gegen die Poisson-Wahrheit gemessen werden; sonst kehrt sich die Aussage
  um. Die Demo rechnet gegen die negativ binomiale Wahrheit.

## Abgrenzung

- **Massey und Colley** (Stück 4): lineares System auf Punktedifferenzen; dort waren Differenzen ein **Ausreißer-Stresstest**, hier entstehen sie aus der Stärke. Im
  vollständigen Plan ist die Poisson-Ordnung die Massey-Ordnung; der Gewinn des Tormodells liegt in der Vorhersage.
- **Bradley-Terry** (Stück 5): Modell der Ausgänge ohne Remis und Heimvorteil; dieses Stück erweitert es (Davidson) und ersetzt es (Poisson).
- [Gruppenphase und K.-o.](https://sebastianhanisch-group-knockout-demo.streamlit.app/) (Turnierplanung) behandelt die Fußball-WM-Struktur, kein Torverteilungsmodell.

## Ehrliche Grenzen

- **Kein Rangvorteil gegenüber Massey** im vollständigen, ausgeglichenen Plan; der Gewinn liegt in der Vorhersage.
- **Eine Stärke je Mannschaft:** Angriff und Verteidigung getrennt (Maher, Dixon-Coles) fehlen; der Generator kennt sie nicht.
- **Unabhängige Tore:** keine Korrelation niedriger Ergebnisse, keine Zeitabhängigkeit, kein Torverlauf; Überstreuung ist nur als negative Binomialverteilung abgebildet.
- **Heimrecht-Verteilungen sind konstruiert;** die Konfundierung ist der Befund, kein Abbild realer Spielpläne.
- **Der Faktor 1,6 bis 1,8 gilt für Fußball-Niveau in der Simulation,** nicht für echte Sportarten; bei vielen Toren ist der Sieg-oder-Niederlage-Rang schon fast perfekt.
- **Alles simuliert**, vollständiger Plan, n ≤ 16.

## Quellen

- Maher, M. J. (1982): Modelling association football scores. *Statistica Neerlandica* 36 (3), 109–118, doi:10.1111/j.1467-9574.1982.tb00782.x (nach dem Abstract:
  unabhängiges Poisson-Modell mit Angriffs- und Abwehrstärke beschreibt Fußballergebnisse angemessen, mit kleinen systematischen Abweichungen; Original nicht eingesehen).
- Dixon, M. J., Coles, S. G. (1997): Modelling association football scores and inefficiencies in the football betting market. *Applied Statistics (JRSS C)* 46 (2),
  265–280 (Poisson-Regression, Korrektur für niedrige Ergebnisse, zeitveränderliche Stärke; hier nicht umgesetzt; nach Abstract und Sekundärquellen, Original nicht
  eingesehen).
- Davidson, R. R. (1970): On extending the Bradley-Terry model to accommodate ties in paired comparison experiments. *Journal of the American Statistical Association* 65,
  317–328 (Remis proportional zum geometrischen Mittel der Stärken; nach Sekundärquellen, Original nicht eingesehen).
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39 (3/4), 324–345.
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472.

## Tests

89 Tests, ein bis zwei Minuten: Einheiten gegen die von Hand gerechneten Mini-Instanzen (`test_goals.py`, `test_models.py`, `test_outcomes.py`), Generator
(`test_generator.py`, wortgleich mit der Wurzel), **Orakel** (`test_oracle_tore_remis.py`: Poisson, Davidson und Bradley-Terry gegen `scipy.optimize` auf der Partienliste,
Inversion gegen `scipy.stats.ppf` für Poisson und negativ binomial, 1-X-2 gegen die Brute-Force-Doppelsumme, Ordnungen gegen Tordifferenz und Punkte, Heimvorteil gegen die
geschlossene Form), Presets, ausblendbarer Regler (`test_hidden_widget_state.py`), Analyse und Verdikte (`test_evaluation.py`), alle Zahlen dieses README
(`test_claims.py`), PDF, Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, 40 Mutanten, nur gegen die Orakel-Tests): 21 getötet, 16 überlebt, 3 Zeitüberschreitungen (zählen nicht
in die Rate); Rate 57 %. Die 16 Überlebenden wurden einzeln auf einer Kopie gegen die Einheits- und Generator-Tests geprüft: **15 getötet** (zwei Rechenfehler in
Torrate und Faltung, drei in der Remisquoten-Vorhersage, zehn im gemeinsamen Generator durch `test_generator.py`), **1 äquivalent** (die Sicherheitsgrenze 30 → 31
beim Abschneiden der Torverteilung). Mit den Einheitstests zusammen sind damit alle nicht-äquivalenten Mutanten getötet; die Zahl nur gegen das Orakel ist die niedrige
Rate, weil das Orakel Schätzer und Verteilungen prüft, nicht die Handwerte.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `trm_goals.py` | Torraten, Inversion der Verteilungsfunktion, Spielplan mit Heimrecht |
| `trm_models.py` | Poisson-Regression, Davidson, Bradley-Terry mit Remis, Massey auf der Tordifferenz, Rangordnung, Kendall-τ |
| `trm_outcomes.py` | 1-X-2 aus Torverteilungen, Mehrverlust, Gegenmodelle |
| `trm_scenario.py` | Reglerwerte (`Settings`) und die Spiele |
| `trm_evaluation.py` | Analyse, fünf Experimente über die festen Turniere, Verdikte |
| `trm_presets.py`, `trm_constants.py` | Permalink, Presets, ausblendbarer Regler, Grenzen |
| `trm_visualization.py`, `trm_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Angriff und Verteidigung getrennt** (Maher, Dixon-Coles): der Generator kennt eine Stärke je Mannschaft.
- **Korrelation niedriger Ergebnisse** (Dixon-Coles) und zeitveränderliche Stärke: eigener Erzeugungsmechanismus; die zeitliche Fortschreibung ist Thema von „Elo“.
- **Unvollständige Pläne:** Lücken und Staffeln behandeln „Massey und Colley“ und „Bradley-Terry“; hier nur vollständige Pläne.
- **Spielerebene:** „Von der Mannschaft zum Einzelnen“.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
