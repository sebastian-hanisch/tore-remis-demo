"""Tore, Remis, Heimvorteil: mehr Information als Sieg oder Niederlage - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Ranking-Linie: Ein Poisson-Modell für Tore, ein Davidson-Modell für Remis und ein Heimvorteil gegen Bradley-Terry auf Sieg oder Niederlage;
fünf Experimente messen, was die Tore wert sind und wo die Annahmen enden. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import trm_constants as C
import trm_evaluation as ev
from trm_pdf_export import build_pdf
from trm_presets import (
    apply_preset,
    bounds,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from trm_scenario import Settings
from trm_visualization import (
    build_dispersion,
    build_draws,
    build_home_hat,
    build_home_tau,
    build_level,
    build_pair,
    build_places,
    build_predict,
    build_value,
)

st.set_page_config(page_title="Tore, Remis, Heimvorteil – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _value():
    return ev.sweep_value(), ev.sweep_equivalent()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _level():
    return ev.sweep_level()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _draws():
    return ev.sweep_draws(), ev.sweep_predict()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _home():
    return ev.sweep_home()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _dispersion():
    return ev.sweep_dispersion()


def verdict(level, text):
    SHOW[level](text)


st.title("⚽ Tore, Remis, Heimvorteil: mehr Information als Sieg oder Niederlage")
st.markdown(
    """
Bradley-Terry sieht nur Sieg oder Niederlage. Diese Demo ist das **sechste Stück der Ranking-Linie** und fragt, was ein Modell gewinnt, das auch die **Tore**,
das **Remis** und den **Heimvorteil** liest. Das Ergebnis ist zweigeteilt: Tore sind **gut anderthalb Partien wert**, aber ihre Rangordnung ist im
vollständigen Plan **dieselbe wie die der Tordifferenz** (Massey): Der eigentliche Gewinn liegt in der **1-X-2-Vorhersage**. Das Remis anders zu behandeln
ändert den Rang nicht; der Heimvorteil lässt sich nur trennen, wenn das Heimrecht nicht an der Stärke hängt; und **überstreute** Tore bringen das Poisson-Modell zu
Fall.

Weiter unten: fünf **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So lesen Tore die Stärke", expanded=True):
    st.markdown(
        """
1. **Das Tormodell:** Die Tore jeder Mannschaft sind Poisson-verteilt mit der Rate λ. Die Heimrate ist exp(μ + h + (θ_H − θ_G)/2), die Gastrate
   exp(μ − (θ_H − θ_G)/2): Das **Torverhältnis** ist exp(θ_H − θ_G + h), also dieselbe Skala wie bei Bradley-Terry; μ setzt das Torniveau, h ist der Heimvorteil.
2. **Remis** gibt es von selbst: Fallen gleich viele Tore, ist es unentschieden. Das Davidson-Modell für Sieg-Remis-Niederlage braucht dafür einen eigenen Parameter ν
   und sieht nur den Ausgang, nicht die Tore.
3. **Der Rang:** Im vollständigen, ausgeglichenen Plan hängt die Schätzung nur an der **Tordifferenz** jeder Mannschaft; das ist auch die Ordnung der
   Massey-Wertung. Das Poisson-Modell ordnet also nicht anders, es ordnet **mit Tordifferenz** statt mit Sieg oder Niederlage, und liefert dazu
   Wahrscheinlichkeiten für 1, X und 2.
4. **Was es annimmt:** unabhängige Tore, eine Stärke je Mannschaft, einen festen Heimvorteil. Das Experiment 5 bricht die erste Annahme.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Mannschaften**")
    n = st.slider("Mannschaften n", *bounds("n_slider"), key="n_slider", help="Wie viele Mannschaften im Turnier antreten (jeder gegen jeden).")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Mannschaften (logistische Skala des Torverhältnisses).")
    k = st.slider("Partien je Paar", *bounds("k_slider"), key="k_slider", help="Wie oft ein Paar gegeneinander spielt (Hin- und Rückspiele wechseln das Heimrecht).")
    st.markdown("**Tore**")
    g = st.slider("Mittlere Tore je Mannschaft", *bounds("g_slider"), step=C.G_STEP, key="g_slider",
                  help="Mittlere Torzahl einer Mannschaft bei Gleichstand und ohne Heimvorteil. Klein = torarm, viele Remis.")
    h = st.slider("Heimvorteil h", *bounds("h_slider"), step=C.H_STEP, key="h_slider",
                  help="Log-Faktor der Heim-Torrate: 0,3 heißt rund 35 % mehr Heimtore.")
    home = st.selectbox("Heimrecht-Verteilung", list(C.HOME_LABELS), format_func=C.HOME_LABELS.get, key="home_select",
                        help="Wer wie oft Heimrecht hat. Hängt es an der Stärke, sind Stärke und Heimvorteil schwer zu trennen.")
    disp = st.checkbox("Tore überstreut (negativ binomial)", key="disp_toggle", help="Mehr Streuung als Poisson: Ausreißer-Spiele mit sehr vielen oder keinen Toren.")
    if disp:
        seed_widget("r_slider")
        r = st.slider("Form r der Überstreuung", *bounds("r_slider"), step=C.R_STEP, key="r_slider",
                      help="Klein = stark überstreut, groß = nahe Poisson.")
        remember("r_slider", r)
    st.markdown("**Schätzung**")
    fit_h = st.checkbox("Heimvorteil mitschätzen", key="fit_h_toggle", help="Aus: Alle Modelle nehmen h = 0 an.")
    seed = st.number_input("Zufalls-Seed des Turniers", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Turnier würfeln", width="stretch", on_click=randomize_seed)

r_value = round(float(current_value("r_slider")), 1)
sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "k_slider": int(k), "g_slider": round(float(g), 1), "h_slider": round(float(h), 2),
                   "home_select": home, "disp_toggle": bool(disp), "r_slider": r_value, "fit_h_toggle": bool(fit_h), "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(k), round(float(g), 1), round(float(h), 2), home, bool(disp), r_value, bool(fit_h), int(seed))
a = _analysis(settings)
gm, tab = a.games, a.table

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Auswertung")
m1, m2, m3, m4 = st.columns(4)
m1.metric("τ gegen die Stärke: Poisson", ev.de(a.tau_s["Poisson"]),
          delta=f"BT {ev.de(a.tau_s['Bradley-Terry'])}, Davidson {ev.de(a.tau_s['Davidson'])}, Massey {ev.de(a.tau_s['Massey'])}", delta_color="off",
          help="Kendall-τ der Ordnung gegen die Generator-Stärke. Poisson und Massey ordnen im vollständigen, ausgeglichenen Plan gleich.")
m2.metric("Remis-Anteil", f"{ev.pct1(a.draw_share)} %", delta=f"{ev.de(float((gm.goals_home + gm.goals_away).mean()), 1)} Tore je Partie", delta_color="off")
m3.metric("Heimvorteil geschätzt", ev.de(a.poisson.h) if settings.fit_h else "–", delta=f"wahr: {ev.de(settings.h)}", delta_color="off",
          help="Aus dem Poisson-Modell: Log-Faktor der Heim-Torrate.")
m4.metric("Mehrverlust 1-X-2: Poisson", ev.de(a.kl["Poisson"], 3),
          delta=f"Davidson {ev.de(a.kl['Davidson'], 3)}, BT + Remisquote {ev.de(a.kl['BT + Remisquote'], 3)}", delta_color="off",
          help="Mittlere KL-Divergenz wahr → Modell über alle geordneten Paare (nat je Partie). Kleiner ist besser.")

order = sorted(range(settings.n), key=lambda i: a.rank["Poisson"][i])
df = pd.DataFrame({
    "Mannschaft": [gm.names[i] for i in order],
    "Platz nach Stärke": [settings.n - int(sorted(range(settings.n), key=lambda j: gm.strengths[j]).index(i)) for i in order],
    "S": [int(tab["wins"][i]) for i in order],
    "U": [int(tab["draws"][i]) for i in order],
    "N": [int(tab["losses"][i]) for i in order],
    "Tore": [f"{int(tab['gf'][i])}:{int(tab['ga'][i])}" for i in order],
    "Differenz": [int(tab["diff"][i]) for i in order],
    "θ Poisson": [round(float(a.poisson.theta[i]), 2) for i in order],
    "Platz Bradley-Terry": [a.rank["Bradley-Terry"][i] + 1 for i in order],
    "Platz Poisson": [a.rank["Poisson"][i] + 1 for i in order],
    "wahre Stärke": [round(float(gm.strengths[i]), 2) for i in order],
})
st.dataframe(df, hide_index=True, width="stretch")
st.plotly_chart(build_places(a), width="stretch", key="places_chart")
st.caption("Je Mannschaft der Platz nach jedem Verfahren, in der wahren Reihenfolge; auf der gestrichelten Linie stimmt der Platz. Poisson und Massey liegen im "
           "vollständigen Plan ohne Heimvorteil aufeinander. Die wahre Stärke ist hier bekannt, weil das Turnier simuliert ist.")

st.markdown("**Eine Paarung vorhersagen**")
names = gm.names
st.session_state.setdefault("pair_home", 0)
st.session_state.setdefault("pair_away", 1)
for key, fallback in (("pair_home", 0), ("pair_away", 1)):
    if st.session_state[key] >= settings.n:
        st.session_state[key] = fallback
pc1, pc2 = st.columns(2)
with pc1:
    home_i = st.selectbox("Heimmannschaft", list(range(settings.n)), format_func=lambda i: names[i], key="pair_home")
with pc2:
    away_i = st.selectbox("Gastmannschaft", list(range(settings.n)), format_func=lambda i: names[i], key="pair_away")
if home_i == away_i:
    st.info("Bitte zwei verschiedene Mannschaften wählen.")
else:
    st.plotly_chart(build_pair(ev.pair_probabilities(a, home_i, away_i)), width="stretch", key="pair_chart")
    st.caption("Wahrscheinlichkeit für Heimsieg, Remis und Auswärtssieg aus der wahren Verteilung und den drei Modellen. Bradley-Terry hat nur eine feste "
               "Remisquote und kann nicht abbilden, dass Remis zwischen Gleichstarken häufiger ist.")
st.download_button("📄 PDF der aktuellen Auswertung", data=build_pdf(a), file_name="tore_remis_heimvorteil.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was lesen die Tore, und wo tragen die Annahmen nicht?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Was Tore wert sind")
st.caption("8 Teilnehmer (feste Instanz, unabhängig vom Regler), vollständig, 1,4 Tore je Mannschaft, kein Heimvorteil; Mittel über 300 feste Turniere.")
rows1, equiv = _value()
st.plotly_chart(build_value(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_value(rows1, equiv))
st.dataframe(pd.DataFrame([{"Partien je Paar": r["k"], "Remis-Anteil": f"{ev.pct1(r['draw'])} %",
                            "Poisson = Tordifferenz": f"{r['same_order']} von {r['tie_free']}",
                            "BT = Davidson = Punkte": f"{r['same_points']} von {r['points_tie_free']}"} for r in rows1]), hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"Partien je Paar (Poisson)": e["k"], "τ Poisson": round(e["tau_poisson"], 3),
                            "entspricht Bradley-Terry mit … Partien": round(e["k_bt"], 1), "Faktor": round(e["factor"], 2)} for e in equiv]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Das Torniveau")
st.caption("8 Teilnehmer (feste Instanz), vollständig, 2 Partien je Paar, kein Heimvorteil; Mittel über 300 feste Turniere.")
rows2 = _level()
st.plotly_chart(build_level(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_level(rows2))

st.subheader("🔬 Experiment 3: Remis und die Vorhersage")
st.caption("8 Teilnehmer (feste Instanz), vollständig, 1,4 Tore je Mannschaft. Links 2 Partien je Paar ohne Heimvorteil, rechts nach Partienzahl mit Heimvorteil "
           "0,3 (geschätzt). Mehrverlust gegen die wahre 1-X-2-Verteilung, nat je Partie; 300 feste Turniere.")
draw_rows, predict_rows = _draws()
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Remis behandeln**")
    st.plotly_chart(build_draws(draw_rows), width="stretch", key="exp3_chart_a")
with c2:
    st.markdown("**Nach Partien je Paar**")
    st.plotly_chart(build_predict(predict_rows), width="stretch", key="exp3_chart_b")
verdict(*ev.verdict_draws(draw_rows, predict_rows))
st.dataframe(pd.DataFrame([{"Verfahren": r["model"], "τ gegen die Stärke": round(r["tau"], 3), "Mehrverlust 1-X-2": round(r["kl"], 3)} for r in draw_rows]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Heimvorteil und wer Heimrecht hat")
st.caption("8 Teilnehmer (feste Instanz), vollständig, 2 Partien je Paar, wahrer Heimvorteil 0,3; Poisson ohne und mit geschätztem Heimvorteil; 300 feste Turniere.")
rows4 = _home()
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_home_tau(rows4), width="stretch", key="exp4_chart_a")
with c2:
    st.plotly_chart(build_home_hat(rows4), width="stretch", key="exp4_chart_b")
verdict(*ev.verdict_home(rows4))
labels = {"alternierend": "alternierend", "zufaellig": "zufällig", "setzliste": "Setzliste", "umgekehrt": "umgekehrt"}
st.dataframe(pd.DataFrame([{"Heimrecht": labels[r["mode"]], "Heimspiele der Stärkeren": f"{ev.pct(r['strong_home'])} %", "ĥ": round(r["h_hat"], 2),
                            "Skalenfehler ohne h": round(r["err0"], 2), "mit h": round(r["err1"], 2), "Massey": round(r["err_m"], 2),
                            "Mehrverlust ohne h": round(r["kl0"], 3), "mit h": round(r["kl1"], 3)} for r in rows4]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 5: Überstreuung")
st.caption("8 Teilnehmer (feste Instanz), vollständig, 4 Partien je Paar, kein Heimvorteil; Tore negativ binomial mit Form r, Wahrheit exakt; 300 feste Turniere.")
rows5 = _dispersion()
st.plotly_chart(build_dispersion(rows5), width="stretch", key="exp5_chart")
verdict(*ev.verdict_dispersion(rows5))
st.dataframe(pd.DataFrame([{"Form r": "∞ (Poisson)" if r["r"] is None else ev.de(r["r"], 1), "τ Bradley-Terry": round(r["tau_bt"], 3), "τ Poisson": round(r["tau_po"], 3),
                            "Mehrverlust Davidson": round(r["kl_dv"], 3), "Mehrverlust Poisson": round(r["kl_po"], 3)} for r in rows5]),
             hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Tore sind unabhängig und Poisson-verteilt** | Überstreute Tore machen das Poisson-Modell überheblich: ab Form 2 sagt Davidson besser voraus (Experiment 5). Die Korrelation niedriger Ergebnisse (Dixon-Coles) ist nicht modelliert. | – |
| **Der Rang kommt aus der Tordifferenz** | Im vollständigen, ausgeglichenen Plan ist die Poisson-Ordnung die der Tordifferenz, also die von Massey; der Gewinn liegt in der Vorhersage (Experiment 1). | Massey und Colley |
| **Der Heimvorteil ist konstant und unabhängig von der Stärke** | Hängt das Heimrecht an der Stärke, weicht ĥ vom wahren Wert ab und der Rang wird nicht besser (Experiment 4). | – |
| **Eine Stärke je Mannschaft** (Angriff gleich Verteidigung) | Angriffs- und Abwehrstärke fallen auf eine Zahl zusammen; getrennte Parameter (Maher) sind nicht umgesetzt. | – |
| **Die Stärke ist konstant** | Verändert sie sich über die Saison, folgt ihr nur ein fortlaufendes Verfahren. | Elo |
| **Eine Partie ist Mannschaft gegen Mannschaft** | Zerlegt man Mannschaftsergebnisse auf Spieler, ist das eine eigene Aufgabe. | Von der Mannschaft zum Einzelnen |
| **Alles ist simuliert** | Die Stärke, das Torniveau, der Heimvorteil und die Überstreuung sind Konstruktionen des Generators; ob echte Torzahlen so streuen, zeigt die Demo nicht. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Tormodell.** Für die Partie $H$ gegen $G$ sind die Tore $Y_H \sim \mathrm{Poisson}(\lambda_H)$ und $Y_G \sim \mathrm{Poisson}(\lambda_G)$ unabhängig, mit
$\log \lambda_H = \mu + h + (\theta_H - \theta_G)/2$ und $\log \lambda_G = \mu - (\theta_H - \theta_G)/2$. Das Verhältnis $\lambda_H/\lambda_G = e^{\theta_H - \theta_G + h}$.
Im Generator ist $\mu = \ln g - h/2$ und $\theta$ die wahre Stärke $s$.

**Schätzung.** Log-Likelihood $\sum \bigl(y \log\lambda - \lambda\bigr) - \tfrac{\lambda_0}{2}\sum_i \theta_i^2$ mit der Vorannahme $\lambda_0$ (Orakel-Wert
$12/(\delta^2(n^2-1))$, nur in der Simulation bekannt); konkav, gelöst per Newton-Verfahren mit Schrittverkürzung und zentriertem Schritt.

**Rang und Tordifferenz.** Die Stationaritätsgleichung für $\theta_i$ lautet $\tfrac12\bigl(\mathrm{GD}_i - \sum_{\text{Partien}} (\lambda_i - \lambda_{\text{Gegner}})\bigr)
= \lambda_0\theta_i$ mit der Tordifferenz $\mathrm{GD}_i$. Im vollständigen Plan mit gleich vielen Heim- und Auswärtsspielen je Paar (ohne $h$) ist
$\lambda_i - \lambda_j = 2e^{\mu}\sinh((\theta_i - \theta_j)/2)$ streng monoton in $\theta_i - \theta_j$; die Lösung ist damit streng monoton in der Tordifferenz,
und bei Gleichstand der Tordifferenz gleich. Im ausgeglichenen Plan stimmt $\hat h = \ln(\text{Heimtore}/\text{Gasttore})$.

**Davidson.** $P(\text{Heim}) \propto \pi_H e^h$, $P(\text{Gast}) \propto \pi_G$, $P(\text{Remis}) \propto \nu\sqrt{\pi_H e^h\,\pi_G}$ mit $\pi = e^{\theta}$. Im
vollständigen, ausgeglichenen Plan ohne $h$ hängt die Schätzung nur an den Punkten (zwei je Sieg, einer je Remis).

**1-X-2.** $P(\text{Heim}) = \sum_x P(Y_H = x)\,P(Y_G < x)$, $P(\text{Remis}) = \sum_x P(Y_H = x)P(Y_G = x)$, bei Poisson und negativ binomial exakt
(Faltung, bei Mittel plus zehn Standardabweichungen abgeschnitten). **Mehrverlust:** mittlere KL-Divergenz $\sum_c p_c \log(p_c/q_c)$ von der wahren
Verteilung $p$ zur Modellverteilung $q$ über alle geordneten Paare.

**Überstreuung.** Negativ binomial mit Mittel $\lambda$ und Form $r$: Varianz $\lambda + \lambda^2/r$ (Gamma-Mischung der Poisson-Rate); $r \to \infty$ ist Poisson.

**Zufall.** Die Torzahl ist die kleinste ganze Zahl $k$ mit $F(k) \ge u$ für eine Gleichverteilte $u$ (SplitMix64): monoton in der Rate.

**Literatur.**
- Maher, M. J. (1982): Modelling association football scores. *Statistica Neerlandica* 36 (3), 109–118, doi:10.1111/j.1467-9574.1982.tb00782.x (unabhängiges
  Poisson-Modell mit Angriffs- und Abwehrstärke; nach dem Abstract beschreibt es Fußballergebnisse angemessen, mit kleinen systematischen Abweichungen; das
  Original wurde nicht eingesehen).
- Dixon, M. J., Coles, S. G. (1997): Modelling association football scores and inefficiencies in the football betting market. *Applied Statistics (Journal of the
  Royal Statistical Society, Series C)* 46 (2), 265–280 (Poisson-Regression mit Korrektur für niedrige Ergebnisse und zeitveränderlicher Stärke; hier nicht
  umgesetzt; nach Abstract und Sekundärquellen, das Original wurde nicht eingesehen).
- Davidson, R. R. (1970): On extending the Bradley-Terry model to accommodate ties in paired comparison experiments. *Journal of the American Statistical
  Association* 65, 317–328 (Remis proportional zum geometrischen Mittel der Stärken; nach Sekundärquellen, das Original wurde nicht eingesehen).
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39 (3/4), 324–345.
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
