"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import trm_constants as C
import trm_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_places(a):
    """Platz je Verfahren für jede Mannschaft, in der wahren Reihenfolge (Stärkste links): auf der Diagonale stimmt das Verfahren."""
    gm = a.games
    n = gm.n
    order = list(range(n - 1, -1, -1))
    names = [gm.names[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=names, y=list(range(1, n + 1)), mode="lines", line=dict(color="rgba(150,150,150,0.5)", dash="dash"), name="wahre Ordnung",
                             hoverinfo="skip"))
    symbols = {"Bradley-Terry": "circle-open", "Davidson": "square-open", "Poisson": "circle", "Massey": "diamond-open"}
    sizes = {"Bradley-Terry": 9, "Davidson": 9, "Poisson": 12, "Massey": 7}
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=names, y=[a.rank[m][i] + 1 for i in order], mode="markers", name=m,
                                 marker=dict(size=sizes[m], color=C.METHOD_COLORS[m], symbol=symbols[m], line=dict(width=2)),
                                 hovertemplate="%{x}: Platz %{y} (" + m + ")<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Mannschaft (wahre Reihenfolge, Stärkste links)")
    fig.update_yaxes(title_text="Platz", autorange="reversed", dtick=1)
    return _base(fig, height=380)


def build_pair(probs):
    """1-X-2 für eine Paarung aus Wahrheit und den drei Modellen."""
    cats = ["Heimsieg", "Remis", "Auswärtssieg"]
    colors = {"Wahrheit": C.COLOR_TRUTH, "BT + Remisquote": C.COLOR_BT, "Davidson": C.COLOR_DAVIDSON, "Poisson": C.COLOR_POISSON}
    fig = go.Figure()
    for m in ("Wahrheit", "BT + Remisquote", "Davidson", "Poisson"):
        fig.add_trace(go.Bar(x=cats, y=list(probs[m]), name=m, marker_color=colors[m], hovertemplate=m + ", %{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Wahrscheinlichkeit", tickformat=".0%", range=[0, 1])
    return _base(fig, height=320)


def build_value(rows):
    xs = [str(r["k"]) for r in rows]
    fig = go.Figure()
    for m, dash in (("Poisson", "solid"), ("Massey", "dot"), ("Davidson", "dash"), ("Bradley-Terry", "solid")):
        fig.add_trace(go.Scatter(x=xs, y=[r[m] for r in rows], mode="lines+markers", name=m + (" (= Tordifferenz)" if m == "Massey" else ""),
                                 line=dict(color=C.METHOD_COLORS[m], width=3 if m in ("Poisson", "Bradley-Terry") else 2, dash=dash),
                                 hovertemplate=m + ", %{x} Partien je Paar: τ = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar", type="category")
    fig.update_yaxes(title_text="τ gegen die Generator-Stärke", range=[0.6, 1.0])
    return _base(fig)


def build_level(rows):
    xs = [ev.de(r["g"], 1) for r in rows]
    fig = go.Figure()
    for m in ("Poisson", "Bradley-Terry"):
        fig.add_trace(go.Scatter(x=xs, y=[r[m] for r in rows], mode="lines+markers", name=m, line=dict(color=C.METHOD_COLORS[m], width=3),
                                 hovertemplate=m + ", g = %{x}: τ = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=xs, y=[r["draw"] for r in rows], name="Remis-Anteil", marker_color="rgba(150,150,150,0.35)", yaxis="y2",
                         hovertemplate="g = %{x}: %{y:.0%} Remis<extra></extra>"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1], tickformat=".0%", title="Remis-Anteil", fixedrange=True))
    fig.update_xaxes(title_text="mittlere Tore je Mannschaft", type="category")
    fig.update_yaxes(title_text="τ gegen die Generator-Stärke", range=[0.5, 1.02])
    return _base(fig)


def build_draws(draw_rows):
    """Mehrverlust der 1-X-2-Vorhersage je Remis-Behandlung."""
    models = [r["model"] for r in draw_rows]
    colors = [C.COLOR_BT, C.COLOR_BT, C.COLOR_DAVIDSON, C.COLOR_POISSON]
    fig = go.Figure(go.Bar(x=models, y=[r["kl"] for r in draw_rows], marker_color=colors, hovertemplate="%{x}: %{y:.3f} nat<extra></extra>"))
    fig.update_yaxes(title_text="Mehrverlust gegen die Wahrheit (nat je Partie)", rangemode="tozero")
    return _base(fig, height=320)


def build_predict(rows):
    xs = [str(r["k"]) for r in rows]
    colors = {"BT + Remisquote": C.COLOR_BT, "Davidson": C.COLOR_DAVIDSON, "Poisson": C.COLOR_POISSON}
    fig = go.Figure()
    for m in C.MODEL_LABELS:
        fig.add_trace(go.Scatter(x=xs, y=[r[m] for r in rows], mode="lines+markers", name=m, line=dict(color=colors[m], width=3),
                                 hovertemplate=m + ", %{x} Partien je Paar: %{y:.3f} nat<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar", type="category")
    fig.update_yaxes(title_text="Mehrverlust gegen die Wahrheit (nat je Partie)", rangemode="tozero")
    return _base(fig)


def build_home_tau(rows):
    modes = [r["mode"] for r in rows]
    labels = {"alternierend": "alternierend", "zufaellig": "zufällig", "setzliste": "Setzliste", "umgekehrt": "umgekehrt"}
    fig = go.Figure()
    for key, name, color in (("tau0", "Poisson ohne Heimvorteil", "#9cc5bf"), ("tau1", "Poisson mit Heimvorteil", C.COLOR_POISSON),
                             ("tau_m", "Massey (Tordifferenz)", C.COLOR_MASSEY)):
        fig.add_trace(go.Bar(x=[labels[m] for m in modes], y=[r[key] for r in rows], name=name, marker_color=color,
                             hovertemplate=name + ", %{x}: τ = %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Heimrecht-Verteilung")
    fig.update_yaxes(title_text="τ gegen die Generator-Stärke", range=[0.7, 0.95])
    return _base(fig)


def build_home_hat(rows, h=C.HOME_H):
    modes = [r["mode"] for r in rows]
    labels = {"alternierend": "alternierend", "zufaellig": "zufällig", "setzliste": "Setzliste", "umgekehrt": "umgekehrt"}
    fig = go.Figure(go.Bar(x=[labels[m] for m in modes], y=[r["h_hat"] for r in rows], marker_color=C.COLOR_POISSON, name="geschätzter Heimvorteil",
                           hovertemplate="%{x}: ĥ = %{y:.2f}<extra></extra>"))
    fig.add_hline(y=h, line_dash="dash", line_color="#c0392b", annotation_text=f"wahr: {ev.de(h)}", annotation_position="top right")
    fig.update_xaxes(title_text="Heimrecht-Verteilung")
    fig.update_yaxes(title_text="geschätzter Heimvorteil ĥ", rangemode="tozero")
    return _base(fig, height=320)


def build_dispersion(rows):
    xs = ["∞ (Poisson)" if r["r"] is None else ev.de(r["r"], 1) for r in rows]
    fig = go.Figure()
    for key, name, color in (("kl_po", "Poisson", C.COLOR_POISSON), ("kl_dv", "Davidson", C.COLOR_DAVIDSON)):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=3),
                                 hovertemplate=name + ", Form %{x}: %{y:.3f} nat<extra></extra>"))
    fig.update_xaxes(title_text="Form r der negativen Binomialverteilung (klein = stark überstreut)", type="category")
    fig.update_yaxes(title_text="Mehrverlust gegen die Wahrheit (nat je Partie)", rangemode="tozero")
    return _base(fig)
