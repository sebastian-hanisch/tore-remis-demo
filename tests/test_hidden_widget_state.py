"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def widget_keys(at):
    return {s.key for s in at.slider} | {s.key for s in at.selectbox}


def test_the_shape_slider_is_only_drawn_with_overdispersion():
    at = fresh()
    assert "r_slider" not in widget_keys(at)
    at.checkbox(key="disp_toggle").check().run(timeout=TIMEOUT)
    assert "r_slider" in widget_keys(at)
    at.checkbox(key="disp_toggle").uncheck().run(timeout=TIMEOUT)
    assert "r_slider" not in widget_keys(at) and not at.exception


def test_hidden_value_survives_and_is_not_written_into_the_hidden_widget():
    at = fresh()
    at.checkbox(key="disp_toggle").check().run(timeout=TIMEOUT)
    at.slider(key="r_slider").set_value(5.0).run(timeout=TIMEOUT)
    at.checkbox(key="disp_toggle").uncheck().run(timeout=TIMEOUT)
    assert "r_slider" not in at.session_state
    at.checkbox(key="disp_toggle").check().run(timeout=TIMEOUT)
    assert at.slider(key="r_slider").value == 5.0
    assert not at.exception


def test_preset_sets_values_and_the_hidden_widget_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Überstreute Tore"]].click().run(timeout=TIMEOUT)
    assert at.checkbox(key="disp_toggle").value is True and at.slider(key="r_slider").value == 2.0 and at.slider(key="k_slider").value == 4
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Heimrecht nach Setzliste"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="home_select").value == "setzliste" and at.checkbox(key="disp_toggle").value is False
    assert "r_slider" not in at.session_state
    assert not at.exception


def test_pair_selection_survives_a_smaller_team_count():
    at = fresh()
    at.selectbox(key="pair_home").select(7).run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(4).run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="pair_home").value < 4
