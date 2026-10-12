"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Der ausblendbare Regler (Form r nur bei überstreuten Toren) folgt dem KEPT-Muster
(DEMO-PLAYBOOK §3): der Wert eines nicht gezeichneten Reglers wird nie in sein Widget geschrieben, sondern in KEPT abgelegt und unmittelbar vor dem Zeichnen
per `seed_widget` eingesetzt."""

import random
from dataclasses import dataclass

import streamlit as st

import trm_constants as C


def _to_bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "ja", "yes")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "delta_slider": SettingSpec("d", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "g_slider": SettingSpec("g", float, C.DEFAULT_G, C.G_MIN, C.G_MAX),
    "h_slider": SettingSpec("h", float, C.DEFAULT_H, C.H_MIN, C.H_MAX),
    "home_select": SettingSpec("home", str, "alternierend", choices=tuple(C.HOME_LABELS)),
    "disp_toggle": SettingSpec("disp", _to_bool, False, choices=(True, False)),
    "r_slider": SettingSpec("r", float, C.DEFAULT_R, C.R_MIN, C.R_MAX),
    "fit_h_toggle": SettingSpec("fith", _to_bool, True, choices=(True, False)),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "delta": "delta_slider", "k": "k_slider", "g": "g_slider", "h": "h_slider", "home": "home_select", "disp": "disp_toggle",
               "r": "r_slider", "fit_h": "fit_h_toggle", "seed": "seed_input"}
KEPT_KEYS = ("r_slider",)


def _kept() -> dict:
    return st.session_state.setdefault("kept_values", {})


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key in KEPT_KEYS:
            _kept().setdefault(state_key, spec.default)
        elif state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines ausblendbaren Reglers: seinen zuletzt gewählten Wert einsetzen."""
    if state_key not in st.session_state:
        st.session_state[state_key] = _kept().get(state_key, SETTING_SPECS[state_key].default)


def remember(state_key, value):
    _kept()[state_key] = value


def current_value(state_key):
    """Aktueller Wert, auch wenn der Regler in diesem Lauf nicht gezeichnet wird (dann der gemerkte)."""
    if state_key in KEPT_KEYS:
        return st.session_state.get(state_key, _kept().get(state_key, SETTING_SPECS[state_key].default))
    return st.session_state.get(state_key, SETTING_SPECS[state_key].default)


def _store(state_key, value):
    if state_key in KEPT_KEYS:
        _kept()[state_key] = value
        st.session_state.pop(state_key, None)    # beim nächsten Zeichnen aus KEPT einsetzen
    else:
        st.session_state[state_key] = value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        _store(state_key, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value)) if isinstance(value, bool) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            _store(state_key, C.PRESETS[name][key])


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
