"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt, was seine Beschreibung verspricht."""

import dataclasses

import trm_constants as C
import trm_evaluation as ev
import trm_presets as P
from trm_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.DELTA_MIN <= p["delta"] <= C.DELTA_MAX and C.K_MIN <= p["k"] <= C.K_MAX, name
        assert C.G_MIN <= p["g"] <= C.G_MAX and C.H_MIN <= p["h"] <= C.H_MAX and C.R_MIN <= p["r"] <= C.R_MAX, name
        assert p["home"] in C.HOME_LABELS and isinstance(p["disp"], bool) and isinstance(p["fit_h"], bool), name
        assert 0 <= p["seed"] <= C.SEED_MAX, name


def test_first_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Fußball-Niveau"]) == Settings()


def test_setting_specs_cover_every_preset_key_and_settings_fields_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(Settings)}
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_kept_keys_are_exactly_the_hideable_widgets():
    assert set(P.KEPT_KEYS) == {"r_slider"}


def test_permalink_bool_parsing():
    assert P._to_bool("1") and P._to_bool("True") and not P._to_bool("0") and not P._to_bool("False")


def _mean(preset, attr, seeds=range(100000, 100060)):
    vals = [attr(ev.analyse(Settings(**{**C.PRESETS[preset], "seed": s}))) for s in seeds]
    return sum(vals) / len(vals)


def test_each_preset_tells_its_story():
    foot = "Fußball-Niveau"
    assert _mean(foot, lambda a: a.kl["Poisson"]) < _mean(foot, lambda a: a.kl["Davidson"]) < _mean(foot, lambda a: a.kl["BT + Remisquote"])
    low = "Torarmes Spiel"
    assert _mean(low, lambda a: a.draw_share) > 0.35
    seat = "Heimrecht nach Setzliste"
    assert _mean(seat, lambda a: a.poisson.h) > C.DEFAULT_H + 0.05                    # ĥ liegt über dem wahren 0,3
    disp = "Überstreute Tore"
    assert _mean(disp, lambda a: a.kl["Poisson"]) > _mean(disp, lambda a: a.kl["Davidson"])
