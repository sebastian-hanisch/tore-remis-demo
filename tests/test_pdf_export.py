"""PDF-Export: erzeugt für jede Konfiguration gültige PDF-Bytes; der Latin-1-Filter ersetzt alle Zeichen außerhalb der Kernschrift."""

import trm_evaluation as ev
import trm_pdf_export as X
from trm_scenario import Settings


def test_pdf_is_built_for_every_home_mode_and_for_overdispersion():
    for s in (Settings(), Settings(n=6, home="setzliste", k=3, seed=3), Settings(n=16, home="umgekehrt", k=1, g=0.5, seed=5, fit_h=False),
              Settings(n=4, disp=True, r=0.5, g=20.0, h=0.0, seed=7)):
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("τ ≥ 0,5; ĥ = 1 → θ − σ **fett** € ∞")
    assert out == "Tau >= 0,5; h-Dach = 1 -> theta - sigma fett EUR unendlich"
    assert out.encode("latin-1")
