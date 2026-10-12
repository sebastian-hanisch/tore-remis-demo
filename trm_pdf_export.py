"""PDF-Export der aktuellen Konfiguration und der Schätzung (fpdf2, Kernschrift: nur Zeichen aus Latin-1, daher „Tau“ statt τ und „theta“ statt θ)."""

from fpdf import FPDF

import trm_constants as C
import trm_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("λ", "lambda"), ("θ", "theta"), ("σ", "sigma"), ("ĥ", "h-Dach"), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-"), ("∞", "unendlich")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s, gm, tab = a.settings, a.games, a.table
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Tore, Remis, Heimvorteil: mehr Information als Sieg oder Niederlage"))
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    disp = f", Form r: {ev.de(s.r, 1)}" if s.disp else ", ohne Überstreuung"
    pdf.multi_cell(0, 5.5, _latin(f"Mannschaften: {s.n}, Partien je Paar: {s.k}, Tore je Mannschaft: {ev.de(s.g, 1)}, Heimvorteil: {ev.de(s.h)}, "
                                  f"Heimrecht: {C.HOME_LABELS[s.home]}{disp}, Heimvorteil geschätzt: {'ja' if s.fit_h else 'nein'}, Seed: {s.seed}."))
    pdf.ln(2)
    level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text))
    pdf.ln(3)
    headers = ("Mannschaft", "S", "U", "N", "Tore", "Diff.", "theta Poisson", "wahre Stärke")
    widths = (28, 12, 12, 12, 24, 16, 36, 36)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in sorted(range(s.n), key=lambda j: a.rank["Poisson"][j]):
        row = (gm.names[i], str(int(tab["wins"][i])), str(int(tab["draws"][i])), str(int(tab["losses"][i])), f"{int(tab['gf'][i])}:{int(tab['ga'][i])}",
               str(int(tab["diff"][i])), ev.de(a.poisson.theta[i]), ev.de(gm.strengths[i]))
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Stärken aus dem Generator der Ranking-Linie, Tore als Poisson bzw. negativ binomial); die Stärke ist eine Konstruktion der "
        "Simulation. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."))
    return bytes(pdf.output())
