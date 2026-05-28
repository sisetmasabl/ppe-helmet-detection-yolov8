"""
Genera report/REPORT.pdf a partir de report/REPORT.md (sin LaTeX/pandoc).

Renderizador Markdown ligero basado en fpdf2 (Python puro):
  soporta # ## ###, párrafos, listas con '-', **negrita**, tablas Markdown
  e imágenes con la sintaxis ![alt](ruta).

Uso:
    pip install fpdf2
    python report/build_pdf.py
"""
import re
from pathlib import Path

from fpdf import FPDF

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MD = HERE / "REPORT.md"
PDF = HERE / "REPORT.pdf"

FONT_DIR = Path(r"C:\Windows\Fonts")
REPLACEMENTS = {
    "≥": ">=", "≤": "<=", "↔": "<->", "→": "->", "←": "<-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "•": "-",
}


def sanitize(s: str) -> str:
    for a, b in REPLACEMENTS.items():
        s = s.replace(a, b)
    # El renderizador markdown de fpdf2 solo interpreta **negrita**, no *cursiva*.
    # Quitamos los asteriscos de cursiva (un solo *) preservando los de negrita.
    s = s.replace("**", "\x00")
    s = s.replace("*", "")
    s = s.replace("\x00", "**")
    return s


class Report(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("rep", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def resolve_image(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (ROOT / path_str).resolve()
    return p


def add_table(pdf: Report, rows):
    if not rows:
        return
    ncol = len(rows[0])
    avail = pdf.w - 2 * pdf.l_margin
    cw = avail / ncol
    pdf.set_font("rep", "", 9)
    line_h = 6
    for ri, row in enumerate(rows):
        style = "B" if ri == 0 else ""
        pdf.set_font("rep", style, 9)
        if ri == 0:
            pdf.set_fill_color(230, 230, 235)
        else:
            pdf.set_fill_color(248, 248, 250)
        for cell in row:
            txt = sanitize(cell.strip())
            pdf.cell(cw, line_h, txt[:40], border=1, fill=True, align="C")
        pdf.ln(line_h)
    pdf.ln(2)


def main():
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Fuentes Unicode desde Windows
    pdf.add_font("rep", "", str(FONT_DIR / "arial.ttf"))
    pdf.add_font("rep", "B", str(FONT_DIR / "arialbd.ttf"))
    pdf.add_font("rep", "I", str(FONT_DIR / "ariali.ttf"))
    pdf.add_page()

    text = MD.read_text(encoding="utf-8")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Tabla Markdown
        if stripped.startswith("|") and "|" in stripped[1:]:
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:\-]+$", "".join(row)):  # saltar separador
                    tbl.append(row)
                i += 1
            add_table(pdf, tbl)
            continue

        # Imagen
        m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if m:
            img = resolve_image(m.group(2))
            if img.exists():
                avail = pdf.w - 2 * pdf.l_margin
                w = min(avail, 150)
                x = (pdf.w - w) / 2
                try:
                    pdf.image(str(img), x=x, w=w)
                except Exception:
                    pass
                if m.group(1):
                    pdf.set_font("rep", "I", 8)
                    pdf.set_text_color(110, 110, 110)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 5, sanitize(m.group(1)), align="C")
                    pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            i += 1
            continue

        # Asegura ancho completo: algunas llamadas previas a multi_cell (markdown)
        # dejan el cursor en el margen derecho; lo reseteamos al margen izquierdo.
        pdf.set_x(pdf.l_margin)
        if stripped.startswith("# "):
            pdf.set_font("rep", "B", 16)
            pdf.multi_cell(0, 8, sanitize(stripped[2:]))
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.ln(1)
            pdf.set_font("rep", "B", 13)
            pdf.multi_cell(0, 7, sanitize(stripped[3:]))
            pdf.ln(1)
        elif stripped.startswith("### "):
            pdf.set_font("rep", "B", 11)
            pdf.multi_cell(0, 6, sanitize(stripped[4:]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.set_font("rep", "", 10)
            pdf.multi_cell(0, 5.5, sanitize("  -  " + stripped[2:]), markdown=True)
        elif stripped == "---" or stripped == "":
            pdf.ln(2)
        else:
            pdf.set_font("rep", "", 10)
            pdf.multi_cell(0, 5.5, sanitize(stripped), markdown=True)
        i += 1

    pdf.output(str(PDF))
    print(f"PDF generado: {PDF}")


if __name__ == "__main__":
    main()
