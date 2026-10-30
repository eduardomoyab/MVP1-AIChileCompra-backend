"""
services/pdf_service.py

PDF técnico del requerimiento de compra ("Especificaciones técnicas"),
pensado para adjuntarse al requerimiento que se publica en ChileCompra.

Solo lleva lo técnico (qué se compra, cuánto, con qué características): los
precios, las comparaciones y los productos revisados NO van en el
documento -- viven en el resumen ejecutivo de la plataforma.

Antes esto se armaba en el navegador con html2pdf.js (una "foto" de la
página dentro de un PDF: texto no seleccionable, borroso, cortes de página
torpes). Acá se genera con ReportLab: texto real, vectorial, liviano y con
paginación controlada. El diseño sigue la maqueta aprobada (A4, Inter,
azul marino institucional); las medidas están escritas en px de la maqueta
y se convierten a puntos con PX.
"""

import datetime
import io
import os
from typing import Any, Dict, List, Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, NextPageTemplate,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from xml.sax.saxutils import escape

PX = 0.75  # 1 px de la maqueta = 0.75 pt

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_FONTS = os.path.join(_ASSETS, "fonts")
_LOGOS = os.path.join(_ASSETS, "logos")

# Paleta de la maqueta
NAVY = HexColor("#0a2d5e")
BLUE = HexColor("#154f96")
ACCENT = HexColor("#1e6cc5")
LABEL_BLUE = HexColor("#0f3d78")
INK = HexColor("#0f172a")
BODY = HexColor("#1e293b")
MUTED = HexColor("#475569")
FAINT = HexColor("#64748b")
RULE = HexColor("#d9e0ea")
RULE_SOFT = HexColor("#e3e8ef")
TINT = HexColor("#f6f9fd")

F_REG, F_SEMI, F_BOLD = "Inter", "Inter-SemiBold", "Inter-Bold"

_MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

_fonts_ready = False


def _register_fonts() -> None:
    global _fonts_ready
    if _fonts_ready:
        return
    pdfmetrics.registerFont(TTFont(F_REG, os.path.join(_FONTS, "Inter-Regular.ttf")))
    pdfmetrics.registerFont(TTFont(F_SEMI, os.path.join(_FONTS, "Inter-SemiBold.ttf")))
    pdfmetrics.registerFont(TTFont(F_BOLD, os.path.join(_FONTS, "Inter-Bold.ttf")))
    pdfmetrics.registerFontFamily(F_REG, normal=F_REG, bold=F_BOLD, italic=F_REG, boldItalic=F_BOLD)
    _fonts_ready = True


# ─── Estilos ────────────────────────────────────────────────────────────────

def _style(name: str, font: str = F_REG, size_px: float = 13, color=INK, leading_px: Optional[float] = None, **kw) -> ParagraphStyle:
    return ParagraphStyle(
        name, fontName=font, fontSize=size_px * PX, textColor=color,
        leading=(leading_px or size_px * 1.4) * PX, **kw,
    )


S_H1 = _style("h1", F_SEMI, 26, NAVY, 30)
S_TITLE = _style("title", F_SEMI, 18, NAVY, 23)
S_NUM = _style("num", F_SEMI, 26, ACCENT, 30)
S_BODY = _style("body", F_REG, 13, BODY, 20)
S_LABEL = _style("lbl", F_REG, 12, MUTED, 15)
S_VALUE = _style("val", F_SEMI, 13, INK, 17, alignment=TA_RIGHT)
S_NOTE = _style("note", F_REG, 12, MUTED, 18)
S_META_VAL = _style("metaval", F_SEMI, 14, INK, 18)
S_QTY = _style("qty", F_REG, 12, MUTED, 16)


class Tracked(Flowable):
    """Texto en mayúsculas con espaciado entre letras (las etiquetas de la
    maqueta). ReportLab no trae letter-spacing en Paragraph, así que se
    dibuja directo en el canvas."""

    def __init__(self, text: str, font: str = F_SEMI, size_px: float = 12, color=MUTED, tracking_em: float = 0.08, upper: bool = True):
        super().__init__()
        self.text = text.upper() if upper else text
        self.font = font
        self.size = size_px * PX
        self.color = color
        self.tracking = tracking_em * self.size

    def wrap(self, aW, aH):
        self.width = stringWidth(self.text, self.font, self.size) + self.tracking * len(self.text)
        self.height = self.size * 1.25
        return min(self.width, aW), self.height

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        t = c.beginText(0, self.size * 0.28)
        t.setFont(self.font, self.size)
        t.setCharSpace(self.tracking)
        t.textOut(self.text)
        c.drawText(t)


class NumberedCanvas(rl_canvas.Canvas):
    """Canvas en dos pasadas para poder escribir "Página x de y"."""

    def __init__(self, *args, footer_left: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[dict] = []
        self.footer_left = footer_left

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, total: int):
        w, _ = A4
        left, right = 48 * PX, w - 48 * PX
        y_line = 42 * PX
        self.setStrokeColor(RULE)
        self.setLineWidth(0.75)
        self.line(left, y_line, right, y_line)
        self.setFont(F_REG, 12 * PX)
        self.setFillColor(FAINT)
        self.drawString(left, y_line - 22 * PX, self.footer_left)
        self.setFont(F_SEMI, 12 * PX)
        self.setFillColor(LABEL_BLUE)
        self.drawRightString(right, y_line - 22 * PX, f"Página {self._pageNumber} de {total}")


# ─── Formato de valores ─────────────────────────────────────────────────────

def _num(n: Any) -> str:
    try:
        f = float(n)
    except (TypeError, ValueError):
        return str(n)
    if f == int(f):
        return str(int(f))
    return f"{f:g}".replace(".", ",")


def _with_unit(n: Any, unit: Optional[str]) -> str:
    if unit == "GB":
        try:
            f = float(n)
            if f >= 1000 and f % 1000 == 0:
                return f"{int(f // 1000)} TB"
        except (TypeError, ValueError):
            pass
        return f"{_num(n)} GB"
    if unit == '"':
        return f"{_num(n)} pulgadas"
    return f"{_num(n)} {unit}".strip() if unit else _num(n)


def format_value(attr, value: Any) -> str:
    """Valor de un atributo tal como se muestra en la tabla del PDF."""
    unit = getattr(attr, "unit", None)
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, dict) and ("min" in value or "max" in value):
        lo, hi = value.get("min"), value.get("max")
        if lo is not None and hi is not None:
            return f"{_with_unit(lo, unit)} a {_with_unit(hi, unit)}"
        if lo is not None:
            return f"{_with_unit(lo, unit)} o más"
        return f"hasta {_with_unit(hi, unit)}"
    if isinstance(value, (list, tuple)):
        parts = [format_value(attr, v) for v in value if v not in (None, "")]
        # Computadores: varias opciones son alternativas ("o"); en Medicamentos
        # varios principios activos son un producto combinado ("+").
        return getattr(attr, "joiner", " o ").join(parts)
    if isinstance(value, (int, float)):
        return _with_unit(value, unit)
    return str(value)


def _has_value(v: Any) -> bool:
    return v is not None and v != "" and v != [] and v != {}


def _present_attrs(schema, atributos: Dict[str, Any]) -> List[tuple]:
    """(AttributeSchema, valor) de lo que el comprador definió, en el orden
    del esquema."""
    out = []
    for key, attr in schema.attributes.items():
        v = (atributos or {}).get(key)
        if _has_value(v):
            out.append((attr, v))
    return out


def _free_labels(schema, atributos: Dict[str, Any]) -> List[str]:
    """Atributos del esquema que quedaron sin exigencia (abiertos a
    propósito: el proveedor puede ofrecer cualquiera)."""
    atributos = atributos or {}
    sin_gpu = atributos.get("tiene_gpu_dedicada") is False
    labels = []
    for key, attr in schema.attributes.items():
        if _has_value(atributos.get(key)):
            continue
        if key == "gpu_dedicada_nombre" and sin_gpu:
            continue  # "sin GPU dedicada" ya responde esto
        label = attr.label.replace("¿", "").replace("?", "").strip()
        labels.append(_lower_first(label))
    return labels


def _lower_first(label: str) -> str:
    """Minúscula inicial para insertar la etiqueta en una frase, salvo siglas
    y nombres propios ("GPU dedicada", "Wi-Fi")."""
    if len(label) > 1 and (label[1].isupper() or label.startswith("Wi-")):
        return label
    return label[:1].lower() + label[1:]


def _join_es(items: List[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def fallback_title(schema, atributos: Dict[str, Any]) -> str:
    """Título del ítem cuando no hay nombre generado por el modelo: tipo de
    equipo + características principales."""
    a = atributos or {}
    parts = []
    by_key = schema.attributes
    title_attrs = getattr(schema, "title_attrs", None)
    if title_attrs:  # categorías que declaran qué atributos forman el título (ej. Medicamentos)
        for key in title_attrs:
            attr, v = by_key.get(key), a.get(key)
            if attr is not None and _has_value(v):
                parts.append(format_value(attr, v))
        return " · ".join(parts) or "Producto requerido"
    for key in ("tipo_equipo", "procesador_principal", "linea_procesador", "total_ram_gb", "total_almacenamiento_gb"):
        if key in ("procesador_principal", "linea_procesador") and parts and any("Intel" in p or "AMD" in p or "Apple" in p for p in parts):
            continue
        attr = by_key.get(key)
        v = a.get(key)
        if attr is None or not _has_value(v):
            continue
        txt = format_value(attr, v)
        if key == "total_ram_gb":
            txt += " RAM"
        parts.append(txt)
    return " · ".join(parts) or "Producto requerido"


# ─── Piezas del documento ───────────────────────────────────────────────────

def _logo(name: str, height_px: float):
    from reportlab.lib.utils import ImageReader
    path = os.path.join(_LOGOS, name)
    img = ImageReader(path)
    iw, ih = img.getSize()
    h = height_px * PX
    return img, h * iw / ih, h


def _meta_strip(fecha: str, referencia: str, categoria: str, n_items: int, width: float) -> Table:
    cells = [("Fecha de emisión", fecha), ("Referencia", referencia), ("Categoría", categoria), ("Compras", str(n_items))]
    # Columnas según el largo real de cada dato (la fecha es la más larga) y
    # con respiro a ambos lados del divisor -- con 4 columnas iguales el
    # valor quedaba pegado a la línea.
    col_w = [width * 0.35, width * 0.26, width * 0.24, width * 0.15]
    row = []
    for i, (label, value) in enumerate(cells):
        row.append([Tracked(label, F_REG, 12, MUTED, 0.06), Spacer(1, 5 * PX), Paragraph(escape(value), S_META_VAL)])
    t = Table([row], colWidths=col_w)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 13 * PX),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13 * PX),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (-1, -1), 18 * PX),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14 * PX),
        ("LINEABOVE", (0, 0), (-1, 0), 0.75, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, RULE),
        ("LINEBEFORE", (1, 0), (-1, 0), 0.75, RULE),
    ]))
    return t


def _title_row(num: str, title: str, qty: Optional[int], width: float) -> Table:
    qty_w = 0
    qty_cell: Any = ""
    if qty:
        qty_txt = f"Cantidad&nbsp; <font name='{F_BOLD}' size='{14 * PX}' color='#0f172a'>{escape(_fmt_int(qty))}</font>"
        qty_para = Paragraph(qty_txt, S_QTY)
        qw = stringWidth("Cantidad  " + _fmt_int(qty), F_BOLD, 14 * PX) + 26 * PX
        qty_w = max(qw, 78 * PX)
        pill = Table([[qty_para]], colWidths=[qty_w])
        pill.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.75, RULE),
            ("ROUNDEDCORNERS", [4.5, 4.5, 4.5, 4.5]),
            ("TOPPADDING", (0, 0), (-1, -1), 5 * PX),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6 * PX),
            ("LEFTPADDING", (0, 0), (-1, -1), 12 * PX),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12 * PX),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        qty_cell = pill
    num_w = 44 * PX
    gap = 12 * PX
    t = Table(
        [[Paragraph(escape(num), S_NUM), Paragraph(escape(title), S_TITLE), qty_cell]],
        colWidths=[num_w, width - num_w - qty_w - (gap if qty_w else 0), qty_w + (gap if qty_w else 0)],
    )
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (2, 0), (2, 0), gap if qty_w else 0),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
    ]))
    return t


def _fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(n)


def _description_box(text: str, width: float) -> Table:
    t = Table([[[Tracked("Descripción del producto requerido", F_SEMI, 12, LABEL_BLUE, 0.08),
                 Spacer(1, 6 * PX),
                 Paragraph(escape(text), S_BODY)]]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT),
        ("BOX", (0, 0), (-1, -1), 0.75, RULE),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
        ("LEFTPADDING", (0, 0), (-1, -1), 18 * PX), ("RIGHTPADDING", (0, 0), (-1, -1), 18 * PX),
        ("TOPPADDING", (0, 0), (-1, -1), 14 * PX), ("BOTTOMPADDING", (0, 0), (-1, -1), 14 * PX),
    ]))
    return t


def _specs_table(rows: List[tuple], width: float) -> Table:
    """Tabla de dos columnas de pares etiqueta/valor (como la maqueta)."""
    half = width / 2
    label_w = half * 0.46
    value_w = half - label_w
    data = []
    for i in range(0, len(rows), 2):
        pair = rows[i:i + 2]
        line = []
        for j in range(2):
            if j < len(pair):
                label, value = pair[j]
                line += [Paragraph(escape(label), S_LABEL), Paragraph(escape(value), S_VALUE)]
            else:
                line += ["", ""]
        data.append(line)
    t = Table(data, colWidths=[label_w, value_w, label_w, value_w])
    n = len(data)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 9 * PX), ("BOTTOMPADDING", (0, 0), (-1, -1), 10 * PX),
        ("LEFTPADDING", (0, 0), (-1, -1), 16 * PX), ("RIGHTPADDING", (0, 0), (-1, -1), 16 * PX),
        ("LEFTPADDING", (1, 0), (1, -1), 4), ("LEFTPADDING", (3, 0), (3, -1), 4),
        ("RIGHTPADDING", (0, 0), (0, -1), 4), ("RIGHTPADDING", (2, 0), (2, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.75, RULE),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
        ("LINEBEFORE", (2, 0), (2, -1), 0.75, RULE_SOFT),
    ]
    for r in range(n - 1):
        style.append(("LINEBELOW", (0, r), (-1, r), 0.75, RULE_SOFT))
    t.setStyle(TableStyle(style))
    return t


def _item_block(schema, item: Dict[str, Any], num: str, width: float) -> KeepTogether:
    atributos = item.get("atributos") or {}
    title = (item.get("nombre") or "").strip() or fallback_title(schema, atributos)
    qty = item.get("cantidad")
    try:
        qty = int(qty) if qty not in (None, "") and int(qty) > 0 else None
    except (TypeError, ValueError):
        qty = None

    flow: List[Any] = [_title_row(num, title, qty, width), Spacer(1, 14 * PX)]
    desc = (item.get("descripcion") or "").strip()
    if desc:
        flow += [_description_box(desc, width), Spacer(1, 14 * PX)]

    rows = [(attr.label, format_value(attr, v)) for attr, v in _present_attrs(schema, atributos)]
    if rows:
        flow += [Tracked("Características técnicas requeridas", F_SEMI, 12, MUTED, 0.08), Spacer(1, 8 * PX),
                 _specs_table(rows, width)]
    free = _free_labels(schema, atributos)
    if free and rows:
        flow += [Spacer(1, 8 * PX),
                 Paragraph(f"<font name='{F_SEMI}' color='#0f172a'>Sin exigencia:</font> {escape(_join_es(free))}.", S_NOTE)]
    return KeepTogether(flow)


# ─── Documento ──────────────────────────────────────────────────────────────

def _fecha_es(d: datetime.date) -> str:
    return f"{d.day} de {_MESES[d.month - 1]} de {d.year}"


def build_requirements_pdf(
    schema,
    items: List[Dict[str, Any]],
    session_id: str = "",
    fecha: Optional[datetime.date] = None,
) -> bytes:
    """items: [{nombre?, texto_original?, atributos: {attr: valor}, descripcion?, cantidad?}]
    Devuelve los bytes del PDF."""
    _register_fonts()
    fecha = fecha or datetime.date.today()
    referencia = "REQ-" + (session_id.replace("-", "")[:6].upper() or "000000")
    w, h = A4
    margin = 48 * PX
    content_w = w - 2 * margin
    footer_left = "Documento generado por el Asistente IA · Compras Públicas"

    first_top = 84 * PX + 3 * PX  # franja de logos + línea azul marino
    later_top = 64 * PX + 24 * PX  # encabezado corrido + respiro
    bottom = 62 * PX

    def draw_first(c, doc):
        c.saveState()
        # Logos
        img, lw, lh = _logo("logo-chilecompra.png", 34)
        c.drawImage(img, margin, h - 42 * PX - lh / 2, lw, lh, mask="auto")
        # Logos de la derecha, de derecha a izquierda, con separadores finos
        x = w - margin
        cy = h - 42 * PX
        for i, (name, hpx) in enumerate((("logo-uch2.png", 30), ("logo-UCBerkeley.png", 22), ("logo-OCP.png", 24))):
            img, lw, lh = _logo(name, hpx)
            x -= lw
            c.drawImage(img, x, cy - lh / 2, lw, lh, mask="auto")
            if i < 2:
                x -= 9 * PX
                c.setStrokeColor(RULE)
                c.setLineWidth(0.75)
                c.line(x, cy - 11 * PX, x, cy + 11 * PX)
                x -= 9 * PX
            else:
                x -= 18 * PX
        # Línea azul marino (borde superior del encabezado del documento)
        c.setFillColor(NAVY)
        c.rect(margin, h - 84 * PX - 3 * PX, content_w, 3 * PX, stroke=0, fill=1)
        c.restoreState()

    def draw_later(c, doc):
        c.saveState()
        img, lw, lh = _logo("logo-chilecompra.png", 26)
        c.drawImage(img, margin, h - 32 * PX - lh / 2, lw, lh, mask="auto")
        c.setFont(F_REG, 12 * PX)
        c.setFillColor(MUTED)
        ref_w = stringWidth(referencia, F_SEMI, 12 * PX)
        c.setFont(F_SEMI, 12 * PX)
        c.setFillColor(LABEL_BLUE)
        c.drawRightString(w - margin, h - 36 * PX, referencia)
        c.setFont(F_REG, 12 * PX)
        c.setFillColor(MUTED)
        c.drawRightString(w - margin - ref_w - 4, h - 36 * PX, "Especificaciones técnicas ·")
        c.setStrokeColor(RULE)
        c.setLineWidth(0.75)
        c.line(margin, h - 64 * PX, w - margin, h - 64 * PX)
        c.restoreState()

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=first_top, bottomMargin=bottom,
        title="Especificaciones técnicas", author="Asistente IA · Compras Públicas",
        subject=f"Requerimiento de compra {referencia}",
    )
    frame_first = Frame(margin, bottom, content_w, h - first_top - bottom, id="first", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_later = Frame(margin, bottom, content_w, h - later_top - bottom, id="later", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[frame_first], onPage=draw_first, autoNextPageTemplate="later"),
        PageTemplate(id="later", frames=[frame_later], onPage=draw_later),
    ])

    story: List[Any] = [
        Spacer(1, 28 * PX),
        Paragraph("Especificaciones técnicas", S_H1),
        Spacer(1, 22 * PX),
        _meta_strip(_fecha_es(fecha), referencia, schema.label, len(items), content_w),
        Spacer(1, 30 * PX),
    ]
    for i, item in enumerate(items):
        story.append(_item_block(schema, item, f"{i + 1:02d}", content_w))
        if i < len(items) - 1:
            story.append(Spacer(1, 34 * PX))

    doc.build(story, canvasmaker=lambda *a, **k: NumberedCanvas(*a, footer_left=footer_left, **k))
    return buf.getvalue()


# ─── Esquema mínimo para categorías sin CategorySchema (Medicamentos) ───────

from dataclasses import dataclass, field


@dataclass
class PdfAttr:
    label: str
    unit: Optional[str] = None
    joiner: str = " o "


@dataclass
class PdfSchema:
    """Lo único que necesita el PDF de una categoría: nombre, atributos en el
    orden en que se muestran y cuáles forman el título de respaldo."""
    label: str
    attributes: Dict[str, PdfAttr] = field(default_factory=dict)
    title_attrs: tuple = ()


# cantidad_requerida NO va acá: es la "Cantidad" del encabezado de cada ítem.
MEDICAMENTOS_SCHEMA = PdfSchema(
    label="Medicamentos",
    attributes={
        "principio_activo": PdfAttr("Principio activo", joiner=" + "),
        "forma_farmaceutica": PdfAttr("Forma farmacéutica"),
        "concentracion": PdfAttr("Concentración"),
        "unidad_venta": PdfAttr("Unidad de compra"),
        "laboratorio": PdfAttr("Laboratorio"),
        "cantidad": PdfAttr("Tamaño del envase"),
    },
    title_attrs=("principio_activo", "forma_farmaceutica", "concentracion"),
)
