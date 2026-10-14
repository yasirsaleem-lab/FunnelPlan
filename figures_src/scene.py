"""
Tiny scene builder for the paper's diagrams. One scene is written twice:
  * a draw.io file (.drawio, editable in diagrams.net; export PDF from there if you edit it), and
  * an SVG with the same coordinates, and
  * a PDF and a 600 dpi PNG drawn with matplotlib from the same scene (cairosvg is optional: --cairo).
Coordinates are pixels on a canvas. The SVG and PDF carry the true print width (7.16 in, the IEEE two column
figure width, unless a figure says otherwise), so LaTeX can place them at natural size. At a 1400 px canvas
printed at 7.16 in, 19 px is about 7 pt, the smallest text the figures use. PNGs are written at 600 dpi.
"""

import html
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import ImageFont

# Liberation Sans and Arial have identical character widths, so the layout is the same on every system.
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"      # SVG font-family list; draw.io uses Helvetica
SYMBOL_FONT = "DejaVu Sans, Segoe UI Symbol, sans-serif"     # for the loop arrow glyph
_WIN = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_HERE = Path(__file__).parent / "fonts"                      # optional: drop LiberationSans-*.ttf here
FONT_CANDIDATES = {
    False: [_HERE / "LiberationSans-Regular.ttf",
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"),
            _WIN / "arial.ttf", Path("/Library/Fonts/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf")],
    True: [_HERE / "LiberationSans-Bold.ttf",
           Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
           Path("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"),
           _WIN / "arialbd.ttf", Path("/Library/Fonts/Arial Bold.ttf"),
           Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")],
}
_FONTS = {}


def _font_file(bold):
    for f in FONT_CANDIDATES[bold]:
        if f.is_file():
            return str(f)
    raise FileNotFoundError("No Liberation Sans or Arial font found for text measurement. Put "
                            "LiberationSans-Regular.ttf and LiberationSans-Bold.ttf in figures_src/fonts/.")


def text_width(s, size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(_font_file(bold), size)
    return _FONTS[key].getlength(s)


def wrap(s, size, width, bold=False):
    words, lines, cur = s.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if text_width(t, size, bold) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Scene:
    def __init__(self, w, h, width_in=7.16):
        self.w, self.h, self.width_in = w, h, width_in
        self.items, self.n, self.warn = [], 0, []

    def _id(self):
        self.n += 1
        return f"s{self.n}"

    # ------------------------------------------------------------ shapes
    def box(self, x, y, w, h, fill="#ffffff", stroke="#000000", sw=1.5, r=8, dash=False, kind="rect", shadow=False):
        i = self._id()
        self.items.append(dict(t="shape", id=i, kind=kind, x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, r=r,
                               dash=dash, shadow=shadow))
        return i

    def text(self, x, y, w, h, lines, align="center", valign="middle", pad=6, lh=1.25, family=None):
        """lines: list of (text, size, bold, color). Checks that every line fits inside the width."""
        i = self._id()
        for s, size, bold, color in lines:
            if s and text_width(s, size, bold) > w - 2 * pad + 0.5:
                self.warn.append(f"overflow: '{s}' ({text_width(s, size, bold):.0f} > {w - 2 * pad:.0f})")
        self.items.append(dict(t="text", id=i, x=x, y=y, w=w, h=h, lines=lines, align=align, valign=valign, pad=pad,
                               lh=lh, family=family or FONT))
        tot = sum(sz * lh for _, sz, _, _ in lines)
        if tot > h + 1:
            self.warn.append(f"too tall: {[l[0] for l in lines][:2]} ({tot:.0f} > {h})")
        return i

    def line(self, pts, color="#3F4756", sw=2.0, dash=False, arrow=True):
        i = self._id()
        self.items.append(dict(t="line", id=i, pts=pts, color=color, sw=sw, dash=dash, arrow=arrow))
        return i

    # ------------------------------------------------------------ SVG
    def svg(self):
        win = self.width_in
        hin = win * self.h / self.w
        colors = sorted({it["color"] for it in self.items if it["t"] == "line" and it["arrow"]})
        markers = "".join(f'<marker id="ah{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                          f'markerHeight="7" orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
                          f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colors)
        S = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{win:.3f}in" height="{hin:.3f}in" '
             f'viewBox="0 0 {self.w} {self.h}">',
             f'<defs>{markers}',
             '<filter id="sh" x="-5%" y="-5%" width="115%" height="125%"><feDropShadow dx="0" dy="1.5" '
             'stdDeviation="1.5" flood-color="#000" flood-opacity="0.18"/></filter></defs>',
             f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>']
        for it in self.items:
            if it["t"] == "shape":
                d = ' stroke-dasharray="7,5"' if it["dash"] else ""
                f = ' filter="url(#sh)"' if it["shadow"] else ""
                if it["kind"] == "diamond":
                    x, y, w, h = it["x"], it["y"], it["w"], it["h"]
                    pts = f"{x + w / 2},{y} {x + w},{y + h / 2} {x + w / 2},{y + h} {x},{y + h / 2}"
                    S.append(f'<polygon points="{pts}" fill="{it["fill"]}" stroke="{it["stroke"]}" '
                             f'stroke-width="{it["sw"]}"{d}{f}/>')
                elif it["kind"] == "ellipse":
                    S.append(f'<ellipse cx="{it["x"] + it["w"] / 2}" cy="{it["y"] + it["h"] / 2}" rx="{it["w"] / 2}" '
                             f'ry="{it["h"] / 2}" fill="{it["fill"]}" stroke="{it["stroke"]}" stroke-width="{it["sw"]}"/>')
                else:
                    S.append(f'<rect x="{it["x"]}" y="{it["y"]}" width="{it["w"]}" height="{it["h"]}" rx="{it["r"]}" '
                             f'fill="{it["fill"]}" stroke="{it["stroke"]}" stroke-width="{it["sw"]}"{d}{f}/>')
            elif it["t"] == "text":
                lines, lh, pad = it["lines"], it["lh"], it["pad"]
                tot = sum(sz * lh for _, sz, _, _ in lines)
                y = it["y"] + (it["h"] - tot) / 2 if it["valign"] == "middle" else it["y"] + pad
                anchor = {"center": "middle", "left": "start", "right": "end"}[it["align"]]
                x = {"center": it["x"] + it["w"] / 2, "left": it["x"] + pad, "right": it["x"] + it["w"] - pad}[it["align"]]
                for s, sz, bold, color in lines:
                    y += sz * lh
                    S.append(f'<text xml:space="preserve" x="{x}" y="{y - sz * (lh - 1) / 2 - sz * 0.22:.1f}" font-family="{it["family"]}" '
                             f'font-size="{sz}" font-weight="{"bold" if bold else "normal"}" fill="{color}" '
                             f'text-anchor="{anchor}">{html.escape(s)}</text>')
            elif it["t"] == "line":
                d = ' stroke-dasharray="7,5"' if it["dash"] else ""
                m = f' marker-end="url(#ah{it["color"][1:]})"' if it["arrow"] else ""
                p = " ".join(f"{x},{y}" for x, y in it["pts"])
                S.append(f'<polyline points="{p}" fill="none" stroke="{it["color"]}" stroke-width="{it["sw"]}" '
                         f'stroke-linejoin="round"{d}{m}/>')
        S.append("</svg>")
        return "\n".join(S)

    # ------------------------------------------------------------ draw.io
    def drawio(self, name):
        mx = ET.Element("mxfile", host="FunnelPlan figures_src")
        dg = ET.SubElement(mx, "diagram", name=name, id=name)
        m = ET.SubElement(dg, "mxGraphModel", dx="1400", dy="900", grid="0", gridSize="10", guides="1", tooltips="1",
                          connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth=str(self.w),
                          pageHeight=str(self.h), math="0", shadow="0")
        root = ET.SubElement(m, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for it in self.items:
            if it["t"] == "shape":
                if it["kind"] == "diamond":
                    st = "rhombus;whiteSpace=wrap;html=1;"
                elif it["kind"] == "ellipse":
                    st = "ellipse;html=1;"
                else:
                    arc = min(50, round(100 * it["r"] / max(1, min(it["w"], it["h"]))))
                    st = f"rounded=1;arcSize={arc};html=1;" if it["r"] else "rounded=0;html=1;"
                st += f"fillColor={it['fill']};strokeColor={it['stroke']};strokeWidth={it['sw']};"
                st += "dashed=1;" if it["dash"] else ""
                st += "shadow=1;" if it["shadow"] else ""
                c = ET.SubElement(root, "mxCell", id=it["id"], value="", style=st, vertex="1", parent="1")
                ET.SubElement(c, "mxGeometry", x=str(it["x"]), y=str(it["y"]), width=str(it["w"]), height=str(it["h"]),
                              **{"as": "geometry"})
            elif it["t"] == "text":
                parts = []
                for s, sz, bold, color in it["lines"]:
                    t = html.escape(s)
                    t = t.replace("  ", "&nbsp;&nbsp;")
                    t = f'<font style="font-size:{sz}px" color="{color}">{t}</font>'
                    parts.append(f"<b>{t}</b>" if bold else t)
                st = (f"text;html=1;whiteSpace=wrap;fillColor=none;strokeColor=none;fontFamily=Helvetica;"
                      f"align={it['align']};verticalAlign={'middle' if it['valign'] == 'middle' else 'top'};"
                      f"spacing={it['pad']};")
                c = ET.SubElement(root, "mxCell", id=it["id"], value="<br>".join(parts), style=st, vertex="1",
                                  parent="1")
                ET.SubElement(c, "mxGeometry", x=str(it["x"]), y=str(it["y"]), width=str(it["w"]), height=str(it["h"]),
                              **{"as": "geometry"})
            elif it["t"] == "line":
                st = (f"endArrow={'block' if it['arrow'] else 'none'};endFill=1;html=1;rounded=0;"
                      f"strokeColor={it['color']};strokeWidth={it['sw']};" + ("dashed=1;" if it["dash"] else ""))
                c = ET.SubElement(root, "mxCell", id=it["id"], value="", style=st, edge="1", parent="1")
                g = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
                (x0, y0), (x1, y1) = it["pts"][0], it["pts"][-1]
                ET.SubElement(g, "mxPoint", x=str(x0), y=str(y0), **{"as": "sourcePoint"})
                ET.SubElement(g, "mxPoint", x=str(x1), y=str(y1), **{"as": "targetPoint"})
                if len(it["pts"]) > 2:
                    arr = ET.SubElement(g, "Array", **{"as": "points"})
                    for x, y in it["pts"][1:-1]:
                        ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))
        return ET.tostring(mx, encoding="unicode")

    # ------------------------------------------------------------ matplotlib (PDF and PNG on any system)
    def _render_mpl(self, base, dpi):
        """Draw the scene with matplotlib, which needs no system library. Same coordinates and text baselines as
        the SVG. Writes <base>.pdf (vector, TrueType fonts embedded) and <base>.png at `dpi`."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon

        k = self.width_in * 72 / self.w                     # canvas px -> points
        installed = {f.name for f in font_manager.fontManager.ttflist}

        def fontprops(family, bold, size):
            """the main text uses the very file text_width() measured; other families use the first installed one"""
            if family == FONT:
                return font_manager.FontProperties(fname=_font_file(bold), size=size)
            names = [x.strip() for x in family.split(",")]
            name = next((n for n in names if n in installed), "DejaVu Sans")   # DejaVu ships with matplotlib
            return font_manager.FontProperties(family=name, weight="bold" if bold else "normal", size=size)
        rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "lines.scale_dashes": False, "svg.fonttype": "none"}
        with plt.rc_context(rc):
            fig = plt.figure(figsize=(self.width_in, self.width_in * self.h / self.w))
            ax = fig.add_axes((0, 0, 1, 1))
            ax.set_xlim(0, self.w)
            ax.set_ylim(self.h, 0)
            ax.axis("off")
            fig.patch.set_facecolor("#ffffff")
            dash = (0, (7 * k, 5 * k))
            for z, it in enumerate(self.items, 1):
                if it["t"] == "shape":
                    x, y, w, h = it["x"], it["y"], it["w"], it["h"]

                    def patch(dx=0.0, dy=0.0, **kw):
                        if it["kind"] == "diamond":
                            pts = [(x + w / 2 + dx, y + dy), (x + w + dx, y + h / 2 + dy),
                                   (x + w / 2 + dx, y + h + dy), (x + dx, y + h / 2 + dy)]
                            return Polygon(pts, closed=True, **kw)
                        if it["kind"] == "ellipse":
                            return Ellipse((x + w / 2 + dx, y + h / 2 + dy), w, h, **kw)
                        r = min(it["r"], w / 2, h / 2)
                        return FancyBboxPatch((x + dx, y + dy), w, h, boxstyle=f"round,pad=0,rounding_size={r}"
                                              if r else "square,pad=0", **kw)
                    if it["shadow"]:
                        ax.add_patch(patch(0, 1.5, facecolor="#000000", edgecolor="none", alpha=0.10, zorder=z))
                    ax.add_patch(patch(facecolor=it["fill"], edgecolor=it["stroke"], linewidth=it["sw"] * k,
                                       linestyle=dash if it["dash"] else "solid", joinstyle="round", zorder=z))
                elif it["t"] == "text":
                    lines, lh, pad = it["lines"], it["lh"], it["pad"]
                    tot = sum(sz * lh for _, sz, _, _ in lines)
                    y = it["y"] + (it["h"] - tot) / 2 if it["valign"] == "middle" else it["y"] + pad
                    x = {"center": it["x"] + it["w"] / 2, "left": it["x"] + pad,
                         "right": it["x"] + it["w"] - pad}[it["align"]]
                    for s, sz, bold, color in lines:
                        y += sz * lh
                        ax.text(x, y - sz * (lh - 1) / 2 - sz * 0.22, s, color=color,
                                fontproperties=fontprops(it["family"], bold, sz * k),
                                ha=it["align"], va="baseline", zorder=z, clip_on=False)
                elif it["t"] == "line":
                    xs, ys = zip(*it["pts"])
                    ax.add_line(Line2D(xs, ys, color=it["color"], linewidth=it["sw"] * k,
                                       linestyle=dash if it["dash"] else "solid", solid_joinstyle="round",
                                       dash_joinstyle="round", solid_capstyle="butt", zorder=z))
                    if it["arrow"]:                  # same head as the SVG marker: 7 px long, 7 px wide
                        (x0, y0), (x1, y1) = it["pts"][-2], it["pts"][-1]
                        d = max(1e-9, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
                        ux, uy = (x1 - x0) / d, (y1 - y0) / d
                        tx, ty = x1 + 0.7 * ux, y1 + 0.7 * uy
                        bx, by = tx - 7 * ux, ty - 7 * uy
                        ax.add_patch(Polygon([(tx, ty), (bx - 3.5 * uy, by + 3.5 * ux), (bx + 3.5 * uy, by - 3.5 * ux)],
                                             closed=True, facecolor=it["color"], edgecolor="none", zorder=z))
            fig.savefig(base.with_suffix(".pdf"))
            fig.savefig(base.with_suffix(".png"), dpi=dpi)
            plt.close(fig)

    def _render_cairo(self, base, dpi, svg):
        import cairosvg                              # needs the cairo system library (usually missing on Windows)
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(base.with_suffix(".pdf")))
        png = base.with_suffix(".png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), output_width=round(self.width_in * dpi))
        from PIL import Image                        # stamp the resolution so Word and LaTeX read the print size
        with Image.open(png) as im:
            im.load()
        im.save(png, dpi=(dpi, dpi), optimize=True)

    def save(self, base, dpi=600, renderer=None):
        """Write <base>.svg, .drawio, .pdf and .png (PNG at `dpi`, default 600). PDF and PNG come from matplotlib,
        which works on every system; pass renderer="cairo" (or run the script with --cairo) to use cairosvg."""
        import sys
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        svg = self.svg()
        base.with_suffix(".svg").write_text(svg, encoding="utf-8")
        base.with_suffix(".drawio").write_text(self.drawio(base.stem), encoding="utf-8")
        renderer = renderer or ("cairo" if "--cairo" in sys.argv else "matplotlib")
        if renderer == "cairo":
            self._render_cairo(base, dpi, svg)
        else:
            self._render_mpl(base, dpi)
        for w in self.warn:
            print("WARNING", w)
        return base
