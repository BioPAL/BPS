"""Fig. 12 — Schematic representation of the range of admissible values for
parameters a and b, i.e., the range of values that ensures that the matrices
returned by the decomposition are semi-definite positive matrices.

ATBD L1 STACK (atbd-l1-stack), section 5.4.2.3 (SKP decomposition). Redrawn
from the original figure and the text of the section: R_g(a), C_v(a), R_v(b)
and C_g(b) (equations 5.59 to 5.62) must be positive; where one of them stops
being so, its determinant vanishes, which fixes the bounds a_min, a_max,
b_min and b_max. This is a schematic: horizontal positions are qualitative.

To modify: change the positions in GEOMETRY (units of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``/usr/bin/python3 fig12.py`` (matplotlib needed): it rewrites
fig12.svg (web page) and fig12.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour: the page puts a white
background behind the figure in both site themes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "a": "#ff0000", "b": "#00b050", "line": "#000000"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 8.5
THICK, THIN = 2.2, 0.9  # line widths (pt)

# -- geometry (pixels of the original figure, y downwards) -------------------
W, H = 1380, 545
Y_TRUE, Y_FALSE = (80, 150), (257, 313)  # upper and lower panel levels
Y_AXIS3 = 420  # axis of the admissible range
# left column (parameter a) and right column (parameter b)
COLS = {
    "a": {
        "x0": 100, "x1": 455, "yaxis": 117,
        "top": (193, 363),     # R_g(a) > 0 is true between these x
        "bot": (277, 422),     # C_v(a) > 0 is true between these x
        "bounds": (277, 363),  # a_min, a_max
        "top_end": 455, "bot_end": 445,
    },
    "b": {
        "x0": 850, "x1": 1205, "yaxis": 866,
        "top": (975, 1145),    # R_v(b) > 0
        "bot": (920, 1065),    # C_g(b) > 0
        "bounds": (975, 1065),  # b_min, b_max
        "top_end": 1195, "bot_end": 1205,
    },
}

# -- labels (mathtext; parts = list of (text, colour key)) --------------------
LABELS = {
    "true": r"$true$", "false": r"$false$",
    "top_a": [(r"$R_g($", "text"), (r"$a$", "a"), (r"$) > 0$", "text")],
    "bot_a": [(r"$C_v($", "text"), (r"$a$", "a"), (r"$) > 0$", "text")],
    "top_b": [(r"$R_v($", "text"), (r"$b$", "b"), (r"$) > 0$", "text")],
    "bot_b": [(r"$C_g($", "text"), (r"$b$", "b"), (r"$) > 0$", "text")],
    "a": [(r"$a$", "a")], "b": [(r"$b$", "b")],
    "amin": [(r"$a_{min}$", "a")], "amax": [(r"$a_{max}$", "a")],
    "bmin": [(r"$b_{min}$", "b")], "bmax": [(r"$b_{max}$", "b")],
    "det_amin": [(r"$\det(C_v($", "text"), (r"$a_{min}$", "a"), (r"$)) = 0$", "text")],
    "det_amax": [(r"$\det(R_g($", "text"), (r"$a_{max}$", "a"), (r"$)) = 0$", "text")],
    "det_bmin": [(r"$\det(R_v($", "text"), (r"$b_{min}$", "b"), (r"$)) = 0$", "text")],
    "det_bmax": [(r"$\det(C_g($", "text"), (r"$b_{max}$", "b"), (r"$)) = 0$", "text")],
}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
renderer = fig.canvas.get_renderer()
LINE = COLOURS["line"]


def hline(x0, x1, y, lw=THIN):
    ax.plot([x0, x1], [y, y], color=LINE, lw=lw, solid_capstyle="butt")


def arrow(x0, y0, x1, y1, lw=THIN):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(
        arrowstyle="-|>,head_length=0.35,head_width=0.15", color=LINE, lw=lw,
        shrinkA=0, shrinkB=0))


def label(x, y, parts, ha="left", va="center"):
    """Text in several colours, laid out left to right from (x, y)."""
    texts, width = [], 0
    for s, key in parts:
        t = ax.text(x, y, s, color=COLOURS[key], ha="left", va=va)
        w = t.get_window_extent(renderer).width * W / (WIDTH_IN * fig.dpi)
        texts.append((t, w))
        width += w
    x0 = x - (width if ha == "right" else width / 2 if ha == "center" else 0)
    for t, w in texts:
        t.set_x(x0)
        x0 += w


def panel(col, level, side):
    c = COLS[col]
    yt, yf = level
    lo, hi = c["top"] if side == "top" else c["bot"]
    end = c["top_end"] if side == "top" else c["bot_end"]
    ax_y = 150 if side == "top" else 313
    # y axis, with the tick at the "true" level (the lower panel has an arrowhead)
    top_y = 65 if side == "top" else 235
    if side == "top":
        ax.plot([c["yaxis"]] * 2, [top_y, ax_y + 35], color=LINE, lw=THIN)
    else:
        arrow(c["yaxis"], ax_y + 50, c["yaxis"], top_y)
    hline(c["yaxis"] - 15, c["yaxis"] + 18, yt)
    label(c["x0"] - 5, yt, [(LABELS["true"], "text")], ha="right")
    label(c["x0"] - 12, yf, [(LABELS["false"], "text")], ha="right")
    # x axis with its arrowhead, thick where the function is false
    hline(c["x0"], c["x1"] - 4, yf)
    arrow(c["x1"] - 20, yf, c["x1"], yf)
    hline(c["x0"], lo, yf, THICK)
    hline(hi, end, yf, THICK)
    # thick true segment
    hline(lo, hi, yt, THICK)
    label(c["x1"] + 7, yf + 28, LABELS[col])


def dotted(x, y0, y1):
    ax.plot([x, x], [y0, y1], color=LINE, lw=THIN, dashes=(6, 2, 1.2, 2))


for col in ("a", "b"):
    c = COLS[col]
    panel(col, (Y_TRUE[0], Y_TRUE[1]), "top")
    panel(col, (Y_FALSE[0], Y_FALSE[1]), "bot")
    # titles
    tx = c["yaxis"] + 8 if col == "a" else c["yaxis"] + 9
    label(tx, 25, LABELS[f"top_{col}"])
    label(tx, 205, LABELS[f"bot_{col}"])
    # admissible-range axis
    lo, hi = c["bounds"]
    hline(c["x0"], c["x1"] - 4, Y_AXIS3)
    arrow(c["x1"] - 20, Y_AXIS3, c["x1"], Y_AXIS3)
    hline(lo, hi, Y_AXIS3, THICK)
    label(c["x1"] + 7, Y_AXIS3 + 32, LABELS[col])
    # dotted bounds
    dotted(hi if col == "a" else lo, Y_TRUE[0], Y_AXIS3)
    dotted(lo if col == "a" else hi, Y_FALSE[0], Y_AXIS3)
    label(lo - 8, 452, LABELS[f"{col}min"])
    label(hi + 2, 452, LABELS[f"{col}max"])

# determinant labels and the arrows towards them
for col, x_left, x_right in (("a", 35, 420), ("b", 727, 1125)):
    lo, hi = COLS[col]["bounds"]
    arrow(lo + 13, 470, lo - 9, 493)
    arrow(hi + 37, 477, hi + 50, 493)
    label(x_left, 518, LABELS[f"det_{col}min"])
    label(x_right, 518, LABELS[f"det_{col}max"])

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
