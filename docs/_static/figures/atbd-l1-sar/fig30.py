"""Fig. 30 — B-aligned axes x and y, and ellipse aligned axes x' and y'. The
latter is acquired by rotating the former coordinate as much as -θ. For
information, east and north directions are shown. The green arrow indicates
the range direction, or LOS.

ATBD L1 SAR (atbd-l1-sar), section 4.10.5 (Step 3.c, equations 4.113 to 4.117).
Redrawn from the original figure and the text of the section: the ellipse
A x^2 + B xy + C y^2 = 1 is centred on the origin of the B-aligned axes (x, y);
rotating the axes by the angle alpha (alpha = -delta/2) gives the principal
axes x' (major axis) and y' (minor axis) of the ellipse, drawn as red arrows.
East, North and the range direction (green) are given for information. This is
a schematic: the axis lengths and the angles are qualitative, taken from the
original figure, not measured.

To modify: change the numbers in GEOMETRY (pixels of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 fig30.py`` (matplotlib needed): it rewrites
fig30.svg (web page) and fig30.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour: the page puts a white
background behind the figure in both site themes.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "black": "#000000", "light": "#8faadc",
           "arc": "#4472c4", "green": "#00b050", "red": "#ff0000"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt

# -- geometry (pixels of the original figure, y downwards) -------------------
X_MIN, X_MAX, Y_MIN, Y_MAX = -15, 725, -15, 705  # canvas, with room for the larger text
ORIGIN = (329, 363)  # origin of the B-aligned axes
X_AXIS = (0, 655)  # x extent of the x axis (arrow on the right)
Y_AXIS = (692, 35)  # y extent of the y axis (arrow at the top)
RANGE_TIP = (311, 35)  # green arrow, range direction (LOS)
NORTH_TIP, EAST_TIP = (506, 88), (605, 540)
ELLIPSE = {"a": 220, "b": 27, "angle": 41.9}  # semi-axes and tilt of the major axis
XP_LEN, YP_LEN = 338, 335  # length of the x' and y' arrows
ALPHA_R = 56  # radius of the alpha arc

# -- labels (mathtext), word for word from the original figure ----------------
LABELS = {
    "x": (r"$x$", (668, 363), "center"),
    "y": (r"$y$", (322, 18), "center"),
    "xp": (r"$x^{\prime}$", (600, 598), "center"),
    "yp": (r"$y^{\prime}$", (572, 106), "center"),
    "north": ("North", (478, 67), "center"),
    "east": ("East", (640, 547), "center"),
    "alpha": (r"$\alpha$", (404, 386), "center"),
}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MAX, Y_MIN)
ax.axis("off")


def arrow(p0, p1, colour, lw=1.0, head=9):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle="-|>", color=COLOURS[colour], lw=lw, shrinkA=0, shrinkB=0,
        mutation_scale=head))


def polar(r, deg):
    """Point at distance r from ORIGIN, deg degrees below the +x axis (y down)."""
    a = math.radians(deg)
    return (ORIGIN[0] + r * math.cos(a), ORIGIN[1] + r * math.sin(a))


ang = ELLIPSE["angle"]
ax.add_patch(Ellipse(ORIGIN, 2 * ELLIPSE["a"], 2 * ELLIPSE["b"], angle=ang, fill=False,
                     edgecolor=COLOURS["black"], lw=1.4))
# axes x, y (B-aligned) and the green range arrow
ax.plot([X_AXIS[0], ORIGIN[0]], [ORIGIN[1], ORIGIN[1]], color=COLOURS["black"], lw=1.0)
arrow((ORIGIN[0], ORIGIN[1]), (X_AXIS[1], ORIGIN[1]), "black")
ax.plot([ORIGIN[0], ORIGIN[0]], [Y_AXIS[0], ORIGIN[1]], color=COLOURS["black"], lw=1.0)
arrow(ORIGIN, (ORIGIN[0], Y_AXIS[1]), "black")
arrow(ORIGIN, RANGE_TIP, "green", lw=0.9)
# east and north, for information
arrow(ORIGIN, NORTH_TIP, "light", lw=1.0, head=11)
arrow(ORIGIN, EAST_TIP, "light", lw=1.0, head=11)
# ellipse-aligned axes x' and y'
arrow(ORIGIN, polar(XP_LEN, ang), "black")
arrow(ORIGIN, polar(YP_LEN, ang - 90), "black")
# principal semi-axes of the ellipse (red)
arrow(ORIGIN, polar(ELLIPSE["a"], ang), "red")
arrow(ORIGIN, polar(ELLIPSE["b"], ang - 90), "red")
# the angle alpha, between x and x'
p0, p1 = polar(ALPHA_R, 0), polar(ALPHA_R, ang)
pm = polar(ALPHA_R, ang / 2)
ctrl = (2 * pm[0] - (p0[0] + p1[0]) / 2, 2 * pm[1] - (p0[1] + p1[1]) / 2)
ax.add_patch(FancyArrowPatch(
    path=MPath([p0, ctrl, p1], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
    arrowstyle="-", color=COLOURS["arc"], lw=0.9, shrinkA=0, shrinkB=0))

for s, (x, y), ha in LABELS.values():
    ax.text(x, y, s, color=COLOURS["text"], ha=ha, va="center")

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
