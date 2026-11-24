"""Fig. 29 — Relation between the east-north and vertical coordinate and the
B-aligned coordinate Here the geomagnetic field B is assigned with declination
Dec and inclination, or geomagnetic dip angle I. The direction of x-axis in the
B-aligned coordinate is parallel to the projection of B on the horizontal plane.

ATBD L1 SAR (atbd-l1-sar), section 4.10.5 (Step 3, equation 4.111 and the text
that follows). Redrawn from the original figure and the text of the section:
the B-aligned frame has x along the horizontal projection of the geomagnetic
field B, z vertically downward and y = z x x; the figure also shows the range
direction (unit vector e_rg, LOS projected on the horizontal plane), the
azimuth direction (unit vector e_az), the propagation vector kappa (green arrow
pointing downwards), the incidence angle theta, the angle phi between the
projection of kappa on the horizontal plane and x, and the projection of B
along kappa (red arrow). This is a schematic perspective sketch: the positions
and angles are qualitative, taken from the original figure, not measured.

To modify: change the positions in GEOMETRY (pixels of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 fig29.py`` (matplotlib needed): it rewrites
fig29.svg (web page) and fig29.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour: the page puts a white
background behind the figure in both site themes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "black": "#000000", "blue": "#4472c4",
           "B": "#0000ff", "green": "#00b050", "red": "#ff0000"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt

# -- geometry (pixels of the original figure, y downwards) -------------------
X_MIN, X_MAX, Y_MIN, Y_MAX = -10, 800, 5, 595  # canvas, with room for the larger text
ORIGIN = (227, 261)  # origin of all the vectors
RANGE_TIP, AZIMUTH_TIP = (190, 57), (617, 203)  # e_rg and e_az
NORTH_TIP, EAST_TIP = (452, 68), (618, 373)
X_TIP, Y_TIP, Z_TIP = (448, 240), (237, 360), (227, 455)  # B-aligned axes
B_TIP = (603, 572)  # the geomagnetic field
B_HORIZ = (603, 225)  # projection of B on the horizontal plane (along x)
KAPPA_TIP = (191, 402)  # propagation vector, green, pointing downwards
B_ALONG_KAPPA = (639, 430)  # projection of B along kappa (red arrow)
# dashed construction lines, each ((x, y), (x, y))
DASHED = [
    (RANGE_TIP, KAPPA_TIP),  # vertical, under the range tip
    (RANGE_TIP, (395, 115)),  # to the north axis
    ((353, 153), B_HORIZ),  # from the north axis to the projection of B
    (RANGE_TIP, (20, 202)),
    ((7, 198), ORIGIN),
    ((478, 334), B_HORIZ),  # from the east axis to the projection of B
    (B_HORIZ, B_TIP),  # vertical, from the projection of B down to B
]
# angle arcs: (start, a point on the arc, end); the arrow points to the end
ARC_PHI = ((373, 247), (303, 191), (214, 185))  # x axis to range direction
ARC_DEC = ((272, 224), (293, 230), (312, 248))  # north to x axis
ARC_I = ((312, 248), (314, 282), (295, 317))  # x axis (horizontal) down to B
ARC_THETA = ((227, 412), (217, 365), (213, 315))  # vertical (z) to kappa

# -- labels (mathtext), word for word from the original figure ----------------
LABELS = {
    "range": (r"Range $\vec{e}_{rg}$", (95, 30), "left"),
    "azimuth": (r"Azimuth $\vec{e}_{az}$", (569, 178), "left"),
    "north": ("North", (452, 47), "left"),
    "east": ("East", (630, 376), "left"),
    "x": (r"$x$", (459, 207), "center"),
    "y": (r"$y$", (253, 376), "center"),
    "z": (r"$z$", (245, 470), "center"),
    "B": (r"$\vec{B}$", (633, 560), "left"),
    "kappa": (r"$\hat{\kappa}$", (172, 432), "center"),
    "phi": (r"$\phi$", (290, 160), "center"),
    "dec": (r"$Dec$", (332, 227), "center"),
    "I": (r"$I$", (336, 273), "center"),
    "theta": (r"$\theta$", (208, 434), "center"),
}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MAX, Y_MIN)
ax.axis("off")


def arrow(p0, p1, colour, lw=1.0, head=9, rad=0.0, style="-|>"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle=style, color=COLOURS[colour], lw=lw, shrinkA=0, shrinkB=0,
        mutation_scale=head, connectionstyle=f"arc3,rad={rad}"))


for p0, p1 in DASHED:
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COLOURS["blue"], lw=0.8,
            dashes=(4, 2.5), zorder=1)

# thin blue lines: north, east, and the projection of B on the horizontal plane
arrow(ORIGIN, NORTH_TIP, "blue", head=11)
arrow(ORIGIN, EAST_TIP, "blue", head=11)
arrow(ORIGIN, B_HORIZ, "blue")
# green: range, azimuth, kappa; green segment closing the triangle with B
arrow(ORIGIN, RANGE_TIP, "green", head=10)
arrow(ORIGIN, AZIMUTH_TIP, "green", head=10)
arrow(ORIGIN, KAPPA_TIP, "green", head=10)
arrow(B_ALONG_KAPPA, B_TIP, "green", head=10)
# black axes x, y, z (B-aligned)
arrow(ORIGIN, X_TIP, "black", lw=1.1, head=10)
arrow(ORIGIN, Y_TIP, "black", lw=1.1, head=10)
arrow(ORIGIN, Z_TIP, "black", lw=1.1, head=10)
# the red arrow (B projected along kappa) and the geomagnetic field B
arrow(ORIGIN, B_ALONG_KAPPA, "red", head=9)
arrow(ORIGIN, B_TIP, "B", lw=2.0, head=11)
# angle arcs
for p0, pm, p1 in (ARC_PHI, ARC_DEC, ARC_I, ARC_THETA):
    ctrl = (2 * pm[0] - (p0[0] + p1[0]) / 2, 2 * pm[1] - (p0[1] + p1[1]) / 2)
    ax.add_patch(FancyArrowPatch(
        path=MPath([p0, ctrl, p1], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
        arrowstyle="-|>", color=COLOURS["blue"], lw=0.9, mutation_scale=8,
        shrinkA=0, shrinkB=0))

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
