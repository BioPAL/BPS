"""Fig. 22 — Geometry of the ionospheric altitude estimation problem.

ATBD L1 SAR (atbd-l1-sar), section 4.10.2.3.1 (Based on feature tracking,
equation 4.78). Redrawn from the original figure and the text of the section:
a perspective sketch of the orbit (height h_sat above the ground, slant range
R at incidence angle theta), the ionosphere (height h_iono, range R_iono) and
the nadir line, the ground, the synthetic aperture of length L_SA seen from
its two ends and from its half, the approaching (blue, APPR.) and departing
(red, DEPR.) look directions, which cross at the ionospheric structure (grey)
and reach the ground (coloured bands) L_sep apart. A third (black) line runs
from the orbit to the ground between them; the double strokes on the blue and
on the red pairs of lines are the equality marks of the original (parallel
lines). This is a schematic: the positions and the angles are
qualitative, taken from the original figure, not measured.

To modify: change the numbers in GEOMETRY (pixels of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 fig22.py`` (matplotlib needed): it rewrites
fig22.svg (web page) and fig22.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour: the page puts a white
background behind the figure in both site themes.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Polygon  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "black": "#000000", "blue": "#0000ff",
           "red": "#ff0000", "light_blue": "#8080ff", "light_red": "#ff8080",
           "grey": "#bfbfbf"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt

# -- geometry (pixels of the original figure, y downwards) -------------------
X_MIN, X_MAX, Y_MIN, Y_MAX = -12, 1240, -22, 898  # canvas, with room for the larger text
# lines along the track, each (left end, right end, arrow at the right end?)
ORBIT = ((62, 228), (955, 68), True)
IONOSPHERE = ((218, 596), (1225, 410), False)
NADIR = ((62, 725), (950, 567), True)
GROUND = ((335, 873), (1225, 712), True)
# cross-track lines (the perspective depth direction)
CROSS = [((70, 188), (120, 218)), ((69, 474), (276, 585)), ((69, 689), (393, 862))]
SAT, GROUND_PT = (120, 218), (393, 862)  # satellite on the vertical, ground point of R
VERTICAL = ((120, 180), (120, 750))  # vertical through the satellite and the nadir
H_SAT = ((85, 697), (85, 197))  # arrow up, from the nadir plane to the orbit
H_IONO = ((155, 735), (155, 520))  # arrow up, from the nadir plane to the ionosphere
R_IONO = ((178, 205), (334, 576))  # arrow from the orbit to the ionosphere
THETA = {"vertical": ((393, 720), (393, 882)), "radius": 75}
# look directions: orbit end -> ground point G0
G0 = (837, 780)
CENTRE_LOOK = (567, 137)  # black, centre of the synthetic aperture
BLUE_LOOK, RED_LOOK = ((485, 152), (900, 768)), ((650, 122), (775, 790))
THIN_BLUE, DASHED_BLUE = (275, 190), (423, 163)  # orbit points of the lines ending at G0
THIN_RED, DASHED_RED = (857, 86), (715, 110)
# the synthetic aperture L_SA and its half
LSA = ((275, 153), (858, 49))  # double arrow
LSA_TICKS = [((275, 117), (275, 190)), ((858, 8), (858, 86))]
LSA_HALF = ((319, 106), (610, 54))  # double arrow
LSA_HALF_EXT = [((300, 96), DASHED_BLUE), ((595, 46), DASHED_RED)]
# equality marks: (point on the line, direction of the line) for each double stroke
MARKS = [((537, 336), DASHED_BLUE, G0, "blue"), ((603, 326), BLUE_LOOK[0], BLUE_LOOK[1], "blue"),
         ((684, 306), RED_LOOK[0], RED_LOOK[1], "red"), ((747, 296), DASHED_RED, G0, "red")]
MARK_LEN, MARK_GAP = 24, 8
# the ionospheric structure (grey) and the two images on the ground (bands)
BAND_DIR = (0.876, 0.482)  # direction of the bands
STRUCTURE = {"centre": (716, 502), "half_length": 85, "half_width": 13, "colour": "grey"}
BANDS = [{"centre": (905, 774), "half_length": 125, "half_width": 14, "colour": "light_blue"},
         {"centre": (775, 796), "half_length": 125, "half_width": 14, "colour": "light_red"}]
SEP_EXT = [((762, 788), (858, 838)), ((887, 763), (984, 814))]  # along the bands' ends
SEP = ((846, 832), (971, 809))  # L_sep, double arrow

# -- labels (mathtext), word for word from the original figure ----------------
# along the lines: (text, point on the line, colour, side of the line)
ALONG = {
    "orbit": ("ORBIT", ORBIT, 135, "above", "text"),
    "appr": ("APPR.", ORBIT, 322, "below", "blue"),
    "depr": ("DEPR.", ORBIT, 734, "below", "red"),
    "iono": ("IONOSPHERE", IONOSPHERE, 975, "above", "text"),
    "nadir": ("NADIR", NADIR, 196, "below", "text"),
    "ground": ("GROUND", GROUND, 432, "above", "text"),
}
LABELS = {
    "L_SA": (r"$L_{SA}$", (772, 38), "center", -10.2),
    "L_SA/2": (r"$\dfrac{L_{SA}}{2}$", (436, 30), "center", -10.2),
    "h_sat": (r"$h_{sat}$", (36, 405), "center", 0),
    "h_iono": (r"$h_{iono}$", (214, 634), "center", 0),
    "R": (r"$\mathrm{R}$", (281, 528), "center", 0),
    "R_iono": (r"$\mathrm{R}_{iono}$", (270, 370), "left", 0),
    "theta": (r"$\theta$", (368, 742), "center", 0),
    "L_sep": (r"$L_{sep}$", (950, 862), "center", 0),
}
STRUCTURE_LABEL = ("Ionospheric structure", (884, 520), (792, 541))  # text, start, leader end

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MAX, Y_MIN)
ax.axis("off")


def line(p0, p1, colour="black", lw=1.1, ls="-"):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COLOURS[colour], lw=lw, ls=ls,
            solid_capstyle="butt", dash_capstyle="butt")


def arrow(p0, p1, colour="black", lw=1.1, style="-|>", head=11):
    """Arrow whose tip is exactly p1 (and p0 for a double arrow)."""
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle=style, color=COLOURS[colour], lw=lw, shrinkA=0, shrinkB=0,
        mutation_scale=head))


def unit(p0, p1):
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return ((p1[0] - p0[0]) / d, (p1[1] - p0[1]) / d)


def band(centre, half_length, half_width, colour):
    ux, uy = BAND_DIR
    nx, ny = -uy, ux
    pts = [(centre[0] + sl * half_length * ux + sw * half_width * nx,
            centre[1] + sl * half_length * uy + sw * half_width * ny)
           for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(Polygon(pts, closed=True, fc=COLOURS[colour], ec="none", zorder=0))


# coloured areas first
band(STRUCTURE["centre"], STRUCTURE["half_length"], STRUCTURE["half_width"], "grey")
for b in BANDS:
    band(b["centre"], b["half_length"], b["half_width"], b["colour"])
# the track lines
for p0, p1, head in (ORBIT, IONOSPHERE, NADIR, GROUND):
    line(p0, p1)
    if head:
        tail = (p0[0] + (p1[0] - p0[0]) * 0.9, p0[1] + (p1[1] - p0[1]) * 0.9)
        arrow(tail, p1, style="->", head=14)
for p0, p1 in CROSS + [VERTICAL, THETA["vertical"], (SAT, GROUND_PT)]:
    line(p0, p1)
arrow(*H_SAT, style="->", head=14)
arrow(*H_IONO, style="->", head=14)
arrow(*R_IONO, style="->", head=14)
# angle theta between the vertical and R at the ground point
ang_r = math.degrees(math.atan2(SAT[1] - GROUND_PT[1], SAT[0] - GROUND_PT[0]))
rad = THETA["radius"]
ax.add_patch(Arc(GROUND_PT, 2 * rad, 2 * rad, theta1=ang_r % 360, theta2=270,
                 color=COLOURS["black"], lw=1.0))
# look directions
for tail, col in ((THIN_BLUE, "light_blue"), (THIN_RED, "light_red")):
    line(tail, G0, col, lw=1.0)
for tail, col in ((DASHED_BLUE, "light_blue"), (DASHED_RED, "light_red")):
    line(tail, G0, col, lw=1.8, ls=(0, (7, 5)))
arrow(CENTRE_LOOK, G0, "black", lw=1.6)
arrow(*BLUE_LOOK, "blue", lw=1.6)
arrow(*RED_LOOK, "red", lw=1.6)
# equality marks
for centre, p0, p1, col in MARKS:
    ux, uy = unit(p0, p1)
    nx, ny = -uy, ux
    for k in (-0.5, 0.5):
        cx, cy = centre[0] + k * MARK_GAP * ux, centre[1] + k * MARK_GAP * uy
        line((cx - nx * MARK_LEN / 2, cy - ny * MARK_LEN / 2),
             (cx + nx * MARK_LEN / 2, cy + ny * MARK_LEN / 2), col, lw=1.3)
# synthetic aperture and L_sep dimensions
for t in LSA_TICKS:
    line(*t)
arrow(*LSA, style="<|-|>", head=11)
arrow(*LSA_HALF, style="<|-|>", head=11)
for p0, p1 in LSA_HALF_EXT + [(e, e2) for e, e2 in SEP_EXT]:
    line(p0, p1)
arrow(*SEP, style="<|-|>", head=11)

# labels along the lines
for text, (p0, p1, _), x, side, col in ALONG.values():
    ux, uy = unit(p0, p1)
    y = p0[1] + (x - p0[0]) * uy / ux
    nx, ny = (uy, -ux) if side == "above" else (-uy, ux)  # unit normal towards the side
    ang = -math.degrees(math.atan2(uy, ux))
    ax.text(x + 9 * nx, y + 9 * ny, text, color=COLOURS[col], rotation=ang,
            rotation_mode="anchor", ha="left" if text != "IONOSPHERE" else "left",
            va="bottom" if side == "above" else "top")
for s, (x, y), ha, rot in LABELS.values():
    ax.text(x, y, s, color=COLOURS["text"], ha=ha, va="center", rotation=rot)
text, pos, end = STRUCTURE_LABEL
ax.text(pos[0], pos[1], text, color=COLOURS["text"], ha="left", va="center", style="italic",
        rotation=10.3, rotation_mode="anchor")  # parallel to the ionosphere line
line((pos[0] - 6, pos[1] + 2), end, lw=0.9)  # leader to the grey structure

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
