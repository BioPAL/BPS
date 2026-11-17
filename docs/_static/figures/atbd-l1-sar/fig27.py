"""Fig. 27 — The definition of geomagnetic time. It is defined by the angle
between two planes, one defined by geomagnetic pole, point of interest and
Earth’s center and the second by the geomagnetic pole, subsolar point and the
Earth’s center. ts is conventional solar time, and tm is geomagnetic time.

ATBD L1 SAR (atbd-l1-sar), section 4.10.5 (equations 4.105 to 4.108 and the
text around them). Redrawn from the original figure and the text of the
section: the Earth as a sphere seen in orthographic projection from 45 degrees
above the equator (so the geographic pole is near the top and the equator, the
45 degree parallel and two meridians are drawn in black), the point of
interest (lambda, phi), the geomagnetic pole (lambda_m, phi_m) and the
subsolar point (longitude -pi/12 T_GMT + pi, latitude delta_sun). The blue
great circles are the meridians of the point of interest and of the subsolar
point (they meet at the pole of the longitudes, where the angle t_s is drawn); the
red great circles are the two planes of the caption, through the geomagnetic
pole (where the angle t_m is drawn). This is a schematic: the three positions
(and so every curve) are qualitative, read from the original figure, not
measured; each curve is computed from them, only the visible side is drawn.

To modify: change the angles in GEOMETRY (degrees), the labels in LABELS
(positions in pixels of the original figure, y downwards) or the colours in
COLOURS, then run ``python3 fig27.py`` (matplotlib and numpy needed): it
rewrites fig27.svg (web page) and fig27.pdf (PDF export) next to this file.
Commit the three files together. Keep a dark text colour: the page puts a
white background behind the figure in both site themes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "black": "#000000", "blue": "#0000ff",
           "red": "#ff0000", "dot": "#44546a", "sun": "#c00000"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt

# -- geometry: angles in degrees (longitude, latitude), qualitative ----------
GEOMETRY = {
    "elevation": 45,  # view direction above the equator
    "point_of_interest": (-135, 60),
    "geomagnetic_pole": (-18, 72),
    "subsolar_point": (-115, 2),
    "black_meridians": (-90, -45),  # -90 is the straight vertical line
    "parallel": 45,  # the black parallel (latitude circle)
    "arc_radius": 7,  # angular radius of the arrows t_s and t_m
}
# canvas and label positions in pixels of the original figure (669 x 659, y down)
CENTRE, RADIUS = (333, 333), 331
X_MIN, X_MAX, Y_MIN, Y_MAX = -6, 672, -6, 666
LEADER_END = (345, 458)  # end of the line from "Subsolar point" to the point
# -- labels (mathtext), word for word from the original figure ----------------
LABELS = {
    "poi": ("Point of\ninterest", (120, 146), "left"),
    "lonlat": (r"$(\lambda,\phi)$", (206, 262), "left"),
    "mag": (r"$(\lambda_m,\phi_m)$", (452, 116), "left"),
    "ts": (r"$t_s$", (410, 106), "left"),
    "tm": (r"$t_m$", (478, 182), "left"),
    "sun": ("Subsolar point", (352, 456), "left"),
    "sunpos": (r"$\left(-\dfrac{\pi}{12}T_{GMT}+\pi,\delta_{\odot}\right)$",
               (352, 502), "left"),
}

# -- geometry of the projection ----------------------------------------------
E = np.radians(GEOMETRY["elevation"])
VIEW = np.array([0, -np.cos(E), np.sin(E)])  # towards the viewer
UP = np.array([0, np.sin(E), np.cos(E)])  # screen up


def vec(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def to_px(p):
    """Unit-sphere point -> pixels of the original figure (y downwards)."""
    return (CENTRE[0] + RADIUS * p[..., 0], CENTRE[1] - RADIUS * (p @ UP))


N_POLE = np.array([0.0, 0.0, 1.0])
POI = vec(*GEOMETRY["point_of_interest"])
MAG = vec(*GEOMETRY["geomagnetic_pole"])
SUN = vec(*GEOMETRY["subsolar_point"])
MID = np.radians(GEOMETRY["subsolar_point"][0])

plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MAX, Y_MIN)
ax.axis("off")


def draw_visible(points, colour, lw=0.9, close=False):
    """Plot the part of a curve of unit-sphere points that faces the viewer."""
    vis = points @ VIEW > 0
    x, y = to_px(points)
    x, y = np.where(vis, x, np.nan), np.where(vis, y, np.nan)
    ax.plot(x, y, color=colour, lw=lw, solid_capstyle="butt")


def great_circle(a, b):
    n = np.cross(a, b)
    n /= np.linalg.norm(n)
    u = a / np.linalg.norm(a)
    w = np.cross(n, u)
    t = np.linspace(0, 2 * np.pi, 1441)[:, None]
    return np.cos(t) * u + np.sin(t) * w


def small_circle(lat, n=1441):
    t = np.linspace(0, 2 * np.pi, n)
    la = np.radians(lat)
    return np.stack([np.cos(la) * np.cos(t), np.cos(la) * np.sin(t),
                     np.full_like(t, np.sin(la))], axis=1)


def arc(centre, start, end, rho, colour):
    """Arrow on the small circle of angular radius rho around `centre`, from
    the direction `start` to the direction `end` (tangent vectors), turning
    towards the right of the figure and ending with an arrowhead."""
    c = centre / np.linalg.norm(centre)
    e1 = start - (start @ c) * c
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    g = end - (end @ c) * c
    ang = np.arctan2(g @ e2, g @ e1) % (2 * np.pi)
    best = None
    for sweep in (ang, ang - 2 * np.pi):
        mid = np.cos(sweep / 2) * e1 + np.sin(sweep / 2) * e2
        p = np.cos(rho) * c + np.sin(rho) * mid
        x, y = to_px(p)
        score = x - 0.5 * (CENTRE[1] - y)
        if best is None or score > best[0]:
            best = (score, sweep)
    sweep = best[1]
    t = np.linspace(0, sweep, 200)[:, None]
    pts = np.cos(rho) * c + np.sin(rho) * (np.cos(t) * e1 + np.sin(t) * e2)
    x, y = to_px(pts)
    ax.plot(x[:-3], y[:-3], color=colour, lw=1.0)
    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-4], y[-4]), arrowprops=dict(
        arrowstyle="-|>", color=colour, lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=9))


def toward(frm, to):
    """Tangent direction at `frm` of the great circle going to `to`."""
    return to - (to @ frm) * frm


# silhouette of the sphere, black meridians, equator and parallel
ax.add_patch(plt.Circle(CENTRE, RADIUS, fill=False, edgecolor=COLOURS["black"], lw=2.0))
for lon in GEOMETRY["black_meridians"]:
    draw_visible(great_circle(N_POLE, vec(lon, 0)), COLOURS["black"])
draw_visible(small_circle(0), COLOURS["black"])
draw_visible(small_circle(GEOMETRY["parallel"]), COLOURS["black"])
# blue meridians (point of interest, subsolar point), red planes (geomagnetic pole)
for p in (POI, SUN):
    draw_visible(great_circle(N_POLE, p), COLOURS["blue"])
    draw_visible(great_circle(MAG, p), COLOURS["red"])
# angles t_s (at the geographic pole) and t_m (at the geomagnetic pole)
r = np.radians(GEOMETRY["arc_radius"])
arc(N_POLE, -toward(N_POLE, vec(GEOMETRY["point_of_interest"][0], 0)),
    toward(N_POLE, vec(GEOMETRY["subsolar_point"][0], 0)), r, COLOURS["blue"])
arc(MAG, -toward(MAG, POI), toward(MAG, SUN), r, COLOURS["blue"])
# the three points
for p, c in ((POI, "dot"), (MAG, "dot"), (SUN, "sun")):
    x, y = to_px(p)
    ax.plot([x], [y], "o", ms=4.5, color=COLOURS[c], zorder=5)

# leader from the label to the subsolar point (between the red circle and the equator)
x0, y0 = to_px(SUN)
ax.plot([x0, LEADER_END[0]], [y0, LEADER_END[1]], color=COLOURS["black"], lw=0.8)

for s, (x, y), ha in LABELS.values():
    ax.text(x, y, s, color=COLOURS["text"], ha=ha, va="center", linespacing=1.15)

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
