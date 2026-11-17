"""Fig. 11 — Determining the elevation of one forest point. The picture
represents a lateral section for one sensor azimuth position. Details of the
represented quantities are given in the text description.

ATBD L2 AGB (atbd-l2-agb), section 3.6.4.1.1 (Update reference DEM
coordinates). Redrawn from the original figure and the text of the section:
P0 is the point on the reference DEM, i_z the normal to the ellipsoid at P0,
i_LOS = (P0 - S0)/|P0 - S0| the line of sight from the sensor S0, i_az the
azimuth direction, i_xr = i_az x i_LOS the cross-range vector, i_E = dE(r)/dr
the ellipsoid direction (i_E . i_z = 0), psi the angle between i_xr and i_E,
dz_tomo the tomographic elevation, dxr = dz_tomo / sin(psi) (equation 3.16)
and P1 = P0 + i_xr dxr (equation 3.17); dz is the elevation of P1 versus the
DEM. This is a schematic: positions and curves are qualitative.

To modify: change the positions in GEOMETRY (pixels of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 fig11.py`` (matplotlib needed): it rewrites
fig11.svg (web page) and fig11.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour: the page puts a white
background behind the figure in both site themes.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "blue": "#4472c4", "dark": "#1f3864",
           "point": "#ff0000"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt

# -- geometry (pixels of the original figure, y downwards) -------------------
W, H = 790, 740
S0, P0, P1 = (115, 48), (335, 492), (471, 428)
GEOMETRY = {
    "los_head": (224, 268),                    # tip of i_LOS, exactly on the line S0-P0
    "az_start": (150, 118),                    # start of the thick i_LOS arrow on S0-P0
    "iz": (350, 405), "ixr": (527, 400), "iE": (532, 522),
    "range": ((265, 95), (397, 117)),
    "dxr": ((272, 362), (403, 292)),           # dimension arrow
    "xr_dash": ((403, 290), (466, 426)),       # dashed line from dxr end to P1
    "p1_down": ((466, 426), (443, 607)),       # dashed line from P1 to the DEM
    "tomo_x": 563, "dz_x": 665,                # x of the two vertical dimensions
    "tomo": ((570, 443), (556, 525)), "dz": ((674, 460), (646, 637)),
    "dash_tomo": (((471, 428), (692, 460)), ((335, 492), (580, 527))),
    "dash_dz": (((443, 607), (700, 640)),),
    # qualitative curves: DEM (solid) and ellipsoid E(r) (dash-dot)
    "dem": [(30, 408), (130, 424), (250, 456), (335, 492), (395, 540),
            (440, 600), (470, 660), (497, 716)],
    "ell": [(0, 514), (90, 494), (180, 488), (270, 489), (335, 492),
            (430, 525), (520, 560), (585, 587)],
    "psi_r": 55,
}

# -- labels (mathtext) --------------------------------------------------------
LABELS = {
    "S0": r"$S_0$", "iaz": r"$\hat{\imath}_{az}$", "ilos": r"$\hat{\imath}_{LOS}$",
    "iz": r"$\hat{\imath}_z$", "ixr": r"$\hat{\imath}_{xr}$",
    "iE": r"$\hat{\imath}_E$", "range": "Range", "dxr": r"$\delta xr$",
    "P0": r"$P_0$", "P1": r"$P_1$", "psi": r"$\psi$", "dem": "DEM",
    "Er": r"$E(r)$", "ell": "Ellipsoid", "tomo": r"$\delta z_{tomo}$",
    "dz": r"$\delta z$",
}
# label positions (pixels, centre of the text)
LABEL_AT = {
    "S0": (78, 52), "iaz": (153, 30), "ilos": (258, 240), "iz": (368, 405),
    "ixr": (546, 417), "iE": (560, 548), "range": (307, 78), "dxr": (332, 316),
    "P0": (335, 524), "P1": (479, 399), "psi": (395, 492), "dem": (160, 468),
    "Er": (232, 513), "tomo": (610, 490), "dz": (692, 562),
}
ELL_AT, ELL_ROT = (548, 664), -10  # "Ellipsoid", right of the DEM curve, touching nothing
LABEL_SIZE = {"dem": 1.35, "ell": 1.15, "range": 1.15}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
BLUE, THIN = COLOURS["blue"], 0.9


def arrow(p, q, lw=1.0, both=False, color=BLUE):
    style = "<|-|>" if both else "-|>"
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle=f"{style},head_length=0.45,head_width=0.18", color=color,
        lw=lw, shrinkA=0, shrinkB=0))


def line(p, q, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=kw.pop("color", BLUE),
            lw=kw.pop("lw", THIN), **kw)


def text(key, **kw):
    size = LABEL_SIZE.get(key, 1.25 if LABELS[key].startswith("$") else 1.0) * FONT_PT
    ax.text(*LABEL_AT[key], LABELS[key], color=COLOURS["text"], ha="center",
            va="center", fontsize=size, **kw)


g = GEOMETRY
# curves
for key, ls, lw in (("dem", "-", 1.8), ("ell", "-.", 1.5)):
    pts = np.array(g[key], float)
    t = np.linspace(0, 1, len(pts))
    tt = np.linspace(0, 1, 200)
    # smooth quadratic-spline-like curve through the points
    from numpy.polynomial import polynomial as poly  # noqa: E402
    cx = poly.polyfit(t, pts[:, 0], 5)
    cy = poly.polyfit(t, pts[:, 1], 5)
    ax.plot(poly.polyval(tt, cx), poly.polyval(tt, cy), color=COLOURS["dark"],
            lw=lw, ls=ls)
# line of sight S0 -> P0, with i_az and i_LOS arrowheads
line(S0, P0)
ax.plot(*zip(S0, S0, strict=True), marker="x", color=BLUE, ms=9, mew=1.5)
arrow(g["az_start"], g["los_head"], lw=1.8)
# unit vectors at P0
arrow(P0, g["iz"])
arrow(P0, g["ixr"])
arrow(P0, g["iE"])
arrow(*g["range"])
# psi
ang_xr = np.degrees(np.arctan2(P1[1] - P0[1], P1[0] - P0[0]))
ang_E = np.degrees(np.arctan2(g["iE"][1] - P0[1], g["iE"][0] - P0[0]))
ax.add_patch(Arc(P0, 2 * g["psi_r"], 2 * g["psi_r"], theta1=ang_xr, theta2=ang_E,
                 color=BLUE, lw=THIN))
# dashed construction lines and dimension arrows
for p, q in (g["xr_dash"], g["p1_down"], *g["dash_tomo"], *g["dash_dz"]):
    line(p, q, ls=(0, (3, 2)), lw=0.7)
arrow(*g["dxr"], both=True)
arrow(*g["tomo"], both=True)
arrow(*g["dz"], both=True)
# points
for p in (P0, P1):
    ax.plot(*p, "o", color=COLOURS["point"], ms=8, zorder=5)
for key in LABEL_AT:
    text(key)
ax.text(ELL_AT[0], ELL_AT[1], LABELS["ell"], color=COLOURS["text"], ha="center",
        va="center", fontsize=FONT_PT * LABEL_SIZE["ell"], rotation=ELL_ROT)

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
