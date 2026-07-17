#!/usr/bin/env python3
"""Generate QSM#3 Open Source readiness slides (2 acts - ~26 slides)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "source" / "slides"
DECK_YAML = ROOT / "source" / "deck.yaml"

TAGLINE = "BIOMASS DISC - QSM#3 - 17/07/2026"
LOGOS = (
    '<img src="../_shared/logos/esa-dark.png" style="height:26px;object-fit:contain;opacity:.6;">'
    '<img src="../_shared/logos/acri-st.png" style="height:24px;object-fit:contain;opacity:.85;">'
    '<img src="../_shared/logos/biomass-disc.png" style="height:28px;object-fit:contain;opacity:.75;">'
)
BIO_LOGO = '<img src="../_shared/logos/biopal-text-right.png" alt="BioPAL" style="height:48px;object-fit:contain">'


def footer(num: str) -> str:
    return (
        f'<div class="slide-footer">'
        f'<span class="slide-footer__tagline">{TAGLINE}</span>'
        f'<div class="slide-footer__logos">{LOGOS}</div>'
        f'<span class="slide-footer__number">{num}</span>'
        f"</div>"
    )


def ribbon(label: str) -> str:
    return f"""\
  <div class="slide-ribbon">
    <div class="slide-ribbon__accent"><div class="slide-ribbon__progress" style="width:0%"></div></div>
    <div class="slide-ribbon__bar">
      <div class="slide-ribbon__label">{label}</div>
      <div class="slide-ribbon__logo">{BIO_LOGO}</div>
    </div>
  </div>"""


def bg() -> str:
    return (
        '<div style="position:absolute;inset:0;background:radial-gradient(ellipse at 68% 0%,rgba(0,122,148,.22) 0%,transparent 52%);"></div>'
        '<div style="position:absolute;inset:0;background:radial-gradient(ellipse at 5% 100%,rgba(151,192,11,.08) 0%,transparent 45%);"></div>'
    )


def bullets(items: list[str], size: int = 38) -> str:
    lis = "".join(
        f'<li style="margin:0 0 22px;line-height:1.35;">{item}</li>' for item in items
    )
    return f'<ul style="margin:0;padding-left:44px;font-size:{size}px;color:rgba(255,255,255,.88);">{lis}</ul>'


def title_block(title: str, subtitle: str | None = None) -> str:
    sub = ""
    if subtitle:
        sub = f'<div style="font-size:32px;color:rgba(255,255,255,.5);margin-top:14px;line-height:1.4;">{subtitle}</div>'
    return (
        f'<div style="font-weight:700;font-size:64px;color:#fff;line-height:1.1;letter-spacing:-.01em;">{title}</div>'
        f'<div style="width:80px;height:4px;background:var(--gr);border-radius:2px;margin:20px 0 32px;"></div>'
        f"{sub}"
    )


def content_slide(
    num: str,
    screen: str,
    act_class: str,
    ribbon_label: str,
    title: str,
    body: str,
) -> str:
    return f"""\
<section class="slide {act_class}" data-label="{num}" data-screen-label="{screen}" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
{ribbon(ribbon_label)}
  <div class="slide-body slide-body--stack" style="z-index:1;">
    <div style="display:flex;flex-direction:column;gap:16px;width:100%;">
      {title_block(title)}
      {body}
    </div>
  </div>
  {footer(num)}
</section>
"""


def chapter_slide(num: str, numeral: str, title: str, line: str, act_class: str) -> str:
    return f"""\
<section class="slide slide--chapter {act_class}" data-label="{num}" data-screen-label="ACT {numeral}" data-deck-chapter="" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1;padding-bottom:40px;">
    <div style="display:flex;flex-direction:column;align-items:center;gap:24px;text-align:center;max-width:1500px;">
      <div class="chapter-numeral" style="font-family:var(--fm);font-weight:700;font-size:220px;background:linear-gradient(135deg,var(--tl) 0%,var(--cy) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:.9;">{numeral}</div>
      <div class="chapter-rule" style="width:140px;height:6px;background:var(--gr);border-radius:3px;"></div>
      <div class="chapter-title" style="font-weight:700;font-size:52px;color:#fff;letter-spacing:.06em;text-transform:uppercase;">{title}</div>
      <div style="font-size:34px;color:rgba(255,255,255,.5);line-height:1.45;max-width:1000px;">{line}</div>
    </div>
  </div>
  {footer(num)}
</section>
"""


def card(title: str, body: str, accent: str = "var(--tl)") -> str:
    return f"""\
<div style="flex:1;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-top:4px solid {accent};border-radius:10px;padding:32px 28px;">
  <div style="font-weight:700;font-size:36px;color:#fff;margin-bottom:16px;line-height:1.2;">{title}</div>
  <div style="font-size:30px;color:rgba(255,255,255,.65);line-height:1.45;">{body}</div>
</div>"""


def browser_shot(url: str, img: str, height: int = 620) -> str:
    """Static browser chrome + screenshot. No interactive hooks."""
    return f"""\
<div style="width:100%;background:#152433;border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.1);box-shadow:0 8px 48px rgba(0,0,0,.4);">
  <div style="height:40px;background:#0d1d2b;display:flex;align-items:center;padding:0 16px;gap:8px;border-bottom:1px solid rgba(255,255,255,.08);">
    <div style="width:11px;height:11px;border-radius:50%;background:var(--pk);"></div>
    <div style="width:11px;height:11px;border-radius:50%;background:#FFD066;"></div>
    <div style="width:11px;height:11px;border-radius:50%;background:var(--gr);"></div>
    <div style="flex:1;background:rgba(255,255,255,.07);border-radius:4px;height:22px;margin:0 12px;display:flex;align-items:center;padding:0 12px;">
      <span style="font-family:var(--fm);font-size:15px;color:rgba(255,255,255,.45);">{url}</span>
    </div>
  </div>
  <div style="height:{height}px;overflow:hidden;">
    <img src="assets/screenshots/{img}" alt="" style="width:100%;height:auto;display:block;object-fit:cover;object-position:top;">
  </div>
</div>"""


def split_shot(
    num: str,
    screen: str,
    act_class: str,
    ribbon_label: str,
    eyebrow: str,
    title: str,
    left_body: str,
    url: str,
    img: str,
) -> str:
    return f"""\
<section class="slide {act_class}" data-label="{num}" data-screen-label="{screen}" style="font-family:var(--fs);background:var(--bg);color:#fff;">
  {bg()}
{ribbon(ribbon_label)}
  <div class="slide-body slide-body--split" style="z-index:1;">
    <div style="flex:0 0 38%;display:flex;flex-direction:column;justify-content:center;">
      <div class="slide-eyebrow">{eyebrow}</div>
      <h2 class="slide-title">{title}</h2>
      <div class="slide-rule"></div>
      {left_body}
    </div>
    <div class="slide-col--grow">{browser_shot(url, img)}</div>
  </div>
  {footer(num)}
</section>
"""


def task_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    col_widths: list[str] | None = None,
    font_size: int = 26,
    cell_padding: str = "14px 14px",
    head_size: int = 20,
) -> str:
    """Reusable SoW table. rows are lists of cell HTML."""
    n = len(headers)
    widths = col_widths or [f"{100 // n}%"] * n
    head = "".join(
        f'<th style="padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.15);'
        f'width:{widths[i]};text-align:left;font-size:{head_size}px;letter-spacing:.06em;'
        f'text-transform:uppercase;color:rgba(255,255,255,.45);font-weight:600;">{h}</th>'
        for i, h in enumerate(headers)
    )
    body = ""
    for row in rows:
        cells = "".join(
            f'<td style="padding:{cell_padding};border-bottom:1px solid rgba(255,255,255,.08);'
            f'color:rgba(255,255,255,.88);line-height:1.3;vertical-align:top;">{c}</td>'
            for c in row
        )
        body += f"<tr>{cells}</tr>"
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:{font_size}px;">'
        f"<tr>{head}</tr>{body}</table>"
    )


SLIDE_DEFS: list[dict] = []


def register(entry: dict, html: str) -> None:
    SLIDE_DEFS.append({**entry, "html": html})


# ===================================================================
# 01 - Title
# ===================================================================

register(
    {"id": "01", "file": "slides/01-title.html", "title": "Title", "act": "I"},
    f"""\
<section class="slide slide--act-i" data-label="01" data-screen-label="Title" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
  <div style="position:absolute;top:0;left:0;width:58%;height:100%;display:flex;flex-direction:column;justify-content:center;padding:100px 80px;box-sizing:border-box;z-index:1;">
    <img src="../_shared/logos/esa-dark.png" style="height:120px;object-fit:contain;opacity:.7;width:auto;max-width:360px;margin-bottom:36px;">
    <div style="font-weight:600;font-size:24px;letter-spacing:.1em;text-transform:uppercase;color:rgba(0,182,240,.85);margin-bottom:12px;">BIOMASS DISC - Quarterly Service Meeting #3</div>
    <div style="font-weight:700;font-size:102px;color:#fff;line-height:.92;letter-spacing:-.02em;">Open Source</div>
    <div style="font-weight:700;font-size:90px;color:var(--gr);line-height:.92;letter-spacing:-.02em;">Task 12 User Support</div>
    <div style="width:140px;height:6px;background:var(--tl);margin:32px 0 28px;border-radius:2px;"></div>
    <div style="font-size:36px;color:rgba(255,255,255,.7);line-height:1.4;">Near-term target & work remaining</div>
  </div>
  <div style="position:absolute;top:0;right:0;width:42%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:1;">
    <img src="../_shared/logos/biopal-text-bottom.png" alt="BioPAL" style="height:300px;object-fit:contain;">
  </div>
  <div style="position:absolute;bottom:0;left:0;right:0;height:68px;background:rgba(255,255,255,.04);border-top:1px solid rgba(255,255,255,.07);display:flex;align-items:center;padding:0 72px;z-index:2;box-sizing:border-box;">
    <div style="flex:0 0 auto;font-size:26px;color:rgba(255,255,255,.45);line-height:1.4;margin-right:40px;"><div>Yoann Rey-Ricord</div><div style="opacity:.7;">ACRI-ST - 17 July 2026</div></div>
    <div style="flex:1;display:flex;align-items:center;justify-content:center;gap:24px;">{LOGOS}</div>
    <span style="font-family:var(--fm);font-size:26px;color:rgba(255,255,255,.3);margin-left:auto;">01</span>
  </div>
</section>
""",
)

# ===================================================================
# 02 - Agenda (2 acts)
# ===================================================================

agenda_cards = [
    ("I", "Where we stand", "Timeline, website, contributions, CI pipeline,<br>repository, governance, then live docs demo"),
    ("II", "Work remaining", "Near-term target, three work tracks,<br>summary and discussion"),
]
agenda_html = "".join(
    f"""\
      <div style="flex:1;border-radius:12px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;padding:44px 30px;text-align:center;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);">
        <div style="font-family:var(--fm);font-weight:700;font-size:110px;background:linear-gradient(135deg,var(--tl) 0%,var(--cy) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:.9;">{n}</div>
        <div style="width:72px;height:5px;background:var(--gr);border-radius:2px;"></div>
        <div style="font-weight:700;font-size:32px;color:#fff;letter-spacing:.04em;text-transform:uppercase;line-height:1.2;">{t}</div>
        <div style="font-size:26px;color:rgba(255,255,255,.55);line-height:1.5;max-width:420px;">{b}</div>
      </div>"""
    for n, t, b in agenda_cards
)
register(
    {"id": "02", "file": "slides/02-todays-plan.html", "title": "Agenda", "act": "I"},
    f"""\
<section class="slide slide--chapter slide--agenda" data-label="02" data-screen-label="Agenda" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
  <div style="position:absolute;inset:0;bottom:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:1;padding:48px 72px 32px;box-sizing:border-box;">
    <div style="text-align:center;margin-bottom:44px;">
      <div style="font-weight:700;font-size:56px;color:#fff;letter-spacing:.06em;text-transform:uppercase;">Agenda</div>
      <div style="width:120px;height:5px;background:var(--gr);border-radius:3px;margin:20px auto;"></div>
    </div>
    <div style="display:flex;gap:36px;width:100%;max-width:1400px;align-items:stretch;">{agenda_html}</div>
  </div>
  {footer("02")}
</section>
""",
)

# ===================================================================
# Act I - Where we stand (03-16)
# ===================================================================

register(
    {"id": "03", "file": "slides/03-act-i.html", "title": "ACT I", "act": "chapter", "kind": "chapter"},
    chapter_slide(
        "03",
        "I",
        "Where we stand",
        "Timeline, deliverables, and how the open programme works today",
        "slide--act-i",
    ),
)

# 04 - Progress timeline
# Suite versions from commits by riccardo.piantanida / Matteo Aletti (release only, no .dev)
timeline_items = [
    (
        "Dec 2025",
        "Public GitHub home under BioPAL",
        [
            "Repo opened at github.com/BioPAL/BPS",
            "<strong style=\"color:#fff;\">v4.2.2</strong> · 11 Dec (first public release)",
            "Full processor suite readable by everyone",
            "Apache 2.0 declared in the repository",
            "Discussions available for questions",
        ],
    ),
    (
        "Jan - May 2026",
        "Suite releases and programme design",
        [
            "<strong style=\"color:#fff;\">v4.3.0 → v4.4.4</strong> on the public repo",
            "Documentation scope and limits defined",
            "Governance and project structure drafted",
            "Five issue forms: bug, feature, algorithm, docs, security",
            "Pull request template aligned with approval workflow",
            "First documentation online in Markdown format",
            "Licence and code remain public throughout",
        ],
    ),
    (
        "June 2026",
        "Contribution, documentation, community",
        [
            "<strong style=\"color:#fff;\">v4.4.5</strong> · 10 June",
            "Blank issues blocked; ask first, then implement",
            "Good-first-issue welcome path for newcomers",
            "Sphinx site live on biomass-disc.info/docs",
            "Guides, contributing, governance, and science pages",
            "Local tutorial + AGB algorithm page on the web",
            "First Developer Meeting held (30 June)",
        ],
    ),
    (
        "July 2026",
        "Foundations prepared",
        [
            "CODEOWNERS, REUSE, and licence texts ready (#24)",
            "Automated checks designed for every proposed change (#30)",
            "Website sources prepared to live with the code (#35)",
            "Inline SPDX headers prepared (#36)",
            "This Statement of work: what remains for near-term readiness",
        ],
    ),
]
timeline_html = "".join(
    f"""\
<div style="flex:1;display:flex;flex-direction:column;gap:14px;min-width:0;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:22px 18px;">
  <div style="font-family:var(--fm);font-weight:700;font-size:26px;color:var(--cy);letter-spacing:.04em;">{when}</div>
  <div style="height:5px;background:rgba(255,255,255,.1);border-radius:2px;position:relative;">
    <div style="position:absolute;left:0;top:50%;transform:translateY(-50%);width:14px;height:14px;border-radius:50%;background:var(--gr);"></div>
  </div>
  <div style="font-weight:700;font-size:28px;color:#fff;line-height:1.2;min-height:68px;">{title}</div>
  <div style="display:flex;flex-direction:column;gap:10px;margin-top:2px;">
    {"".join(
        f'<div style="display:flex;gap:12px;align-items:flex-start;">'
        f'<div style="width:8px;height:8px;border-radius:50%;background:var(--gr);flex-shrink:0;margin-top:10px;"></div>'
        f'<div style="font-size:24px;color:rgba(255,255,255,.75);line-height:1.3;">{item}</div>'
        f"</div>"
        for item in items
    )}
  </div>
</div>"""
    for when, title, items in timeline_items
)
register(
    {"id": "04", "file": "slides/04-progress-timeline.html", "title": "Progress to date", "act": "I"},
    f"""\
<section class="slide slide--act-i" data-label="04" data-screen-label="Progress to date" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
{ribbon("ACT I - TIMELINE")}
  <div class="slide-body slide-body--stack" style="z-index:1;">
    <div style="display:flex;flex-direction:column;gap:12px;width:100%;">
      <div style="font-weight:700;font-size:56px;color:#fff;line-height:1.1;letter-spacing:-.01em;">Progress to date</div>
      <div style="width:80px;height:4px;background:var(--gr);border-radius:2px;margin:4px 0 16px;"></div>
      <div style="display:flex;gap:16px;align-items:stretch;width:100%;">{timeline_html}</div>
    </div>
  </div>
  {footer("04")}
</section>
""",
)

# 05 - Documentation website
register(
    {"id": "05", "file": "slides/05-docs-portal.html", "title": "Documentation website", "act": "I"},
    split_shot(
        "05",
        "Documentation website",
        "slide--act-i",
        "ACT I - DOCUMENTATION",
        "DOCUMENTATION",
        "Documentation website",
        '<div style="display:flex;flex-direction:column;gap:18px;margin-bottom:24px;">'
        '<div style="display:flex;gap:14px;align-items:center;"><span style="font-family:var(--fm);font-weight:700;font-size:36px;color:var(--tl);">50+</span><span style="font-size:34px;color:rgba(255,255,255,.7);">pages</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;"><span style="font-family:var(--fm);font-weight:700;font-size:36px;color:var(--tl);">8</span><span style="font-size:34px;color:rgba(255,255,255,.7);">main sections</span></div>'
        '<div style="font-size:32px;color:rgba(255,255,255,.65);line-height:1.4;margin-top:8px;">Start here - User guide - Science - How to contribute - Who decides</div>'
        '<div style="font-size:30px;color:var(--cy);font-family:var(--fm);margin-top:12px;">biomass-disc.info/docs/</div>'
        "</div>",
        "biomass-disc.info/docs/",
        "docs-portal-home.png",
    ),
)

# 06 - Get Started
register(
    {"id": "06", "file": "slides/06-get-started.html", "title": "Get Started", "act": "I"},
    content_slide(
        "06",
        "Get Started",
        "slide--act-i",
        "ACT I - GET STARTED",
        "Get Started",
        '<div style="font-size:30px;color:rgba(255,255,255,.6);margin-bottom:24px;line-height:1.45;">'
        "The front door of the documentation. Points every visitor to the right resources in less than thirty seconds."
        "</div>"
        '<div style="display:flex;gap:24px;align-items:stretch;margin-bottom:24px;">'
        + card(
            '<i class="fa-solid fa-book-open" style="color:var(--tl);margin-right:10px;"></i>Use BPS',
            "User Guide, Science Guide, Tutorials, official PDF documents.",
            "var(--tl)",
        )
        + card(
            '<i class="fa-regular fa-comments" style="color:var(--cy);margin-right:10px;"></i>Ask or discuss',
            "Q&amp;A, ideas, scientific discussions, show and tell. No coding needed.",
            "var(--cy)",
        )
        + card(
            '<i class="fa-solid fa-code-merge" style="color:var(--gr);margin-right:10px;"></i>Propose a change',
            "Open or pick an issue. Wait for approval. Implement. Open a PR. Review.",
            "var(--gr)",
        )
        + "</div>"
        + '<div style="background:rgba(0,122,148,.1);border:1px solid rgba(0,122,148,.3);border-left:4px solid var(--tl);border-radius:8px;padding:18px 24px;font-size:28px;color:rgba(255,255,255,.85);line-height:1.45;">'
        '<strong style="color:var(--cy);">biomass-disc.info/docs/getting-started/</strong>'
        "</div>",
    ),
)

# 07 - User Guide
register(
    {"id": "07", "file": "slides/07-user-guide.html", "title": "User Guide", "act": "I"},
    split_shot(
        "07",
        "User Guide",
        "slide--act-i",
        "ACT I - USER GUIDE",
        "USER GUIDE",
        "User Guide",
        '<div style="display:flex;flex-direction:column;gap:22px;margin-bottom:24px;">'
        '<div style="font-size:34px;color:rgba(255,255,255,.7);line-height:1.45;">The authoritative user-facing reference for BPS is the official Software User Manual (SUM).</div>'
        '<div style="background:rgba(255,255,255,.05);border-left:4px solid var(--tl);border-radius:8px;padding:20px 22px;">'
        '<div style="font-weight:700;font-size:30px;color:#fff;margin-bottom:8px;">BIO-BPS-SUM-ARE-010479</div>'
        '<div style="font-size:26px;color:rgba(255,255,255,.55);">Version 4.4.1 - 13 March 2025</div>'
        '<div style="font-size:26px;color:var(--cy);margin-top:10px;">PDF download on the ESA dissemination portal</div>'
        "</div>"
        '<div style="font-size:28px;color:rgba(255,255,255,.5);line-height:1.4;">Full list of applicable documents (ATBDs, ICD, IODD, auxiliary formats) available under About.</div>'
        "</div>",
        "biomass-disc.info/docs/user-guide/",
        "docs-user-guide.png",
    ),
)

# 08 - Tutorials
register(
    {"id": "08", "file": "slides/08-tutorials.html", "title": "Tutorials", "act": "I"},
    split_shot(
        "08",
        "Tutorials",
        "slide--act-i",
        "ACT I - TUTORIALS",
        "TUTORIALS",
        "Tutorials",
        '<div style="font-size:34px;color:rgba(255,255,255,.7);line-height:1.45;margin-bottom:24px;">Hands-on walkthroughs. Processing chain from Level-1 framed data to Level-2A.</div>'
        '<div style="display:flex;flex-direction:column;gap:16px;">'
        '<div style="background:rgba(255,255,255,.05);border-left:4px solid var(--tl);border-radius:8px;padding:18px 20px;"><div style="font-weight:700;font-size:34px;color:#fff;">Run BPS on your own computer</div><div style="font-size:28px;color:rgba(255,255,255,.55);">Bundle install or build from source (developer)</div></div>'
        '<div style="background:rgba(255,255,255,.05);border-left:4px solid var(--cy);border-radius:8px;padding:18px 20px;"><div style="font-weight:700;font-size:34px;color:#fff;">Run BPS on ESA MAAP</div><div style="font-size:28px;color:rgba(255,255,255,.55);">Cloud JupyterLab notebooks (ESA Member State eligibility)</div></div>'
        '<div style="background:rgba(255,255,255,.05);border-left:4px solid var(--gr);border-radius:8px;padding:18px 20px;"><div style="font-weight:700;font-size:34px;color:#fff;">Companion scripts and notebooks</div><div style="font-size:28px;color:rgba(255,255,255,.55);">JobOrder templates, helper scripts, runnable notebooks</div></div>'
        "</div>",
        "biomass-disc.info/docs/tutorials/",
        "docs-tutorial.png",
    ),
)

# 09 - Science Guide
register(
    {"id": "09", "file": "slides/09-science-guide.html", "title": "Science Guide", "act": "I"},
    split_shot(
        "09",
        "Science Guide",
        "slide--act-i",
        "ACT I - SCIENCE GUIDE",
        "SCIENCE GUIDE",
        "Science Guide",
        '<div style="display:flex;flex-direction:column;gap:18px;margin-bottom:24px;">'
        '<div style="font-size:34px;color:rgba(255,255,255,.7);line-height:1.4;">Catalogue of ATBDs and Product Format Documents for every BPS processor.</div>'
        '<div style="background:rgba(255,255,255,.05);border-left:4px solid var(--gr);border-radius:8px;padding:18px 20px;">'
        '<div style="font-weight:700;font-size:30px;color:#fff;margin-bottom:6px;">L2 Above-Ground Biomass (AGB)</div>'
        '<div style="font-size:26px;color:rgba(255,255,255,.55);">First full web conversion live</div>'
        "</div>"
        '<div style="font-size:28px;color:rgba(255,255,255,.6);line-height:1.4;">Forest Height, Forest Disturbance, Level-1: next conversions</div>'
        "</div>"
        '<div style="background:rgba(255,208,102,.1);border:1px solid rgba(255,208,102,.35);border-left:4px solid #FFD066;border-radius:8px;padding:18px 20px;font-size:28px;color:rgba(255,255,255,.85);line-height:1.45;">Official PDFs remain the reference until each web page is approved by science experts.</div>',
        "biomass-disc.info/docs/science-guide/",
        "docs-science-guide.png",
    ),
)

# 10 - Communication
register(
    {"id": "10", "file": "slides/10-communication.html", "title": "Communication", "act": "I"},
    content_slide(
        "10",
        "Communication",
        "slide--act-i",
        "ACT I - COMMUNICATION",
        "Communication",
        '<div style="font-size:30px;color:rgba(255,255,255,.6);margin-bottom:20px;line-height:1.45;">'
        "How the BioPAL community communicates, collaborates, and works together."
        "</div>"
        '<div style="display:flex;gap:24px;align-items:stretch;margin-bottom:24px;">'
        + card(
            '<i class="fa-solid fa-comments" style="color:var(--cy);margin-right:10px;"></i>Channels',
            '<div style="display:flex;flex-direction:column;gap:10px;">'
            '<div style="display:flex;gap:12px;align-items:center;"><i class="fa-solid fa-circle-exclamation" style="color:var(--pk);width:24px;text-align:center;font-size:20px;"></i><span>GitHub Issues: actionable items</span></div>'
            '<div style="display:flex;gap:12px;align-items:center;"><i class="fa-regular fa-comment-dots" style="color:var(--cy);width:24px;text-align:center;font-size:20px;"></i><span>Discussions: Q&amp;A, ideas, science, show&amp;tell</span></div>'
            '<div style="display:flex;gap:12px;align-items:center;"><i class="fa-regular fa-envelope" style="color:var(--tl);width:24px;text-align:center;font-size:20px;"></i><span>Email for private matters</span></div>'
            "</div>",
            "var(--cy)",
        )
        + card(
            '<i class="fa-solid fa-calendar" style="color:var(--tl);margin-right:10px;"></i>Meetings and community',
            '<div style="display:flex;flex-direction:column;gap:10px;">'
            '<div style="display:flex;gap:12px;align-items:center;"><i class="fa-solid fa-users" style="color:var(--gr);width:24px;text-align:center;font-size:20px;"></i><span>Developer meetings</span></div>'
            '<div style="display:flex;gap:12px;align-items:center;"><i class="fa-solid fa-chalkboard" style="color:var(--tl);width:24px;text-align:center;font-size:20px;"></i><span>Session slides published</span></div>'
            '<div style="display:flex;gap:12px;align-items:center;"><i class="fa-solid fa-scale-balanced" style="color:var(--pk);width:24px;text-align:center;font-size:20px;"></i><span>Conflict resolution process</span></div>'
            "</div>",
            "var(--tl)",
        )
        + "</div>"
        + '<div style="background:rgba(151,192,11,.08);border:1px solid rgba(151,192,11,.25);border-left:4px solid var(--gr);border-radius:8px;padding:18px 24px;font-size:28px;color:rgba(255,255,255,.85);line-height:1.45;">'
        "Code of Conduct and getting-help guides ensure a respectful environment."
        "</div>",
    ),
)

# 11 - Contributing
register(
    {"id": "11", "file": "slides/11-contributing.html", "title": "Contributing", "act": "I"},
    split_shot(
        "11",
        "Contributing",
        "slide--act-i",
        "ACT I - CONTRIBUTING",
        "CONTRIBUTING",
        "Contributing",
        '<div style="font-size:30px;color:rgba(255,255,255,.65);margin-bottom:20px;line-height:1.45;">Every contribution follows the same five steps. No code before approval.</div>'
        '<div style="display:flex;flex-direction:column;gap:12px;margin-bottom:18px;">'
        '<div style="display:flex;gap:14px;align-items:center;background:rgba(151,192,11,.08);border:1px solid rgba(151,192,11,.25);border-radius:8px;padding:12px 16px;">'
        '<span style="font-family:var(--fm);font-weight:700;font-size:24px;color:var(--gr);min-width:32px;">1</span>'
        '<span style="font-size:28px;color:rgba(255,255,255,.85);">Open an issue (5 templates)</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;background:rgba(0,122,148,.08);border:1px solid rgba(0,122,148,.25);border-radius:8px;padding:12px 16px;">'
        '<span style="font-family:var(--fm);font-weight:700;font-size:24px;color:var(--tl);min-width:32px;">2</span>'
        '<span style="font-size:28px;color:rgba(255,255,255,.85);">Triage by maintainer</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;background:rgba(255,208,102,.08);border:1px solid rgba(255,208,102,.25);border-radius:8px;padding:12px 16px;">'
        '<span style="font-family:var(--fm);font-weight:700;font-size:24px;color:#FFD066;min-width:32px;">3</span>'
        '<span style="font-size:28px;color:rgba(255,255,255,.85);">Approval gate (On Going)</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;background:rgba(0,182,240,.08);border:1px solid rgba(0,182,240,.25);border-radius:8px;padding:12px 16px;">'
        '<span style="font-family:var(--fm);font-weight:700;font-size:24px;color:var(--cy);min-width:32px;">4</span>'
        '<span style="font-size:28px;color:rgba(255,255,255,.85);">Fork, implement, open a PR</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;background:rgba(255,126,121,.08);border:1px solid rgba(255,126,121,.25);border-radius:8px;padding:12px 16px;">'
        '<span style="font-family:var(--fm);font-weight:700;font-size:24px;color:var(--pk);min-width:32px;">5</span>'
        '<span style="font-size:28px;color:rgba(255,255,255,.85);">CODEOWNERS review, CI green, squash merge</span></div>'
        "</div>"
        '<div style="font-size:24px;color:rgba(255,255,255,.45);font-style:italic;">Scientists can propose algorithm or docs changes without writing code.</div>',
        "biomass-disc.info/docs/contributing/",
        "docs-contributing.png",
    ),
)

# 12 - CI/CD pipeline design (right after Contributing)
ci_tiers = [
    ("Baseline", "var(--gr)", "Lint, licence headers, SPDX check, unit tests", "Every push and pull request", "5 min"),
    ("Extended", "var(--tl)", "Integration tests, build verification, doc generation", "Pull requests to shared branch", "20 min"),
    ("Heavy", "var(--cy)", "Full processing chain, regression tests, performance", "Nightly or on-demand", "2 hours"),
]
ci_tiers_html = "".join(
    f"""\
<div style="flex:1;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-top:4px solid {color};border-radius:10px;padding:24px 22px;display:flex;flex-direction:column;gap:14px;">
  <div style="font-weight:700;font-size:36px;color:#fff;">{name}</div>
  <div style="font-size:26px;color:rgba(255,255,255,.65);line-height:1.4;">{checks}</div>
  <div style="margin-top:auto;display:flex;flex-direction:column;gap:6px;">
    <div style="font-size:22px;color:rgba(255,255,255,.45);"><strong style="color:rgba(255,255,255,.6);">Trigger:</strong> {trigger}</div>
    <div style="font-size:22px;color:rgba(255,255,255,.45);"><strong style="color:rgba(255,255,255,.6);">Maximum duration:</strong> {max_time}</div>
  </div>
</div>"""
    for name, color, checks, trigger, max_time in ci_tiers
)
register(
    {"id": "12", "file": "slides/12-ci-pipeline.html", "title": "CI/CD pipeline design", "act": "I"},
    content_slide(
        "12",
        "CI/CD pipeline design",
        "slide--act-i",
        "ACT I - CI/CD",
        "CI/CD pipeline design",
        '<div style="font-size:28px;color:rgba(255,255,255,.55);margin-bottom:20px;line-height:1.4;">'
        "Three-tier system. Each proposed change is classified automatically based on what it touches. The base branch determines which checks run."
        "</div>"
        + f'<div style="display:flex;gap:22px;margin-bottom:24px;">{ci_tiers_html}</div>'
        + '<div style="display:flex;gap:28px;align-items:center;">'
        '<div style="background:rgba(151,192,11,.1);border:1px solid rgba(151,192,11,.3);border-radius:8px;padding:14px 20px;font-size:26px;color:rgba(255,255,255,.85);line-height:1.4;">'
        '<strong style="color:var(--gr);">tier-policy.yml</strong> defines the classification rules. Auto-classification from changed file paths.'
        '</div>'
        '<div style="background:rgba(0,182,240,.08);border:1px solid rgba(0,182,240,.25);border-radius:8px;padding:14px 20px;font-size:26px;color:rgba(255,255,255,.85);line-height:1.4;">'
        '<strong style="color:var(--cy);">PR #30</strong> implements the pipeline. Checks still failing: needs green before acceptance.'
        '</div>'
        "</div>",
    ),
)

# 13 - Public repository
register(
    {"id": "13", "file": "slides/13-public-home.html", "title": "Public repository", "act": "I"},
    split_shot(
        "13",
        "Public repository",
        "slide--act-i",
        "ACT I - REPOSITORY",
        "REPOSITORY",
        "Public repository",
        bullets(
            [
                "<strong>github.com/BioPAL/BPS</strong>: where the source code lives",
                "<strong>Apache License 2.0</strong>: free to use and improve; authors keep their copyright",
                "<strong>Discussions</strong>: ask questions and share ideas (no coding required)",
                "All processors are visible to anyone",
            ],
            size=34,
        ),
        "github.com/BioPAL/BPS",
        "github-biopal-bps-repo.png",
    ),
)

# 14 - GitHub live tour (pause card)
register(
    {"id": "14", "file": "slides/14-github-tour.html", "title": "GitHub live tour", "act": "I"},
    f"""\
<section class="slide slide--act-i" data-label="14" data-screen-label="GitHub live tour" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
{ribbon("ACT I - DEMO")}
  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1;padding-bottom:40px;">
    <div style="display:flex;flex-direction:column;align-items:center;gap:32px;text-align:center;max-width:1200px;">
      <div style="font-family:var(--fm);font-weight:700;font-size:26px;letter-spacing:.12em;text-transform:uppercase;color:var(--cy);">LIVE WALKTHROUGH</div>
      <div style="font-weight:700;font-size:72px;color:#fff;line-height:1.05;">GitHub repository</div>
      <div style="width:120px;height:5px;background:var(--cy);border-radius:3px;"></div>
      <div style="font-size:34px;color:rgba(255,255,255,.5);line-height:1.4;">Issues - Discussions - Pull requests - Templates - CODEOWNERS</div>
      <div style="display:flex;gap:28px;align-items:center;margin-top:16px;">
        <a href="https://github.com/BioPAL/BPS" target="_blank" rel="noopener noreferrer"
           onclick="event.stopPropagation()"
           style="display:inline-flex;align-items:center;gap:14px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);border-radius:10px;padding:20px 40px;text-decoration:none;box-shadow:0 4px 24px rgba(0,0,0,.3);">
          <i class="fa-brands fa-github" style="font-size:30px;color:#fff;"></i>
          <span style="font-family:var(--fm);font-size:30px;font-weight:700;color:#fff;">Open github.com/BioPAL/BPS</span>
        </a>
      </div>
    </div>
  </div>
  {footer("14")}
</section>
""",
)

# 15 - Governance
register(
    {"id": "15", "file": "slides/15-governance.html", "title": "Governance", "act": "I"},
    split_shot(
        "15",
        "Governance",
        "slide--act-i",
        "ACT I - GOVERNANCE",
        "GOVERNANCE",
        "Governance",
        '<div style="display:flex;flex-direction:column;gap:14px;margin-bottom:20px;">'
        '<div style="display:flex;gap:14px;align-items:center;font-size:32px;color:rgba(255,255,255,.85);"><i class="fa-solid fa-satellite-dish" style="color:var(--cy);width:30px;text-align:center;"></i><span><strong style="color:#fff;">ESA</strong>: final authority, release gate</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;font-size:32px;color:rgba(255,255,255,.85);"><i class="fa-solid fa-shield-halved" style="color:var(--tl);width:30px;text-align:center;"></i><span><strong style="color:#fff;">Maintainers</strong>: guard the code, review PRs, manage CI</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;font-size:32px;color:rgba(255,255,255,.85);"><i class="fa-solid fa-users" style="color:var(--gr);width:30px;text-align:center;"></i><span><strong style="color:#fff;">Stewards</strong>: guide each area, triage issues</span></div>'
        '<div style="display:flex;gap:14px;align-items:center;font-size:32px;color:rgba(255,255,255,.85);"><i class="fa-solid fa-microscope" style="color:var(--pk);width:30px;text-align:center;"></i><span><strong style="color:#fff;">SMEs</strong>: validate science per bps-* module</span></div>'
        "</div>"
        '<div style="background:rgba(151,192,11,.08);border:1px solid rgba(151,192,11,.25);border-left:4px solid var(--gr);border-radius:8px;padding:18px 20px;font-size:28px;color:rgba(255,255,255,.85);line-height:1.45;">'
        "Any release to production requires ESA approval on the version and the changelog."
        "</div>",
        "biomass-disc.info/docs/governance/",
        "docs-governance.png",
    ),
)

# 16 - Website live tour (end of Act I)
register(
    {"id": "16", "file": "slides/16-website-tour.html", "title": "Website live tour", "act": "I"},
    f"""\
<section class="slide slide--act-i" data-label="16" data-screen-label="Website live tour" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
{ribbon("ACT I - DEMO")}
  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1;padding-bottom:40px;">
    <div style="display:flex;flex-direction:column;align-items:center;gap:32px;text-align:center;max-width:1200px;">
      <div style="font-family:var(--fm);font-weight:700;font-size:26px;letter-spacing:.12em;text-transform:uppercase;color:var(--gr);">LIVE WALKTHROUGH</div>
      <div style="font-weight:700;font-size:72px;color:#fff;line-height:1.05;">Documentation site</div>
      <div style="width:120px;height:5px;background:var(--gr);border-radius:3px;"></div>
      <div style="font-size:34px;color:rgba(255,255,255,.5);line-height:1.4;">Get Started - User Guide - Tutorials - Science Guide - Communication</div>
      <div style="display:flex;gap:28px;align-items:center;margin-top:16px;">
        <a href="https://biomass-disc.info/docs/" target="_blank" rel="noopener noreferrer"
           onclick="event.stopPropagation()"
           style="display:inline-flex;align-items:center;gap:14px;background:var(--tl);border-radius:10px;padding:20px 40px;text-decoration:none;box-shadow:0 4px 24px rgba(0,122,148,.35);">
          <i class="fa-solid fa-arrow-up-right-from-square" style="font-size:24px;color:#fff;"></i>
          <span style="font-family:var(--fm);font-size:30px;font-weight:700;color:#fff;">Open biomass-disc.info/docs</span>
        </a>
      </div>
    </div>
  </div>
  {footer("16")}
</section>
""",
)

# ===================================================================
# Act II - Work remaining (17-25)
# ===================================================================

register(
    {"id": "17", "file": "slides/17-act-ii.html", "title": "ACT II", "act": "chapter", "kind": "chapter"},
    chapter_slide(
        "17",
        "II",
        "Work remaining",
        "Near-term target - three work tracks",
        "slide--act-ii",
    ),
)

# 18 - Near-term target (vs follow-on)
ready_now = [
    ("Foundation PRs", "#24 reviewer routing, #30 CI pipeline, #35 docs sources, #36 SPDX headers"),
    ("CI/CD pipeline", "Three-tier checks green and enforced on the shared branch"),
    ("Licence & security", "Apache-2.0 visible on GitHub - SECURITY.md contact page"),
    ("Platform roles", "GitHub and GitLab roles documented"),
    ("Data & software access", "Auxiliary data policy published - one official package entry point"),
    ("Framing processor", "Clear public position: installable package or updated tutorial"),
    ("Software Reuse File", "CREDITS file public - redistributable vs restricted parts clarified"),
]
ready_later = [
    "Community hygiene: open issues labelled and owned - FAQ pinned in Discussions",
    "Complete web User Manual",
    'All applicable documents on the web (ATBDs, PFDs, SUM, ICD, …)'
    '<div style="margin-top:6px;font-size:20px;color:#FFD066;line-height:1.35;">If required and validated</div>',
    "AGB page PDF export fix (#39)",
    "Real unit tests for at least one processor family",
    "Relicensing of restricted modules to Apache-2.0",
    "Align package metadata with licences",
    "First official GitHub Release (tagged suite version + notes)",
    "Unified build system (single entry point)",
    "Full reproducible build from public source only",
]
ready_now_html = "".join(
    f"""\
<div style="display:flex;gap:16px;align-items:flex-start;padding:12px 0;border-bottom:1px solid rgba(255,255,255,.08);">
  <div style="width:10px;height:10px;border-radius:50%;background:var(--gr);flex-shrink:0;margin-top:12px;"></div>
  <div>
    <div style="font-weight:700;font-size:26px;color:#fff;line-height:1.25;margin-bottom:4px;">{title}</div>
    <div style="font-size:24px;color:rgba(255,255,255,.6);line-height:1.35;">{body}</div>
  </div>
</div>"""
    for title, body in ready_now
)
ready_later_html = "".join(
    f"""\
<div style="display:flex;gap:14px;align-items:flex-start;padding:10px 0;border-bottom:1px solid rgba(255,255,255,.08);">
  <div style="width:8px;height:8px;border-radius:50%;background:rgba(0,182,240,.7);flex-shrink:0;margin-top:10px;"></div>
  <div style="font-size:24px;color:rgba(255,255,255,.65);line-height:1.35;">{item}</div>
</div>"""
    for item in ready_later
)
register(
    {"id": "18", "file": "slides/18-near-term-target.html", "title": "Near-term target", "act": "II"},
    f"""\
<section class="slide slide--act-ii" data-label="18" data-screen-label="Near-term target" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
{ribbon("ACT II - TARGET")}
  <div class="slide-body slide-body--stack" style="z-index:1;">
    <div style="display:flex;flex-direction:column;gap:16px;width:100%;">
      <div style="font-weight:700;font-size:56px;color:#fff;line-height:1.1;letter-spacing:-.01em;">Near-term target</div>
      <div style="width:80px;height:4px;background:var(--gr);border-radius:2px;margin:4px 0 4px;"></div>
      <div style="display:flex;gap:28px;">
        <div style="flex:1.35;background:rgba(151,192,11,.06);border:1px solid rgba(151,192,11,.25);border-top:4px solid var(--gr);border-radius:10px;padding:22px 26px;">
          <div style="font-family:var(--fm);font-weight:700;font-size:22px;letter-spacing:.08em;text-transform:uppercase;color:var(--gr);margin-bottom:8px;">In scope</div>
          {ready_now_html}
        </div>
        <div style="flex:1;background:rgba(0,182,240,.05);border:1px solid rgba(0,182,240,.22);border-top:4px solid var(--cy);border-radius:10px;padding:22px 26px;">
          <div style="font-family:var(--fm);font-weight:700;font-size:22px;letter-spacing:.08em;text-transform:uppercase;color:var(--cy);margin-bottom:8px;">Follow-on</div>
          {ready_later_html}
        </div>
      </div>
    </div>
  </div>
  {footer("18")}
</section>
""",
)

# 19 - Track A: Quality and review
track_a_rows = [
    [
        "Accept reviewer routing and licence files",
        '<span style="color:var(--cy);font-family:var(--fm);font-weight:700;">#24</span>',
        "Routes reviews to the right experts and places licence texts correctly.",
        '<span style="color:var(--gr);">Ready</span>',
    ],
    [
        "Bring automated checks to green, then accept",
        '<span style="color:var(--cy);font-family:var(--fm);font-weight:700;">#30</span>',
        "Closes the quality gate so every proposed change is checked.",
        '<span style="color:#FFD066;">In progress</span>',
    ],
    [
        "Licence notice in source files (SPDX)",
        '<span style="color:var(--cy);font-family:var(--fm);font-weight:700;">#36</span>',
        "Adds SPDX headers once #24 is in place.",
        '<span style="color:#FFD066;">In progress</span>',
    ],
    [
        "Protect the shared working branch",
        "repo",
        "Require automated checks to pass before acceptance.",
        '<span style="color:#FFD066;">In progress</span>',
    ],
    [
        "Document GitHub and GitLab roles",
        "n/a",
        "Write down what runs where so contributors know the path.",
        '<span style="color:rgba(255,255,255,.5);">To do</span>',
    ],
    [
        "Add a security contact page",
        "SECURITY.md",
        "Clear channel for reporting security issues.",
        '<span style="color:rgba(255,255,255,.5);">To do</span>',
    ],
    [
        "Add real unit tests for one processor family",
        "CI",
        "Replace placeholder tests with meaningful checks.",
        '<span style="color:rgba(255,255,255,.5);">To do</span>',
    ],
]
register(
    {"id": "19", "file": "slides/19-workstream-repo-ci.html", "title": "Track A - Quality and review", "act": "II"},
    content_slide(
        "19",
        "Track A - Quality and review",
        "slide--act-ii",
        "ACT II - TRACK A",
        "Track A - Quality and review",
        '<div style="font-size:26px;color:rgba(255,255,255,.55);margin-bottom:16px;line-height:1.4;">'
        "Foundation PRs and automated checks. Suggested order: #24, #30, #36. Website sources (#35) are under Track B."
        "</div>"
        + task_table(
            ["Task", "Ref", "Description", "Status"],
            track_a_rows,
            col_widths=["26%", "10%", "44%", "20%"],
            font_size=21,
        ),
    ),
)

# 20 - Track B: Documentation
track_b_rows = [
    ["Put website sources with the code", "#35", "Resolve conflicts and accept so docs and code share one review path.", '<span style="color:#FFD066;">In progress</span>'],
    ["Publish the site from the shared branch", "deploy", "Ensure biomass-disc.info/docs follows the shared working branch.", '<span style="color:var(--gr);">Ready</span>'],
    ["Fix AGB algorithm page PDF export", "#39", "The web AGB page should also produce a clean PDF.", '<span style="color:rgba(255,255,255,.5);">To do</span>'],
    ["User Manual on the web (start)", "TBD", "Navigation and the most used chapters first; full manual later.", '<span style="color:rgba(255,255,255,.5);">To do</span>'],
]
register(
    {"id": "20", "file": "slides/20-workstream-docs.html", "title": "Track B - Documentation", "act": "II"},
    content_slide(
        "20",
        "Track B - Documentation",
        "slide--act-ii",
        "ACT II - TRACK B",
        "Track B - Documentation",
        '<div style="font-size:26px;color:rgba(255,255,255,.55);margin-bottom:16px;line-height:1.4;">'
        "Documentation and code live together so updates are reviewable like any other change."
        "</div>"
        + task_table(["Task", "Ref", "Description", "Status"], track_b_rows, col_widths=["28%", "10%", "44%", "18%"], font_size=22),
    ),
)

# 21 - Document conversion plan (applicable documents)
doc_done = '<span style="color:var(--gr);">Done</span>'
doc_todo = '<span style="color:rgba(255,255,255,.5);">Not started</span>'
doc_rows = [
    ["Above-Ground Biomass Product ATBD", "ATBD", "BIO-BPS-AGB-ATBD-ARE-024912", "3.1.4", doc_done],
    ["Forest Height Product ATBD", "ATBD", "BIO-BPS-FH-ATBD-ARE-10343", "2.2.0", doc_todo],
    ["Forest Disturbance Product ATBD", "ATBD", "BIO-BPS-FD-ATBD-ARE-10344", "2.1.8", doc_todo],
    ["L1 SAR Product ATBD", "ATBD", "BIO-BPS-L1-SAR-ATBD-ARE-010165", "1.2.4", doc_todo],
    ["L1c Stack Product ATBD", "ATBD", "BIO-BPS-L1-STACK-ATBD-ARE-010166", "1.4.0", doc_todo],
    ["AGB Products Format Specification", "PFD", "BIO-BPS-AGB-PFD-ARE-010257", "3.4.0", doc_todo],
    ["Forest Height Products Format Spec.", "PFD", "BIO-BPS-FH-PFD-ARE-010256", "3.4.0", doc_todo],
    ["Forest Disturbance Products Format Spec.", "PFD", "BIO-BPS-FD-PFD-ARE-010258", "3.4.0", doc_todo],
    ["L1a/b/c Products Format Specification", "PFD", "BIO-BPS-L1-PFD-ARE-010076", "1.6.1", doc_todo],
    ["Software User Manual", "SUM", "BIO-BPS-SUM-ARE-010479", "4.4.1", doc_todo],
    ["Release Note", "RN", "BIO-BPS-RN-ARE-010556", "4.4.5", doc_todo],
    ["Processing Interface Control Document", "ICD", "BIO-BPS-ICD-ARE-010113", "3.2.3", doc_todo],
    ["Processing Input & Output Data Definition", "IODD", "BIO-BPS-IODD-ARE-010112", "3.1.2", doc_todo],
    ["BPS Auxiliary Products Format", "AUX", "BIO-BPS-AUX-FMT-ARE-010163", "3.6.1", doc_todo],
]
register(
    {"id": "21", "file": "slides/21-atbd-plan.html", "title": "Document conversion plan", "act": "II"},
    content_slide(
        "21",
        "Document conversion plan",
        "slide--act-ii",
        "ACT II - TRACK B",
        "Document conversion plan",
        '<div style="font-size:24px;color:rgba(255,255,255,.55);margin-bottom:12px;line-height:1.35;">'
        "Applicable documents: convert from Word/PDF to reStructuredText and publish on the documentation website. "
        "Version-controlled, editable, citable. Official PDFs remain the reference until each web page is approved."
        "</div>"
        + task_table(
            ["Document", "Kind", "Reference", "Version", "Status"],
            doc_rows,
            col_widths=["32%", "8%", "34%", "10%", "16%"],
            font_size=17,
            cell_padding="6px 10px",
            head_size=16,
        )
        + '<div style="margin-top:14px;font-size:22px;color:rgba(255,255,255,.5);line-height:1.35;">'
        "AGB ATBD: web version done (PDF export still to fix). Typical effort ~2-3 days per document. "
        "Scientific Module Experts validate content after conversion. Word/PDF → rST (not the other way around)."
        "</div>",
    ),
)

# 22 - Track C: Licensing and community
track_c_rows = [
    ["Publish the Software Reuse File", "CREDITS", "Replace the empty CREDITS link with a public reuse file.", '<span style="color:var(--gr);">Ready</span>'],
    ["Align package metadata with licences", "recipes", "Ensure each module states the correct licence in package metadata.", '<span style="color:#FFD066;">In progress</span>'],
    ["List redistributable vs restricted parts", "licensing", "Clarify what can ship under Apache-2.0 and what remains restricted.", '<span style="color:rgba(255,255,255,.5);">To do</span>'],
    ["Triage open questions", "issues", "Add labels and clear ownership so community questions do not stall.", '<span style="color:rgba(255,255,255,.5);">To do</span>'],
    ["Pinned FAQ in Discussions", "Discussions", "Cover package, data access, MAAP, and versions in a few pinned threads.", '<span style="color:rgba(255,255,255,.5);">To do</span>'],
]
register(
    {"id": "22", "file": "slides/22-workstream-licensing-community.html", "title": "Track C - Licensing and community", "act": "II"},
    content_slide(
        "22",
        "Track C - Licensing and community",
        "slide--act-ii",
        "ACT II - TRACK C",
        "Track C - Licensing and community",
        '<div style="font-size:26px;color:rgba(255,255,255,.55);margin-bottom:16px;line-height:1.4;">'
        "Clarify redistribution boundaries and keep open questions visible."
        "</div>"
        + task_table(["Task", "Ref", "Description", "Status"], track_c_rows, col_widths=["28%", "12%", "42%", "18%"], font_size=22),
    ),
)

# ===================================================================
# Close (23-25)
# ===================================================================
close_points = [
    "Public face is live: docs site, contribution path, and repository",
    "Foundations are designed; four proposed changes still need acceptance into the shared working branch",
    "Near-term target: operable and auditable; deeper openness follows",
    "Remaining work is organised in three tracks: quality, documentation, licensing and community",
]
close_html = "".join(
    f'<div style="display:flex;gap:20px;align-items:flex-start;padding:14px 0;border-bottom:1px solid rgba(255,255,255,.1);font-size:30px;">'
    f'<div style="font-family:var(--fm);font-weight:700;color:var(--gr);min-width:40px;">{i}</div>'
    f'<div style="color:rgba(255,255,255,.88);line-height:1.35;">{a}</div></div>'
    for i, a in enumerate(close_points, 1)
)
register(
    {"id": "23", "file": "slides/23-close.html", "title": "Summary", "act": "close"},
    content_slide("23", "Summary", "slide--act-ii", "CLOSE", "Summary",
        close_html + '<div style="margin-top:32px;font-size:28px;color:rgba(255,255,255,.45);line-height:1.5;">Docs: biomass-disc.info/docs/ - Code: github.com/BioPAL/BPS</div>'),
)

# 24 - Discussion prompts
discussion_items = [
    (
        "Data access policy",
        "What should be freely downloadable versus request-only?",
    ),
    (
        "ATBD web conversion",
        "First: do we accept converting ATBDs from Word/PDF to reStructuredText on the documentation website? "
        "Only then: which document comes next after AGB?",
    ),
    (
        "Licensing and repository configuration (#24)",
        "A dedicated meeting is needed before merge: REUSE vs LICENSE.md / CREDITS, Software Reuse File consistency, "
        "CODEOWNERS routing, and ESA review on licence topics.",
    ),
]
discussion_html = "".join(
    f'<div style="display:flex;gap:20px;align-items:flex-start;padding:20px 0;border-bottom:1px solid rgba(255,255,255,.1);">'
    f'<div style="width:10px;height:10px;border-radius:50%;background:var(--cy);flex-shrink:0;margin-top:14px;"></div>'
    f'<div><div style="font-weight:700;font-size:32px;color:#fff;line-height:1.25;margin-bottom:6px;">{title}</div>'
    f'<div style="font-size:28px;color:rgba(255,255,255,.6);line-height:1.4;">{question}</div></div></div>'
    for title, question in discussion_items
)
register(
    {"id": "24", "file": "slides/24-discussion.html", "title": "Discussion", "act": "close"},
    content_slide("24", "Discussion", "slide--act-ii", "CLOSE", "Discussion",
        '<div style="font-size:30px;color:rgba(255,255,255,.55);margin-bottom:16px;line-height:1.4;">Decisions that would help us move forward:</div>'
        + discussion_html),
)

# 25 - Questions / close
register(
    {"id": "25", "file": "slides/25-questions.html", "title": "Questions", "act": "close"},
    f"""\
<section class="slide slide--act-ii" data-label="25" data-screen-label="Questions" style="font-family:var(--fs);background:var(--bg);">
  {bg()}
{ribbon("CLOSE")}
  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1;padding-bottom:40px;">
    <div style="display:flex;flex-direction:column;align-items:center;gap:28px;text-align:center;max-width:1200px;">
      <div style="font-family:var(--fm);font-weight:700;font-size:76px;letter-spacing:.12em;text-transform:uppercase;color:var(--gr);">END</div>
      <div style="font-weight:700;font-size:96px;color:#fff;line-height:1.05;letter-spacing:-.02em;">Questions?</div>
      <div style="width:120px;height:5px;background:var(--gr);border-radius:3px;"></div>
      <div style="font-size:36px;color:rgba(255,255,255,.65);line-height:1.4;">Thank you</div>
      <div style="display:flex;gap:40px;align-items:center;margin-top:12px;font-size:26px;color:rgba(255,255,255,.45);line-height:1.4;">
        <span>biomass-disc.info/docs</span>
        <span style="opacity:.4;">·</span>
        <span>github.com/BioPAL/BPS</span>
      </div>
    </div>
  </div>
  {footer("25")}
</section>
""",
)


def main() -> None:
    SLIDES.mkdir(parents=True, exist_ok=True)
    for old in SLIDES.glob("*.html"):
        old.unlink()

    for entry in SLIDE_DEFS:
        path = ROOT / "source" / entry["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        html = entry["html"]
        path.write_text(html, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")

    lines = [
        "title: '[BIOMASS DISC] QSM#3 - Open Source readiness'",
        "id: 2026-07-QSM-3",
        "date: '2026-07-17'",
        "width: 1920",
        "height: 1080",
        "slides:",
    ]
    for entry in SLIDE_DEFS:
        lines.append(f"- id: '{entry['id']}'")
        lines.append(f"  file: {entry['file']}")
        lines.append(f"  title: {entry['title']}")
        lines.append(f"  act: {entry['act']}")
        if entry.get("kind"):
            lines.append(f"  kind: {entry['kind']}")
        lines.append("  interactions: []")
    DECK_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {DECK_YAML.relative_to(ROOT)} ({len(SLIDE_DEFS)} slides)")


if __name__ == "__main__":
    main()
