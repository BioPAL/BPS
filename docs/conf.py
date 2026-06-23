# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
#
# Sphinx configuration for the BioPAL documentation site (BPS content).
# Theme: PyData Sphinx Theme (same family as docs.xarray.dev).
# Source: Markdown via MyST parser.

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("_ext"))

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "BioPAL"
author = "ACRI-ST / ESA / Aresys"
copyright = f"{datetime.now().year}, {author}"
release = "0.2.0"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "myst_parser",  # Markdown support (CommonMark + extensions)
    "sphinx_copybutton",  # Copy button on code blocks
    "sphinx_design",  # Cards, grids, tabs, etc.
    "sphinxcontrib.mermaid",  # Mermaid diagrams (used in CONTRIBUTING_PART1)
    "sphinxcontrib.bibtex",  # BibTeX bibliography (ATBD/PFD conversions)
    "atbd_sidebar_toc",  # PDF-style collapsible TOC in ATBD left sidebar
    "atbd_doc_refs",  # [ADn] / [RDn] / Fig.N / sec. / eq. links in the ATBD
    "science_guide_nav",  # Science Guide hub: ATBD list sidebar (not chapter tree)
    "bps_version",  # Latest bps-v* tag for homepage version badge
]

# Tell Sphinx that .md files exist alongside .rst
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# Master document (the landing page)
master_doc = "index"

# Patterns to ignore
exclude_patterns = [
    "_build",
    ".venv",
    "**/.venv/**",
    "_ext",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "**/README.md",
    "science-guide/_includes/*",
    "science-guide/draft-pfd-l2-fh.md",
    # Tutorial bundle assets: shipped with the docs but not Sphinx-rendered.
    "tutorials/run-bps-locally/scripts/*",
    "tutorials/run-bps-locally/notebooks/*",
    "tutorials/run-bps-locally/CONFIGURATION_FILE/**",
]

# Suppress noisy warnings.
# - "image.not_readable": raw <img> tags reference SVGs in `_extra_static/`,
#   which Sphinx doesn't track as source images. They render correctly in the
#   browser, so the warning is misleading.
suppress_warnings = [
    "image.not_readable",
    "toc.not_included",
    "myst.header",
]

# Language
language = "en"

# -----------------------------------------------------------------------------
# MyST parser configuration
# -----------------------------------------------------------------------------
# Enable common MyST extensions for richer Markdown.
myst_enable_extensions = [
    "amsmath",  # AMS math environments in MyST
    "colon_fence",  # ::: fenced directives
    "deflist",  # definition lists
    "dollarmath",  # $...$ and $$...$$ math (ATBD equations)
    "html_admonition",  # raw HTML admonitions
    "html_image",  # raw HTML <img> tags
    "linkify",  # auto-link bare URLs
    "replacements",  # textual replacements (e.g. (c) -> ©)
    "smartquotes",  # smart quotes
    "substitution",  # variable substitution
    "tasklist",  # GitHub-style task lists
]

# Auto-generate anchors for headings up to level 6 (ATBD §3.4.4.2.1, etc.)
myst_heading_anchors = 6

# Numbered figures and tables ({numref}`fig:...` in ATBD pages).
# Equation numbers use PDF-style \tag{} in ATBD sources (not auto (1), (2), …).
numfig = True
math_numfig = False

# -----------------------------------------------------------------------------
# BibTeX (sphinxcontrib-bibtex). Edit docs/references.bib for the site build.
# To refresh from your local private pipeline: make sync-bib (requires private/atbd-conversion/).
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

# -----------------------------------------------------------------------------
# HTML output configuration
# -----------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_title = "BioPAL Documentation"
html_short_title = "BioPAL"

# Base URL when docs are served under a subpath (e.g. biomass-disc.info/docs/).
html_baseurl = os.environ.get("SPHINX_HTML_BASEURL", "/docs/")

# Main website URL for the "return to site" navbar link (site root when embedded).
site_home_url = os.environ.get("SPHINX_SITE_HOME_URL", "/")

# Static assets (custom CSS, logos, favicons)
html_static_path = ["_static"]

# Extra paths copied as-is to the build output (raw files, not processed by Sphinx).
# The CONTENTS of these folders are copied to the build root, so we wrap our
# `images/` folder inside `_extra_static/images/` to get the path right.
# We use this for SVG diagrams referenced via raw HTML <img> / <picture> tags,
# which Sphinx does not track automatically.
html_extra_path = ["_extra_static"]

# Logo and favicon. The PyData theme auto-scales the logo in the navbar header.
html_logo = "_static/logos/BioPAL_textonright.png"
# html_favicon = "_static/favicon.ico"

# Right sidebar: every page gets "On this page" + Edit on GitHub.
# Homepage hides the secondary sidebar entirely via its own frontmatter.
# ATBD chapters add the PDF download button in addition.
_secondary_sidebar_items = {
    "index": [],
    "**": ["page-toc", "edit-this-page"],
    "science-guide/atbd-l2-agb/*": [
        "page-toc",
        "atbd-download-pdf.html",
        "edit-this-page",
    ],
}

# Theme-specific options (PyData Sphinx Theme).
# See https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html
html_theme_options = {
    # Top navbar xarray-style: section names in the header, search on the right
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    # How many top-nav links to show before collapsing into "More" dropdown
    "header_links_before_dropdown": 6,
    # Icon links (top right)
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BioPAL/BPS",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    # Left sidebar (section navigation)
    # show_nav_level = 1 → only show top-level section name expanded,
    # sub-pages appear when the user is inside that section. This matches xarray.
    "show_nav_level": 1,
    "navigation_depth": 4,
    "collapse_navigation": False,
    # Right sidebar (page table of contents)
    "show_toc_level": 2,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    # Misc
    "use_edit_page_button": True,
    "show_prev_next": True,
    "announcement": "BIOMASS BPS is an Open Science project — every contribution is welcome.",
    # Secondary sidebar (right): in-page TOC on content pages; section hubs
    # and the homepage omit "On this page" (see _SECTION_INDEX_PAGES).
    "secondary_sidebar_items": _secondary_sidebar_items,
    # Primary sidebar (left): always start collapsed at depth 1, then expand
    # the current section. This gives the same UX as xarray.
    # Ethical ads removed: local builds showed a violet RTD placeholder and
    # this site is not served from Read the Docs.
    "primary_sidebar_end": [],
}

# Context used by the "Edit on GitHub" button.
# These values feed the URL pattern: github.com/<user>/<repo>/edit/<version>/<doc_path>/<page>.rst
html_context = {
    "github_user": "BioPAL",
    "github_repo": "BPS",
    "github_version": os.environ.get(
        "SPHINX_GITHUB_VERSION", "docs/sphinx-site-migration"
    ),
    "doc_path": "docs",
    "site_home_url": site_home_url,
    # default_mode: "auto" lets the user's choice (stored in localStorage)
    # take precedence on subsequent page loads. "auto" also matches the OS
    # preference on first visit.
    "default_mode": "auto",
}

# Left sidebar: section navigation on all pages except those listed below.
# - Homepage: full-width, no sidebars.
# - Hubs without sibling chapters (user-guide, getting-started): nothing
#   meaningful to show, so the sidebar is hidden rather than rendered empty.
# - developer-guide/index: orphan redirect page.
# - Science Guide and ATBD chapters use custom collapsible templates.
html_sidebars = {
    "index": [],
    "user-guide/index": [],
    "getting-started/index": [],
    "developer-guide/index": [],
    "science-guide/index": [
        "sidebar-collapse",
        "science-guide-atbd-nav.html",
    ],
    "science-guide/atbd-l2-agb/*": [
        "sidebar-collapse",
        "atbd-sidebar-toc.html",
    ],
    "**": ["sidebar-nav-bs"],
}

# Custom CSS to override / extend theme defaults
html_css_files = [
    "custom.css",
]

templates_path = ["_templates"]

# Single-ATBD PDF export (make atbd-pdf ATBD=atbd-l2-agb).
_sphinx_atbd_pdf = os.environ.get("SPHINX_ATBD_PDF")
if _sphinx_atbd_pdf:
    _atbd_doc = _sphinx_atbd_pdf.removesuffix("/index").removesuffix("/")
    if _atbd_doc.startswith("science-guide/"):
        _atbd_doc = _atbd_doc.removeprefix("science-guide/")
    _atbd_latex_stem = _atbd_doc.split("/")[-1]
    _atbd_latex_titles = {
        "atbd-l2-agb": "Above-Ground Biomass Product ATBD (BioPAL web export)",
    }
    _atbd_latex_title = _atbd_latex_titles.get(_atbd_latex_stem, "BioPAL ATBD")
    latex_documents = [
        (
            f"science-guide/{_atbd_doc}/index",
            f"{_atbd_latex_stem}.tex",
            _atbd_latex_title,
            author,
            "manual",
        ),
    ]
    latex_engine = "pdflatex"
    latex_elements = {
        "papersize": "a4paper",
        "pointsize": "11pt",
        # Use default Computer Modern (no extra .sty packages required on CI).
        "fontpkg": "",
        "preamble": r"""
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
""",
    }
    # Mermaid diagrams need mermaid-cli (mmdc) and a headless browser; they are
    # skipped with a warning when unavailable. Re-run `make atbd-pdf` after
    # installing mmdc for complete figure coverage.
    mermaid_output_format = "png"
