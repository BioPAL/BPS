# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
#
# Sphinx configuration for the BIOMASS BPS documentation site.
# Theme: PyData Sphinx Theme (same family as docs.xarray.dev).
# Source: Markdown via MyST parser.

from datetime import datetime
import os

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "BIOMASS BPS"
author = "ESA / Aresys / ACRI-ST"
copyright = f"{datetime.now().year}, {author}"
release = "0.2.0"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "myst_parser",            # Markdown support (CommonMark + extensions)
    "sphinx_copybutton",      # Copy button on code blocks
    "sphinx_design",          # Cards, grids, tabs, etc.
    "sphinxcontrib.mermaid",  # Mermaid diagrams (used in CONTRIBUTING_PART1)
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
    "Thumbs.db",
    ".DS_Store",
    "README.md",
]

# Suppress noisy warnings.
# - "image.not_readable": raw <img> tags reference SVGs in `_extra_static/`,
#   which Sphinx doesn't track as source images. They render correctly in the
#   browser, so the warning is misleading.
suppress_warnings = [
    "image.not_readable",
]

# Language
language = "en"

# -----------------------------------------------------------------------------
# MyST parser configuration
# -----------------------------------------------------------------------------
# Enable common MyST extensions for richer Markdown.
myst_enable_extensions = [
    "colon_fence",       # ::: fenced directives
    "deflist",           # definition lists
    "html_admonition",   # raw HTML admonitions
    "html_image",        # raw HTML <img> tags
    "linkify",           # auto-link bare URLs
    "replacements",      # textual replacements (e.g. (c) -> ©)
    "smartquotes",       # smart quotes
    "substitution",      # variable substitution
    "tasklist",          # GitHub-style task lists
]

# Auto-generate anchors for headings up to level 3 (cross-doc linking)
myst_heading_anchors = 3

# -----------------------------------------------------------------------------
# HTML output configuration
# -----------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_title = "BIOMASS BPS Documentation"
html_short_title = "BIOMASS BPS"

# GitHub Pages project site is served at https://biopal.github.io/BPS/ .
# CI sets DOCS_BASEURL; leave empty for local builds (file:// or localhost).
html_baseurl = os.environ.get("DOCS_BASEURL", "")

# Static assets (custom CSS, logos, favicons)
html_static_path = ["_static"]

# Extra paths copied as-is to the build output (raw files, not processed by Sphinx).
# The CONTENTS of these folders are copied to the build root, so we wrap our
# `images/` folder inside `_extra_static/images/` to get the path right.
# We use this for SVG diagrams referenced via raw HTML <img> / <picture> tags,
# which Sphinx does not track automatically.
html_extra_path = ["_extra_static"]

# Logo and favicon. The PyData theme auto-scales the logo in the navbar header.
html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

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
    "announcement": "BIOMASS BPS is an Open Science project - every contribution is welcome.",

    # Secondary sidebar (right): only the "Edit on GitHub" button.
    # We hide page-toc (already covered by left sidebar) and sourcelink.
    "secondary_sidebar_items": {
        "**": ["edit-this-page"],
        "index": [],  # no right sidebar on the landing page
    },

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
    "github_version": os.environ.get("DOCS_GITHUB_VERSION", "develop"),
    "doc_path": "docs",
    # default_mode: "auto" lets the user's choice (stored in localStorage)
    # take precedence on subsequent page loads. "auto" also matches the OS
    # preference on first visit.
    "default_mode": "auto",
}

# Left sidebar configuration.
# On the landing page (`index`) we hide the sidebar entirely so the homepage
# stays clean (xarray pattern). On every other page we show the default
# section navigation (sidebar-nav-bs), which lists the pages of the current
# top-level section.
html_sidebars = {
    "index": [],
    "getting-started/index": [],
    "user-guide/index": [],
    "tutorials/index": [],
    "science-guide/index": [],
    "**": ["sidebar-nav-bs"],
}

# Custom CSS to override / extend theme defaults
html_css_files = [
    "custom.css",
]
