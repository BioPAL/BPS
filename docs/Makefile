# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Makefile for the BIOMASS BPS Sphinx documentation.
#
# First-time setup (from this directory):
#   make install
#   make html
#   make livehtml

# Prefer docs/.venv when present (avoids missing extensions on system Python).
VENV        := .venv
ifeq ($(wildcard $(VENV)/bin/sphinx-build),)
  SPHINXBUILD ?= sphinx-build
  AUTOBUILD   ?= sphinx-autobuild
  PIP         ?= pip
else
  SPHINXBUILD ?= $(VENV)/bin/sphinx-build
  AUTOBUILD   ?= $(VENV)/bin/sphinx-autobuild
  PIP         ?= $(VENV)/bin/pip
endif

SPHINXOPTS    ?=
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help install html html-all clean livehtml linkcheck diagrams sync-bib atbd-pdf presentation presentation-shared

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo
	@echo "Extra targets:"
	@echo "  make presentation        Build dated deck slides into _extra_static/"
	@echo "  make presentation-shared Sync shared logos only (after editing _shared/logos/)"
	@echo "  make atbd-pdf     Build web-export PDF for the L2 AGB ATBD (requires LaTeX)"
	@echo "  make html-all     atbd-pdf + html"
	@echo "  make sync-bib     Copy references.bib from the private conversion pipeline"

install:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"
	$(PIP) install -r requirements.txt sphinx-autobuild
	@echo "Docs venv ready. Run: make html  (or: make livehtml)"

presentation:
	@$(MAKE) -C presentations/2026-06-30-first-dev-meeting publish

presentation-shared:
	@python3 presentations/2026-06-30-first-dev-meeting/scripts/publish_docs.py --shared-only

# Standard HTML build (includes presentation static files when built)
html: presentation-shared presentation
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html in your browser."

# HTML plus ATBD web-export PDF (requires LaTeX for atbd-pdf)
html-all: atbd-pdf html

# Clean the build directory
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Build directory cleaned."

# Live-rebuild server (install via make install)
livehtml:
	$(AUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Check external links
linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Re-render every D2 source into SVG. Requires the d2 CLI (brew install d2).
diagrams:
	@for f in _static/diagrams/*.d2; do \
	  out="$${f%.d2}.svg"; \
	  echo "  d2  $$f  ->  $$out"; \
	  d2 --layout=elk --theme=0 --pad=20 "$$f" "$$out"; \
	done

# Copy shared bibliography from the local private conversion pipeline.
PRIVATE_BIB ?= ../private/atbd-conversion/pipeline/references.bib

sync-bib:
	@test -f "$(PRIVATE_BIB)" || (echo "Missing $(PRIVATE_BIB) — private pipeline is local-only."; exit 1)
	cp "$(PRIVATE_BIB)" references.bib
	@echo "Synced references.bib from private/atbd-conversion/pipeline/"

# Build a standalone PDF for one multi-page ATBD (requires LaTeX: texlive or MacTeX).
# Includes every chapter listed in the ATBD index toctree (through References).
# Usage:  make atbd-pdf
#         make atbd-pdf ATBD=atbd-l2-agb
# Optional: install mermaid-cli (`npm i -g @mermaid-js/mermaid-cli`) so flowcharts
# render in the PDF; otherwise PNG figures from the PDF conversion are included.
ATBD ?= atbd-l2-agb
ATBD_INDEX = $(ATBD)/index
ATBD_PDF = _static/pdf/$(ATBD).pdf
ATBD_LATEX_DIR = $(BUILDDIR)/atbd-pdf/latex

atbd-pdf:
	@echo "Building web-export PDF for science-guide/$(ATBD)/ …"
	@SPHINX_ATBD_PDF=$(ATBD) $(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)/atbd-pdf" $(SPHINXOPTS)
	@cd "$(ATBD_LATEX_DIR)" && \
	  if command -v latexmk >/dev/null 2>&1; then \
	    latexmk -xelatex -interaction=nonstopmode "$(ATBD).tex"; \
	  elif command -v xelatex >/dev/null 2>&1; then \
	    xelatex -interaction=nonstopmode "$(ATBD).tex"; \
	    xelatex -interaction=nonstopmode "$(ATBD).tex"; \
	  elif command -v pdflatex >/dev/null 2>&1; then \
	    pdflatex -interaction=nonstopmode "$(ATBD).tex"; \
	    pdflatex -interaction=nonstopmode "$(ATBD).tex"; \
	  else \
	    echo "No LaTeX engine found. Install MacTeX or TeX Live (pdflatex or latexmk)."; \
	    echo "LaTeX sources are in $(ATBD_LATEX_DIR)/"; exit 1; \
	  fi
	@if [ -f "$(ATBD_LATEX_DIR)/$(ATBD).pdf" ]; then \
	  mkdir -p _static/pdf; \
	  cp "$(ATBD_LATEX_DIR)/$(ATBD).pdf" "$(ATBD_PDF)"; \
	  echo "PDF written to $(ATBD_PDF) (linked from the ATBD sidebar and References page)"; \
	else \
	  echo "PDF build failed (no output file). Check $(ATBD_LATEX_DIR)/$(ATBD).log"; exit 1; \
	fi

# Catch-all: route any unknown target to sphinx-build via the M shortcut
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
