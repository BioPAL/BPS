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

.PHONY: help install html html-all clean livehtml linkcheck diagrams sync-bib atbd-pdf atbd-pdf-all presentation presentation-shared

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo
	@echo "Extra targets:"
	@echo "  make presentation        Build dated deck slides into _extra_static/"
	@echo "  make presentation-shared Sync shared logos only (after editing _shared/logos/)"
	@echo "  make atbd-pdf     Build web-export PDF of one document (ATBD=<folder>, default atbd-l2-agb; requires LaTeX)"
	@echo "  make atbd-pdf-all Build web-export PDFs of every Science Guide document"
	@echo "  make html-all     atbd-pdf-all + html"
	@echo "  make sync-bib     Copy references.bib from the private conversion pipeline"

install:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"
	$(PIP) install -r requirements.txt sphinx-autobuild
	@echo "Docs venv ready. Run: make html  (or: make livehtml)"

presentation:
	@$(MAKE) -C presentations/2026-06-30-first-dev-meeting publish
	@$(MAKE) -C presentations/2026-11-02-second-dev-meeting publish

presentation-shared:
	@python3 presentations/2026-06-30-first-dev-meeting/scripts/publish_docs.py --shared-only

# Standard HTML build (includes presentation static files when built)
html: presentation-shared presentation
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html in your browser."

# HTML plus every Science Guide web-export PDF (PDF failures do not block HTML)
html-all:
	-@$(MAKE) --no-print-directory atbd-pdf-all
	@$(MAKE) --no-print-directory html

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

# Build a standalone PDF for one Science Guide document (requires LaTeX: TeX Live
# or MacTeX). Includes every chapter listed in the document index toctree.
# Usage:  make atbd-pdf                      (L2 AGB ATBD)
#         make atbd-pdf ATBD=pfd-l2-fh
#         make atbd-pdf-all                  (every science-guide/<folder>/index.md)
#
# LaTeX from a container instead of a local TeX Live:
#         make atbd-pdf-all LATEX_IMAGE=bps-latex:ci
# or any latexmk-compatible command (run from the LaTeX output directory):
#         make atbd-pdf LATEXMK='docker run --rm -v $$PWD:/w -w /w img latexmk'
#
# Mermaid diagrams are rendered to PNG when mermaid-cli (`mmdc`) is found on
# PATH or given with MMDC=/path/to/mmdc (`npm i -g @mermaid-js/mermaid-cli`).
ATBD ?= atbd-l2-agb
ATBD_PDF = _static/pdf/$(ATBD).pdf
ATBD_PDF_BUILDDIR = $(BUILDDIR)/atbd-pdf
ATBD_LATEX_DIR = $(ATBD_PDF_BUILDDIR)/$(ATBD)
ATBD_DOCS := $(sort $(patsubst science-guide/%/index.md,%,$(wildcard science-guide/*/index.md)))

LATEX_IMAGE ?=
ifneq ($(LATEX_IMAGE),)
  LATEXMK ?= docker run --rm -u $$(id -u):$$(id -g) -v "$$PWD":/w -w /w $(LATEX_IMAGE) latexmk
endif
LATEXMK ?=
# -f: go on past recoverable LaTeX errors (counted in the atbd-pdf-all summary)
# so that one bad equation does not drop the whole PDF. A fatal error (the
# run stopped before the end of the document) still fails the build.
LATEXMK_OPTS ?= -f
ATBD_LATEX_FATAL = ^! Emergency stop|^!  ==> Fatal error|^! TeX capacity exceeded|^No pages of output

MMDC ?= $(shell command -v mmdc 2>/dev/null)
ifneq ($(MMDC),)
  # --pdfFit: a PDF page the size of the diagram (default: a Letter page, which
  # crops wide diagrams and leaves the rest of the page blank under the others)
  ATBD_MERMAID_OPTS = -D mermaid_cmd=$(MMDC) -D mermaid_params=-p,$(abspath $(ATBD_PDF_BUILDDIR))/puppeteer.json,--pdfFit
endif

# Warnings of sphinxcontrib.mermaid for a diagram it did not render (mmdc not
# found, or failed): the PDF would show the captions without the figures.
ATBD_MERMAID_ERRORS = needed for mermaid output|WARNING: mm code

atbd-pdf:
	@echo "Building web-export PDF for science-guide/$(ATBD)/ …"
	@test -f "science-guide/$(ATBD)/index.md" || (echo "No science-guide/$(ATBD)/index.md"; exit 1)
	@mkdir -p "$(ATBD_LATEX_DIR)"
	@rm -f "$(ATBD_LATEX_DIR)/$(ATBD).pdf" "$(ATBD_PDF)"
	@printf '{"args": ["--no-sandbox"]}\n' > "$(ATBD_PDF_BUILDDIR)/puppeteer.json"
	@# sphinxcontrib.mermaid reuses a rendered diagram whatever the mmdc
	@# options: render them again when the options change.
	@if [ "$$(cat "$(ATBD_LATEX_DIR)/.mermaid-opts" 2>/dev/null)" != '$(ATBD_MERMAID_OPTS)' ]; then \
	  rm -f "$(ATBD_LATEX_DIR)"/mermaid-*.pdf; \
	  echo '$(ATBD_MERMAID_OPTS)' > "$(ATBD_LATEX_DIR)/.mermaid-opts"; \
	fi
	@SPHINX_ATBD_PDF=$(ATBD) $(SPHINXBUILD) -b latex -d "$(ATBD_PDF_BUILDDIR)/doctrees" \
	  -w "$(ATBD_PDF_BUILDDIR)/$(ATBD).warnings" \
	  "$(SOURCEDIR)" "$(ATBD_LATEX_DIR)" $(ATBD_MERMAID_OPTS) $(SPHINXOPTS)
	@n=$$(grep -Ec '$(ATBD_MERMAID_ERRORS)' "$(ATBD_PDF_BUILDDIR)/$(ATBD).warnings"); \
	if [ "$$n" -gt 0 ]; then \
	  echo "PDF build failed: $$n Mermaid diagram(s) of $(ATBD) not rendered (mermaid-cli"; \
	  echo "mmdc not found or failed, MMDC='$(MMDC)'). Install it (npm i -g @mermaid-js/mermaid-cli)"; \
	  echo "or pass MMDC=/path/to/mmdc. Warnings: $(ATBD_PDF_BUILDDIR)/$(ATBD).warnings"; \
	  exit 1; \
	fi
	@cd "$(ATBD_LATEX_DIR)" && \
	  if [ -n '$(LATEXMK)' ]; then \
	    $(LATEXMK) -xelatex $(LATEXMK_OPTS) -interaction=nonstopmode "$(ATBD).tex"; \
	  elif command -v latexmk >/dev/null 2>&1; then \
	    latexmk -xelatex $(LATEXMK_OPTS) -interaction=nonstopmode "$(ATBD).tex"; \
	  elif command -v xelatex >/dev/null 2>&1; then \
	    xelatex -interaction=nonstopmode "$(ATBD).tex"; \
	    xelatex -interaction=nonstopmode "$(ATBD).tex"; \
	  else \
	    echo "No LaTeX engine found. Install TeX Live (latexmk + xelatex) or set LATEX_IMAGE."; \
	    echo "LaTeX sources are in $(ATBD_LATEX_DIR)/"; exit 1; \
	  fi || true
	@log="$(ATBD_LATEX_DIR)/$(ATBD).log"; pdf="$(ATBD_LATEX_DIR)/$(ATBD).pdf"; \
	if [ -f "$$log" ] && grep -Eq '$(ATBD_LATEX_FATAL)' "$$log"; then \
	  grep -E -m3 '$(ATBD_LATEX_FATAL)' "$$log"; \
	  echo "PDF build failed (fatal LaTeX error, the PDF would be truncated). Check $$log"; \
	  rm -f "$$pdf"; exit 1; \
	fi; \
	if [ -f "$$pdf" ] && ! tail -c 1024 "$$pdf" | grep -q '%%EOF'; then \
	  echo "PDF build failed (incomplete $$pdf)"; rm -f "$$pdf"; exit 1; \
	fi
	@if [ -f "$(ATBD_LATEX_DIR)/$(ATBD).pdf" ]; then \
	  mkdir -p _static/pdf; \
	  cp "$(ATBD_LATEX_DIR)/$(ATBD).pdf" "$(ATBD_PDF)"; \
	  echo "PDF written to $(ATBD_PDF) (linked from the document sidebar and References page)"; \
	else \
	  echo "PDF build failed (no output file). Check $(ATBD_LATEX_DIR)/$(ATBD).log"; exit 1; \
	fi

# Build the PDF of every Science Guide document; one failure does not stop the
# others. Exits non-zero when at least one PDF is missing.
atbd-pdf-all:
	@ok=""; failed=""; \
	for d in $(ATBD_DOCS); do \
	  if $(MAKE) --no-print-directory atbd-pdf ATBD=$$d; then ok="$$ok $$d"; \
	  else failed="$$failed $$d"; fi; \
	done; \
	echo; echo "Science Guide PDF summary:"; \
	for d in $$ok; do \
	  log="$(ATBD_PDF_BUILDDIR)/$$d/$$d.log"; \
	  printf '  OK      %-16s %4s pages, %s LaTeX error(s)\n' "$$d" \
	    "$$(sed -n 's/.*Output written on .*(\([0-9]*\) pages.*/\1/p' "$$log" 2>/dev/null | tail -1)" \
	    "$$(grep -c '^!' "$$log" 2>/dev/null)"; \
	done; \
	for d in $$failed; do \
	  if grep -Eq '$(ATBD_MERMAID_ERRORS)' "$(ATBD_PDF_BUILDDIR)/$$d.warnings" 2>/dev/null; then \
	    printf '  FAILED  %-16s Mermaid diagrams not rendered, see %s\n' "$$d" "$(ATBD_PDF_BUILDDIR)/$$d.warnings"; \
	  else printf '  FAILED  %-16s see %s\n' "$$d" "$(ATBD_PDF_BUILDDIR)/$$d/$$d.log"; fi; \
	done; \
	test -z "$$failed"

# Catch-all: route any unknown target to sphinx-build via the M shortcut
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
