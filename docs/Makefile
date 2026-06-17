# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Makefile for the BIOMASS BPS Sphinx documentation.

# You can set these variables from the command line, and also from the
# environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help install html clean livehtml linkcheck diagrams

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

install:
	pip install -r requirements.txt

# Standard HTML build
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html in your browser."

# Clean the build directory
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Build directory cleaned."

# Live-rebuild server (requires sphinx-autobuild: pip install sphinx-autobuild)
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

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

# Catch-all: route any unknown target to sphinx-build via the M shortcut
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
