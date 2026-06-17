<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Developer Guide

Technical reference for developers working on (or extending) BIOMASS BPS.

This section covers the **internal organisation** of the codebase, the **coding conventions**
we follow, and the **documentation standards** that all contributions must respect.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🏗️ Architecture
:link: architecture
:link-type: doc

Monorepo layout, `bps-*` modules, dependency graph, and processor architecture.
:::

:::{grid-item-card} 📝 Code Standards
:link: code-standards
:link-type: doc

Coding conventions, linters, type hints, formatting, and quality bars.
:::

:::{grid-item-card} 📚 Documentation Standards
:link: documentation-standards
:link-type: doc

Documentation conventions, docstring style, and writing guidelines.
:::

::::

## Interface and auxiliary specifications

The processing interfaces and the auxiliary data formats are specified in three
official documents published on the ESA dissemination portal. They are the
authoritative references for anyone integrating BPS into a processing chain
or generating auxiliary inputs.

:::{admonition} Processing Interface Control Document (ICD): `BIO-BPS-ICD-ARE-010113`
:class: note

**Version 3.2.3** &middot; 29 September 2025

[Download the ICD (PDF)](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_ICD_v3_2_3.pdf)
:::

:::{admonition} Processing Input & Output Data Definition (IODD): `BIO-BPS-IODD-ARE-010112`
:class: note

**Version 3.1.2** &middot; 29 September 2025

[Download the IODD (PDF)](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_IODD_v3_1_2.pdf)
:::

:::{admonition} BPS Auxiliary Products Format: `BIO-BPS-AUX-FMT-ARE-010163`
:class: note

**Version 3.6.1** &middot; 2 April 2026

[Download the Auxiliary Products Format (PDF)](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AUX_FMT_v3_6_1.pdf)
:::

For the full list of applicable documents (release note, ATBDs, SUM), see
[About → Applicable documents](../about/applicable-documents.md).

```{toctree}
:caption: Developer Guide
:maxdepth: 2
:hidden:

architecture
code-standards
documentation-standards
```
