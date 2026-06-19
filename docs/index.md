---
html_theme.sidebar_secondary.remove: true
---

<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# BIOMASS BPS documentation

The **BIOMASS Processing Suite (BPS)** is the open-source software that processes
Level 1, Level 2A, and Level 2B data from the ESA **BIOMASS** Earth Explorer
satellite mission.

**Version**: 0.2.0

<div class="brand-link-bar" markdown="1">

**Useful links:**  
[BIOMASS Mission](https://www.esa.int/Applications/Observing_the_Earth/FutureEO/Biomass) |
[BioPAL on GitHub](https://github.com/BioPAL) |
[Code Repository](https://github.com/BioPAL/BPS) |
[Issues](https://github.com/BioPAL/BPS/issues) |
[Discussions](https://github.com/BioPAL/BPS/discussions) |
[Releases](https://github.com/BioPAL/BPS/releases)

</div>

::::{grid} 1 1 2 2
:gutter: 3
:class-container: intro-grid

:::{grid-item-card}
:link: getting-started/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-regular fa-compass fa-fw" aria-hidden="true"></i> **Get started**
^^^

Installation, first steps, and how to find your way around BPS.
:::

:::{grid-item-card}
:link: user-guide/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-regular fa-book-open fa-fw" aria-hidden="true"></i> **User guide**
^^^

Software User Manual and authoritative user reference (PDF).
:::

:::{grid-item-card}
:link: science-guide/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-satellite fa-fw" aria-hidden="true"></i> **Science guide**
^^^

ATBDs and product format documents for every BPS processor.
:::

:::{grid-item-card}
:link: contributing/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-regular fa-handshake fa-fw" aria-hidden="true"></i> **Contribute**
^^^

Contribution workflow, standards, governance, and community channels.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:
:caption: For users

Get Started <getting-started/index>
User Guide <user-guide/index>
Tutorials <tutorials/index>
Science Guide <science-guide/index>
Communication <communication/index>
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: For contributors

Contributing <contributing/index>
Governance <governance/index>
About <about/index>
```
