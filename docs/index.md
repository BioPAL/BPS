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

**Useful links**:
[BIOMASS Mission](https://www.esa.int/Applications/Observing_the_Earth/FutureEO/Biomass) |
[Code Repository](https://github.com/BioPAL/BPS) |
[Issues](https://github.com/BioPAL/BPS/issues) |
[Discussions](https://github.com/BioPAL/BPS/discussions) |
[Releases](https://github.com/BioPAL/BPS/releases)

::::{grid} 1 1 2 2
:gutter: 3
:class-container: intro-grid

:::{grid-item-card}
:link: getting-started/index
:link-type: doc
:class-card: intro-card

🚀 **Get started**
^^^

*New to BIOMASS BPS?*
Start here with installation instructions and a brief overview of the suite.
:::

:::{grid-item-card}
:link: user-guide/index
:link-type: doc
:class-card: intro-card

📖 **User guide**
^^^

*Ready to deepen your understanding of BPS?*
Visit the user guide for detailed explanations of the data structures, common
processing patterns, and more.
:::

:::{grid-item-card}
:link: science-guide/index
:link-type: doc
:class-card: intro-card

🛰️ **Science Guide**
^^^

*Looking for the science behind a specific processor?*
Algorithm descriptions, inputs and outputs, validation, and references for every BPS processor.
:::

:::{grid-item-card}
:link: contributing/index
:link-type: doc
:class-card: intro-card

🤝 **Contribute**
^^^

*Saw a typo in the documentation? Want to improve an existing feature?*
Please review our guide on improving BPS.
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
