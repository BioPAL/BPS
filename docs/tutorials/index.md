<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Tutorials

Hands-on, end-to-end walkthroughs of common BPS workflows. Each tutorial
assumes basic familiarity with a Linux/macOS terminal but does not require
prior BPS experience.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: intro-grid

:::{grid-item-card}
:link: run-bps-locally/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-laptop fa-fw" aria-hidden="true"></i> **Run BPS on your own computer**
^^^
Install BPS in a dedicated conda environment on your local workstation,
download a full Tomographic stack, and run the complete processing chain
from L1 framing to L2A. Three install paths covered: bundle, source
(developer), and a pointer to MAAP.

**Companion scripts and notebooks:**
[`docs/tutorials/run-bps-locally/`](https://github.com/BioPAL/BPS/tree/main/docs/tutorials/run-bps-locally)
:::

:::{grid-item-card}
:link: https://github.com/BioPAL/MAAP_BPS_scripts/tree/main/CODING
:link-type: url
:class-card: intro-card

<i class="doc-icon fa-solid fa-cloud fa-fw" aria-hidden="true"></i> **Run BPS on the ESA MAAP platform ↗**
^^^
Prefer not to install anything locally? ESA provides BPS pre-installable on
the [ESA MAAP](https://biomass.pal.maap.eo.esa.int/) JupyterLab platform.
Opens the BioPAL/MAAP_BPS_scripts walkthrough on GitHub.
:::

::::

```{admonition} Contributing a tutorial
:class: tip
New tutorials are welcome. Open a [Documentation issue](../contributing/index.md)
describing the use case, and pair the tutorial with a runnable notebook or
script when possible.
```

```{toctree}
:caption: Tutorials
:maxdepth: 2
:hidden:

Run BPS on your own computer <run-bps-locally/index>
```
