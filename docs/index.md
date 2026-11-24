---
html_theme.sidebar_secondary.remove: true
---

<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

:::{div} index-hero

<div class="index-hero__inner">
<div class="index-hero__logo-stage" markdown="0">
  <span class="index-hero__halo" aria-hidden="true"></span>
  <div class="index-hero__logo-motion">
    <img class="index-hero__logo" src="_static/logos/BioPAL.png" alt="" />
  </div>
</div>
<div class="index-hero__copy-col">
<div class="index-hero__copy" markdown="0">
  <p class="index-hero__eyebrow">BIOMASS Product Algorithm Laboratory</p>
  <h1 class="index-hero__title">BioPAL documentation</h1>
  <p class="index-hero__tagline">Open-source processing software for the ESA BIOMASS Earth Explorer mission.</p>
  <!-- bps-version-badge -->
</div>
</div>
</div>

:::

<div class="biopal-scope-banner" markdown="1">

**About this site.** [BioPAL](https://github.com/BioPAL) (_BIOMASS Product Algorithm Laboratory_)
hosts this documentation portal. All pages here cover **BPS** (BIOMASS Processing Suite) today.
Documentation for other BioPAL repositories will be added here as it becomes available.

</div>

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
