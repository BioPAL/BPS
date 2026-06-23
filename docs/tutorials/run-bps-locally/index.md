<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Run BPS on your own computer

This tutorial walks through installing and running the BIOMASS Processing
Suite (BPS) **locally**, on your own Linux or macOS workstation, from L1
framing through to L2A biomass estimation.

The companion scripts, notebooks, JobOrder templates and configuration files
live in this same folder:
[`docs/tutorials/run-bps-locally/`](https://github.com/BioPAL/BPS/tree/main/docs/tutorials/run-bps-locally).
They are adapted from
[BioPAL/MAAP_BPS_scripts](https://github.com/BioPAL/MAAP_BPS_scripts) and
ship with `<BPS_ROOT>` placeholders so you can drop them straight into your
local filesystem.

By the end of this tutorial you will have:

- BPS installed on your workstation (one of three install paths below)
- A full Tomographic stack (7 repeat passes) downloaded from the ESA
  catalogue
- The complete processing chain executed locally: L1F → L1 → STA → L2A

:::{admonition} Audience
:class: tip
Scientific users and developers who want to run BPS on their own hardware
(laptop or workstation) rather than on a cloud platform. Basic familiarity
with conda and a Linux/macOS terminal is assumed.
:::

:::{admonition} Need a cloud environment instead?
:class: note
ESA provides BPS pre-installable on the
[ESA MAAP](https://biomass.pal.maap.eo.esa.int/) platform. The
[MAAP_BPS_scripts README](https://github.com/BioPAL/MAAP_BPS_scripts/tree/main/CODING)
walks through that setup end-to-end.
:::

## Choose an install path

::::{tab-set}

:::{tab-item} A. Bundle (recommended)
**Who it's for:** scientific users who want a working BPS without modifying
processor code.

Installs all 14 processors at once from the Aresys-distributed conda
bundle. See [Install BPS from the bundle](02-install-bundle.md).
:::

:::{tab-item} B. Source (developer mode)
**Who it's for:** contributors who want to modify Python processor code and
test changes without rebuilding the bundle.

Installs the Python processors from this repository in editable mode
(`pip install -e .`), plus only the native binary packages from the bundle.
See [Install BPS from source](03-install-from-source.md).

A fully source-based install **is not possible** because the native
binaries (`bps-l1_binaries`, `bps-stack_binaries`, `libl1framing.so`) are
not in the public repository. They are delivered via the Aresys bundle.
:::

:::{tab-item} C. MAAP platform
**Who it's for:** users who don't want to install anything locally.

ESA provides BPS pre-installable on the
[ESA MAAP](https://biomass.pal.maap.eo.esa.int/) platform. Follow the
[MAAP_BPS_scripts README](https://github.com/BioPAL/MAAP_BPS_scripts/tree/main/CODING)
instead of this tutorial.
:::

::::

```{toctree}
:caption: Run BPS locally
:maxdepth: 1
:hidden:

1. Prerequisites <01-prerequisites>
2. Install from the bundle <02-install-bundle>
3. Install from source <03-install-from-source>
4. Download raw inputs <04-download-inputs>
5. Configure AUX files <05-aux-files>
6. Run the processing chain <06-run-the-chain>
7. Reference and troubleshooting <07-reference>
```
