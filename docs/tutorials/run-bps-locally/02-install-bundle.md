<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# 2. Install BPS from the bundle

This is the recommended path for scientific users who want a working BPS
without modifying processor code. Installs all 14 processors at once from
the Aresys-distributed conda bundle.

## Download the BPS bundle

1. Go to [service.aresys.it/downloads](https://service.aresys.it/downloads/)
2. Request access if you do not have credentials yet (free, account-based)
3. Download the bundle matching the version you want, e.g.
   `bps-bundle-v4.4.3.tar.gz`
4. Place it in `~/bps-work/SW/`:

   ```bash
   mv ~/Downloads/bps-bundle-v4.4.3.tar.gz ~/bps-work/SW/
   mkdir -p ~/bps-work/SW/BPS_V443
   tar -xzf ~/bps-work/SW/bps-bundle-v4.4.3.tar.gz -C ~/bps-work/SW/BPS_V443
   ```

## Run the install notebook

Open
[`notebooks/0_BPS_installation.ipynb`](https://github.com/BioPAL/BPS/blob/main/docs/tutorials/run-bps-locally/notebooks/0_BPS_installation.ipynb)
in JupyterLab, VS Code, or any notebook front-end.

Set the BPS version at the top of cell 0:

```python
BPS_VERSION = "4.4.3"   # match your tarball
```

If your working directory is not `/home/jovyan/`, also edit the path
constants near the top of cell 0 to point at your local layout
(`~/bps-work/SW/`, `~/bps-work/BPS/docs/tutorials/run-bps-locally/`).

Run all cells. The notebook:

1. Creates a conda environment (e.g. `BPS_443`) with Python 3.12
2. Builds a local conda channel from the bundle packages
3. Installs all BPS processors (`bps-l1_processor`,
   `bps-l1_framing_processor`, `bps-stack_processor`,
   `bps-l2a_processor`, ...)
4. Patches `set_environment.bash` with the correct paths
5. Adds a shell alias to `~/.bashrc` (or `~/.zshrc`):

   ```bash
   alias bps443='source ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE/set_environment.bash'
   ```

6. Verifies the installation by running `--help` on every processor

## Verify

After install completes, open a fresh terminal and check:

```bash
bps443
which bps_l1_processor
bps_l1_processor --help
```

:::{admonition} If installation fails
:class: warning
A failed install often leaves a partially created conda environment, which
causes subsequent retries to fail. Clean up before retrying:

```bash
conda env list                       # 1. confirm the broken env exists
conda env remove --name BPS_443      # 2. remove it
conda env list                       # 3. confirm it is gone
```

Then re-run the install steps. If it fails again at the same point, the
most likely causes are insufficient RAM (the conda solver crashes when
resolving all packages at once) or a corrupted bundle.
:::
