<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# 3. Install BPS from source (hybrid)

This path is for developers. You install the Python processors from this
repository in editable mode, and pull only the native binary packages from
the Aresys bundle.

:::{note}
A fully source-based install **is not possible** because the native
binaries (`bps-l1_binaries`, `bps-stack_binaries`, `libl1framing.so`) are
not in the public repository. They are delivered via the Aresys bundle.
:::

## 3.1 Create the environment

```bash
cd ~/bps-work/BPS
conda create -n bps-dev python=3.12
conda activate bps-dev
```

## 3.2 Install Python processors in editable mode

The Python source for every processor sits at the repo root under
`bps-common/`, `bps-l1_processor/`, etc. Install them in dependency order:

```bash
pip install -e bps-common
pip install -e bps-transcoder
pip install -e bps-l1_pre_processor
pip install -e bps-l1_core_processor
pip install -e bps-l1_framing_processor
pip install -e bps-l1_processor
pip install -e bps-stack_pre_processor
pip install -e bps-stack_coreg_processor
pip install -e bps-stack_cal_processor
pip install -e bps-stack_processor
pip install -e bps-l2a_processor
pip install -e bps-l2b_agb_processor
pip install -e bps-l2b_fh_processor
pip install -e bps-l2b_fd_processor
```

Any Python edit you make under `bps-l1_processor/bps/l1_processor/...` is
picked up immediately, with no rebuild step.

## 3.3 Install the native binary packages from the bundle

The `bps-l1_binaries` and `bps-stack_binaries` packages are not in this
repository (the source is not public). Extract them from the Aresys bundle
and install via conda:

```bash
tar -xzf ~/bps-work/SW/bps-bundle-v4.4.3.tar.gz -C ~/bps-work/SW/BPS_V443/
conda install -c "file://$HOME/bps-work/SW/BPS_V443/bundle/bps_conda_channel" \
    bps-l1_binaries bps-stack_binaries
```

## 3.4 Verify

```bash
which bps_l1_processor   # should resolve to your editable install
which libl1framing       # comes from bps-l1_binaries
bps_l1_processor --help
```

:::{admonition} If installation fails
:class: warning
Clean up the partially created environment before retrying:

```bash
conda env list
conda env remove --name bps-dev
conda env list
```

The most common failure causes are insufficient RAM (the conda solver
crashes when resolving all packages at once) or a missing Aresys
dependency. See
[`ModuleNotFoundError: arepytools`](07-reference.md#troubleshooting).
:::
