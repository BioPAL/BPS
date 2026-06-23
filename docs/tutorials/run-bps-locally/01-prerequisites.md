<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# 1. Prerequisites

## Hardware

| Resource | Minimum | Recommended |
|---|---|---|
| OS | Linux x86_64 (Ubuntu 22.04 or RHEL 9) or macOS 13+ | Linux x86_64 |
| RAM | 20 GB | 64 GB (4 GB per core) |
| Free disk (install) | 5 GB | 5 GB |
| Free disk (per processing run) | 35 GB | 100 GB+ for multiple runs |
| Shared memory (`/dev/shm` on Linux) | 10 GB | 10 GB+ |

Numbers are from section 4.1 of the
[BPS SUM v4.4.1](../../about/applicable-documents.md). RAM and shared
memory are **hard** requirements: BPS uses memory-mapped IO between
processors, and the L1 processor peaks at ~10 GB in `/dev/shm` when the
delete-on-consume option is on.

:::{warning}
On Linux, `/dev/shm` is sized to half of total RAM by default. With less
than 20 GB of RAM you may have less than 10 GB of shared memory and the L1
processor will fail. Either add RAM, or remount `/dev/shm` larger:

```bash
sudo mount -o remount,size=12G /dev/shm
```
:::

## Software

- **conda** or **mamba** (the BPS bundle is installed as a conda channel).
  Miniforge is recommended:
  [conda-forge.org/miniforge](https://conda-forge.org/miniforge/)
- **git** to clone this repository
- **unzip** and **tar** (preinstalled on most Linux/macOS systems)

:::{warning}
**Windows is not officially supported.** BPS depends on native shared
libraries built for Linux x86_64. On Windows, use WSL2 with an Ubuntu 22.04
distribution; the rest of this tutorial then applies inside the WSL shell.
:::

## Working directory

Throughout this tutorial we use `~/bps-work/` as the root working directory.
Replace it with any path of your choosing. Create the layout up front:

```bash
mkdir -p ~/bps-work/{SW,data}
cd ~/bps-work
git clone https://github.com/BioPAL/BPS
```

You should end up with:

```
~/bps-work/
├── BPS/                       this repository (clone)
│   └── docs/tutorials/run-bps-locally/  scripts and notebooks for this tutorial
├── SW/                        BPS bundle tarball will go here
└── data/                      raw input products and outputs
```

## Replace the `<BPS_ROOT>` placeholders

The shipped `config.ini` and `set_environment.bash` use `<BPS_ROOT>`
placeholders. Replace them once with your actual root path:

```bash
cd ~/bps-work/BPS/docs/tutorials/run-bps-locally
sed -i.bak "s|<BPS_ROOT>|$HOME/bps-work|g" scripts/config.ini CONFIGURATION_FILE/set_environment.bash
```

(On macOS, `sed -i.bak` produces `.bak` files; remove them after the
substitution succeeds.)
