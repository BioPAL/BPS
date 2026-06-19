<!--
SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)

SPDX-License-Identifier: Apache-2.0
-->

# BIOMASS Processing Suite (BPS)
## L1_F Processor
### Installation package README file

#### Package content
- `pkgs`: conda packages
- `lib`: eo-cfi shared libraries

#### Prerequisites
- internet connection is required
- conda-build  (`conda install conda-build`)

#### Installation procedure

1. Create a dedicated conda environment with python (suggested: `python=3.12`)

2. Initialize a local conda channel

    ```sh
    mkdir -p local_l1f_channel && cp -r ./pkgs/* local_l1f_channel && conda index --verbose ${PWD}/local_l1f_channel
    ```

3. Configure conda channels priorities

    In order to ensure that the proper channels are used in the correct order execute

    ```sh
    conda config --prepend channels conda-forge && conda config --prepend channels ${PWD}/local_l1f_channel
    ```

4. Install the L1_F Processor

    ```sh
    conda install bps-l1_framing_processor
    ```

#### Execution procedure
The L1_F Processor can be executed with
```sh
bps_l1_framing_processor BIO_L1F_P_02.21_0001_20220101T000000_JobOrder.xml
```

More information is available in the help
```sh
bps_l1_framing_processor --help
```
or in the documentation.
