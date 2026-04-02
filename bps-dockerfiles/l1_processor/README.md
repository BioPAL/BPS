# BIOMASS Processing Suite (BPS)
## L1_P Processor
### Installation package README file

#### Package content
- `pkgs`: conda packages

#### Prerequisites
- internet connection is required
- conda-build  (`conda install conda-build`)

#### Installation procedure

1. Create a dedicated conda environment with python (suggested: `python=3.12`)

2. Initialize a local conda channel

    ```sh
    mkdir -p local_l1p_channel && cp -r ./pkgs/* local_l1p_channel && conda index --verbose ${PWD}/local_l1p_channel
    ```

3. Configure conda channels priorities

    In order to ensure that the proper channels are used in the correct order execute:
    ```sh
    conda config --prepend channels conda-forge && conda config --prepend channels ${PWD}/local_l1p_channel
    ```

4. Install the L1_P Processor

    ```sh
    conda install bps-l1_processor
    ```

#### Execution procedure
The L1_P Processor can be executed with
```sh
bps_l1_processor BIO_L1_P_02.21_0001_20220101T000000_JobOrder.xml
```

More information is available in the help
```sh
bps_l1_processor --help
```
or in the documentation.
