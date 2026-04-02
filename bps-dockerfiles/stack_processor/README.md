# BIOMASS Processing Suite (BPS)
## STA_P Processor
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
    mkdir -p local_stack_channel && cp -r ./pkgs/* local_stack_channel && conda index --verbose ${PWD}/local_stack_channel
    ```

3. Configure conda channels priorities

    In order to ensure that the proper channels are used in the correct order execute

    ```sh
    conda config --prepend channels conda-forge && conda config --prepend channels ${PWD}/local_stack_channel
    ```

4. Install the STA_P Processor

    ```sh
    conda install bps-stack_processor
    ```

#### Execution procedure
The STA_P Processor can be executed with
```sh
bps_stack_processor BIO_STA_P_02.21_0001_20220101T000000_JobOrder.xml
```

More information is available in the help
```sh
python -m bps.stack_processor --help
```
or in the documentation.
