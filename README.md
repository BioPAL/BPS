# BIOMASS Processing Suite (BPS)

The BIOMASS Product Algorithm Laboratory hosts official tools for processing and analysing ESA\'s BIOMASS mission data.

- Website: [www.biomass-disc.info](https://www.biomass-disc.info/)
- Documentation: [www.biomass-disc.info/release_note](https://www.biomass-disc.info/release_note)
- Repository: [github.com/BioPAL/BPS](https://github.com/BioPAL/BPS)
- Bug reports: [github.com/BioPAL/BPS/issues](https://github.com/BioPAL/BPS/issues)


# Objective

BIOMASS is ESA's (European Space Agency) seventh Earth Explorer mission, launched in 2025. The satellite is the first P-band SAR (Synthetic Aperture Radar) sensor in space and operates in fully polarimetric interferometric and tomographic modes. The mission main aim is to map forest properties globally, but the sensor also allows exploring subsurface scenarios (ice, desert).

The BIOMASS Processing Suite (BPS) is the collection of operational processors for BIOMASS mission, in charge of processing BIOMASS Level-0 data up to Level-3, generating a wide set of products.

This repository collects the BPS source code, for all the processing levels. More details about BIOMASS processors and products can be found [here](https://lps25.esa.int/lps25-presentations/presentations/827/_827.pdf#:~:text=Forest%20Height%20Obtained%20by%20processing%20an%20entire%20L1C%20stack%20(TOM%20or%20INT).).


# Structure of the Project

This repository is organized as follows:

- BPS L1 Processor
  - **bps-l1_processor**: BPS L1 Processor main package
  - **bps-l1_pre_processor**: Pre-processing package
  - **bps-l1_core_processor**: Core processing package
- BPS Stack Processor
  - **bps-stack_processor**: BPS Stack Processor main package
  - **bps-stack_pre_processor**: Pre-processing package
  - **bps-stack_coreg_processor**: Coregistration package
  - **bps-stack_cal_processor**: Calibration package
- BPS L2A Processor
  - **bps-l2a_processor**: BPS L2A Processor main package
- BPS L2B Processors (FD, FH, AGB)
  - **bps-l2b_fd_processor**: BPS L2B FD Processor main package
  - **bps-l2b_fh_processor**: BPS L2B FH Processor main package
  - **bps-l2b_agb_processor**: BPS L2B AGB Processor main package
- BPS L1 Framing Processor (support tool for the implementation of L0 slicing and L1 framing strategy)
  - **bps-l1_framing_processor**: BPS L1 Framing Processor main package
- BPS common dependencies
  - **bps-common**: BPS processors common utilities package
  - **bps-transcoder**: BPS products I/O package


# Getting Started

Here below are reported the instructions to install and use BPS processors starting from official conda packages (for those who may want to use BPS) or from source code repository (for those who may want to develop it). For more details refer to official [documentation](https://www.biomass-disc.info/release_note).

Both installation and usage procedures make use of the open-source package management system [conda](https://docs.conda.io/projects/conda/en/latest/), to be pre-installed.

## Install BPS (for users)

1. Create an empty BPS environment and activate it:

    ```bash
    conda create --name bps-env python=3.12
    conda activate bps-env
    ```

2. Install BPS conda packages. Note that you don't have to install all the BPS processors to make it work, but only those that you are interested in:

    ```bash
    conda install biomass-bps::bps-l1_processor           # only for BPS L1 Processor
    conda install biomass-bps::bps-stack_processor        # only for BPS Stack Processor
    conda install biomass-bps::bps-l2a_processor          # only for BPS L2A Processor
    conda install biomass-bps::bps-l2b_fd_processor       # only for BPS L2B FD Processor
    conda install biomass-bps::bps-l2b_fh_processor       # only for BPS L2B FH Processor
    conda install biomass-bps::bps-l2b_agb_processor      # only for BPS L2B AGB Processor
    ```

3. Check proper installation running help command:

    ```bash
    bps_l1_processor --help
    bps_stack_processor --help
    bps_l2a_processor --help
    bps_l2b_fd_processor --help
    bps_l2b_fh_processor --help
    bps_l2b_agb_processor --help
    ```

## Install BPS (for developers)

Open a command window with *conda* available and follow this procedure:

1. Clone repository

2. Create an empty BPS environment and activate it:

    ```bash
    conda create --name bps-dev-env python=3.12
    conda activate bps-dev-env
    ```

3. Install GDAL library:

    ```bash
    conda install -c conda-forge GDAL=3.10
    ```

4. Install BPS packages. Note that you don't have to install all the BPS processors to make it work, but only those that you are interested in. The installation order reported below ensures that inter-dependencies will be correctly resolved:

    ```bash
    pip install -e ./bps-common
    pip install -e ./bps-transcoder
    pip install -e ./bps-l1_pre_processor        # only for BPS L1 Processor
    pip install -e ./bps-l1_core_processor       # only for BPS L1 Processor
    pip install -e ./bps-l1_processor            # only for BPS L1 Processor
    pip install -e ./bps-stack_pre_processor     # only for BPS Stack Processor
    pip install -e ./bps-stack_coreg_processor   # only for BPS Stack Processor
    pip install -e ./bps-stack_cal_processor     # only for BPS Stack Processor
    pip install -e ./bps-stack_processor         # only for BPS Stack Processor
    pip install -e ./bps-l2a_processor           # only for BPS L2A Processor
    pip install -e ./bps-l2b_fd_processor        # only for BPS L2B FD Processor
    pip install -e ./bps-l2b_fh_processor        # only for BPS L2B FH Processor
    pip install -e ./bps-l2b_agb_processor       # only for BPS L2B AGB Processor
    ```

5. (for L1 and Stack Processors only) Recover binary dependencies from BPS bundle package (you can find it [here](https://service.aresys.it/downloads/)), copy them into a desired folder and add it to your path:

    ```bash
    export PATH=${PATH}:/path/to/l1/binaries/folder
    export PATH=${PATH}:/path/to/stack/binaries/folder
    ```

## Run BPS

In order to run a BPS processor you need:
- the input and auxiliary products you want to process (they can be found on [MAAP Explorer](https://explorer.maap.eo.esa.int/))
- the internal resources of the processor you want to use
- a dedicated `JobOrder.xml` file

Detailed information on how to recover internal resources and on how to create a Job Order file can be found in BPS [Software User Manual](https://www.biomass-disc.info/release_note).

Once ready, open a command window with *conda* available and run the desired BPS processor:

```bash
bps_l1_processor /path/to/JobOrder.xml
bps_stack_processor /path/to/JobOrder.xml
bps_l2a_processor /path/to/JobOrder.xml
bps_l2b_fd_processor /path/to/JobOrder.xml
bps_l2b_fh_processor /path/to/JobOrder.xml
bps_l2b_agb_processor /path/to/JobOrder.xml
```


# Call for Contributions

BPS is an open source project supported by a community who appreciates help from a wide range of different backgrounds. Large or small, any contribution makes a big difference; and if you\'ve never contributed to an open source project before, we hope you will start with BPS!

Beyond enhancing the processing algorithms, there are many ways to contribute:

- Submit a bug report or feature request on GitHub issues
- Contribute a Jupyter notebook to our examples gallery
- Assist us with user testing
- Add to the documentation or help with our website
- Write unit or integration tests for our project
- Answer questions on our issues, MAAP Forums, and elsewhere
- Write a blog post, tweet, or share our project with others
- Teach someone how to use BPS

As you can see, there are lots of ways to get involved and we would be very happy for you to join us! The only thing we ask is that you abide by the principles of openness, respect, and consideration of others as described in our Code of Conduct.

If you are interested in contributing, please not that a more complete and detailed contributor\'s guide in under preparation.


# Documentation

Documentation, including BPS Software User Manual, can be found in [www.biomass-disc.info/release_note](https://www.biomass-disc.info/release_note)


# History

BPS was originally written and is currently maintained by ARESYS team on behalf of ESA as part of BIOMASS DISC project.

DISC team includes reperesentatives of several european research institutions, see [DISC website](https://www.biomass-disc.info/) for more details.


# Citing

If you use BPS, please add these citations:

-   *BPS: BIOMASS Processing Suite, https://github.com/BioPAL/BPS*
