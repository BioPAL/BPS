These are the development instructions for manual conda packages generation.

# Repository setup

1. Install __git-lfs__ package in your system
2. Clone the bps repository

# Conda packages generation

1. Build the source distribution for each BPS package.

    For instance, the bps-common package is built by

    ```bash
    cd bps-common
    python -m build --sdist
    cd ..
    ```

    The source distribution files are inside each package folder in the __dist__ folder.

2. Create a local conda channel and initialize it with the conda packages of arepytools and arepyextras packages

    ```bash
    mkdir -p local_bps_channel
    cp -r /path/to/bundle/prereq/pkgs/* local_bps_channel
    conda-index local_bps_channel
    ```

3. For each package, build the conda package

    ```bash
    cd bps-common
    conda build recipe --output-folder ../local_bps_channel
    cd -
    ```

    This order ensures that inter-dependencies will be correctly resolved

    - bps-common
    - bps-transcoder
    - bps-l1_framing_processor
    - bps-l1_pre_processor
    - bps-l1_core_processor
    - bps-l1_processor
    - bps-stack_cal_processor
    - bps-stack_pre_processor
    - bps-stack_coreg_processor
    - bps-stack_processor
    - bps-l2a_processor
    - bps-l2b_fh_processor
    - bps-l2b_fd_processor
    - bps-l2b_agb_processor
