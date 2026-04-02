# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Processor Tests - Execution Tests
---------------------------------------
"""

# Import the aux-pps utilities.
# fmt: off
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "aux_pps_utils"))
# fmt: on
import glob
import os
import tempfile
import traceback

import arepyextras.xml as xml
from arepyextras.test import DataRepository, Environment, TestSession
from arepytools.timing.precisedatetime import PreciseDateTime
from aux_pps_utils import initialize_aux_pps
from bps.common import __version__
from bps.common.bps_logger import get_version_in_logger_format
from bps.common.common import retrieve_aux_product_data_single_content
from bps.transcoder.sarproduct.biomass_l1product_reader import BIOMASSL1ProductReader
from bps.transcoder.sarproduct.biomass_stackproduct import REQUIRED_LUT_LAYERS
from bps.transcoder.sarproduct.biomass_stackproduct_reader import (
    BIOMASSStackProductReader,
)

CURRENT_VERSION = get_version_in_logger_format(__version__)
OUTPUT_DIR = "od"  # Short name to keep the paths short enough for Windows.
WORKER_THREADS = 5
FRAMES_INT_TEST = 3

# Procuct name L1a frame, used to find TDS in the testplan dataset.
INT_FRAME_NAME = "BIO_S3_SCS__1S_20170225T015929_20170225T015940_I_G03_M03_C03_T000_F001_01_DNQUD0"

# Default print formats.
FAIL_FORMAT_STR = "  [ FAIL ] {:s}"
PASS_FORMAT_STR = "  [  OK  ] {:s}"


def test_bps_sta_001_int_nominal(session: TestSession, env: Environment, data: DataRepository):
    """
    Stack Processor execution in INT phase with nominal configuration
    on data of size 11k x 900.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )
    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_002_int_no_calibration(session: TestSession, env: Environment, data: DataRepository):
    """
    This tests aims at verifying that the mandatory LUTs are exported even if no
    calibration module is running.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )
    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_003_int_insufficient_RAM(session: TestSession, env: Environment, data: DataRepository):
    """
    This tests aims at verifying that the STA_P stops when not enough
    RAM is allocated by the job order..
    """
    max_ram_usage_MB = 20
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    job_order = xml.Document(job_order_filename)
    job_order.set("//List_of_Tasks/Task/Amount_of_RAM/text()", max_ram_usage_MB)

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    session.expect_run_failure(stack_processor, fatal=True)


def test_bps_sta_004_int_insufficient_disk_space(session: TestSession, env: Environment, data: DataRepository):
    """
    This tests aims at verifying that the STA_P stops when not enough
    disk space is allocated by the job order..
    """
    max_disk_usage_MB = 15
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)

    job_order = xml.Document(job_order_filename)
    job_order.set("//List_of_Tasks/Task/Disk_Space/text()", max_disk_usage_MB)

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    session.expect_run_failure(stack_processor, fatal=True)


def test_bps_sta_005_int_toi(session: TestSession, env: Environment, data: DataRepository):
    """
    This test verifies the correct execution of a stack task
    on when a TOI is provided in the JobOrder.
    """
    coreg_primary_image_index = 1
    aux_pps, job_order_filename, frame_paths = _initialize_stack_int_nominal_test(
        data, coreg_primary_index=coreg_primary_image_index
    )
    job_order = xml.Document(job_order_filename)

    # Enable the MSC but with phase removal only.
    aux_pps.set("//multiSquintCalibrationFlag/text()", "true")
    aux_pps.set("//disableMultiSquintIonoCorrectionFlag/text()", "true")

    # Retrieve the start/stop time of the primary.
    l1a_coreg_primary_annot = BIOMASSL1ProductReader(frame_paths[coreg_primary_image_index]).read(annotation_only=True)

    _set_toi(
        job_order,
        time_begin=l1a_coreg_primary_annot.start_time + 1.5,  # [UTC] + [s]
        time_end=l1a_coreg_primary_annot.stop_time - 1.5,  # [UTC] - [s].
    )

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def _initialize_stack_int_nominal_test(data, coreg_primary_index=1, calibration_primary_index=None):
    """Setup INT phase test with nominal configuration on data of size 11k x 900."""
    # Fetch the JobOrder.
    job_order_filename = data.pull("input/sta_int/joborder")
    job_order = xml.Document(job_order_filename)

    # Initialize the AUX-PPS.
    aux_pps_product_dir = data.pull("configurations/sta/CONF-BPS-PPS-001")
    aux_pps_path = retrieve_aux_product_data_single_content(aux_pps_product_dir)
    initialize_aux_pps(aux_pps_path)
    aux_pps = xml.Document(aux_pps_path)

    # Get list of frames in the CI-TDS.
    l1a_frame_list = _get_ci_frame_list(
        data,
        tds_path="product/sta_int/l1a/",
        product_name=INT_FRAME_NAME,
        only_first_n_frames=FRAMES_INT_TEST,
    )

    # Fill the job order.
    l1a_frame_env_list = _fill_ci_job_order_in_place(
        data,
        job_order,
        aux_pps_product_dir,
        coreg_primary_index=coreg_primary_index,
        calibration_primary_index=calibration_primary_index,
        Nimg=FRAMES_INT_TEST,
        frame_list=l1a_frame_list,
        bps_version=CURRENT_VERSION,
        output_dir=OUTPUT_DIR,
        worker_threads=WORKER_THREADS,
    )

    return aux_pps, job_order_filename, l1a_frame_env_list


def _read_l1c(root_path: Path):
    """Check by reading the L1c product and LUT ADS"""
    errors = []
    l1c_product_paths = glob.glob(os.path.join(root_path, OUTPUT_DIR, "BIO*"))
    if len(l1c_product_paths) == 0:
        errors.append(
            FAIL_FORMAT_STR.format(
                f"No L1c products were found in the output directory at {os.path.join(root_path, OUTPUT_DIR)}"
            )
        )

    for l1c_product_path in l1c_product_paths:
        try:
            stack_product = BIOMASSStackProductReader(l1c_product_path)
            stack_product.read()
        except:  # noqa: E722
            errors.append(FAIL_FORMAT_STR.format(f"Invalid L1c product {l1c_product_path}"))
            traceback.print_exc()

        try:
            luts, *_ = stack_product.read_lut_annotation()

            missing_luts = {lut_name for lut_name in set(REQUIRED_LUT_LAYERS) if lut_name not in luts}
            missing_luts.discard("coregistrationShiftsQuality")  # This one depends on the config.
            if len(missing_luts) > 0:
                errors.append(FAIL_FORMAT_STR.format(f"Missing mandatory LUTS: {missing_luts}"))
        except:  # noqa: E722
            errors.append(FAIL_FORMAT_STR.format(f"Ivalid L1c product LUTS {l1c_product_path}"))
            traceback.print_exc()

    if errors:
        raise RuntimeError("\n".join(errors))


def _fill_ci_job_order_in_place(
    data: DataRepository,
    job_order: xml.Document,
    aux_pps_filename: Path,
    *,
    coreg_primary_index: int,
    calibration_primary_index: int | None,
    Nimg: int,
    frame_list: list[Path],
    bps_version: str,
    output_dir: str,
    worker_threads: int,
    max_ram_usage_MB: int = 64000,
    max_disk_usage_MB: int = 55000,
) -> list[Path]:
    """
    Fill a CI job order in place, using arepyextras.xml.

    Parameters
    ----------
    data: DataRepository
        CI dataset.

    job_order: xml.Document
       The job order to fill.

    aux_pps_filename: xml.Document
        AUX-PPS path to write to the job order.

    coreg_primary_index: int
        Frame index to use for primary image.

    Nimg: int
        Numer of images in the test.

    frame_list: list[Path]
        The list of frames in the CI-TDS.

    bps_version: str
        The BPS SW version to write in the job order.

    output_dir: str
        The output_dir to write in the job order.

    worker_threads: int
        The number of worker threads to set in the job order.

    max_ram_usage_MB: int = 64000 [MB]
        The amount of max RAM (in Megabytes) to write in the job order.

    max_disk_usage_MB: int = 50000 [MB]
        The amout of disk usage (in Megabytes) to write in the job order.

    Return
    ------
    list[Path]
        The paths of the L1a input stack.

    """
    if coreg_primary_index >= Nimg:
        raise ValueError("Coregistration primary index must be smaller than Nimg.")

    # Pull the L1a input stack.
    l1a_frame_paths = [_pull_wrap(data, frame_list, i) for i in range(Nimg)]

    # Set the input stack.
    for n, l1a_frame_path in enumerate(l1a_frame_paths, start=1):
        job_order.set(
            f"//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[{n}]/text()",
            str(l1a_frame_path),
        )

    # Set the primary image.
    job_order.set(
        "//Proc_Parameter[Name/text()='primary_image']/Value",
        str(l1a_frame_paths[coreg_primary_index]),
    )

    # Populate the AUX-PPS.
    job_order.set(
        "//Input[Input_ID/text()='AUX_PPS___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        aux_pps_filename,
    )

    job_order.set("//Output/File_Dir/text()", output_dir)
    job_order.set("//Processor_Version/text()", bps_version)
    job_order.set("//Task_Version/text()", bps_version)
    job_order.set("//Number_of_CPU_Cores/text()", worker_threads)
    job_order.set("//List_of_Tasks/Task/Amount_of_RAM/text()", max_ram_usage_MB)
    job_order.set("//List_of_Tasks/Task/Disk_Space/text()", max_disk_usage_MB)
    job_order.set("//Cfg_File_Name/text()", data.pull("input/AUX-FNF"))

    return l1a_frame_paths


def _pull_wrap(
    data: DataRepository,
    frame_list: list[Path],
    frame_index: int,
) -> Path:
    """
    Create and return a symbolic link to the frame at frame_index,
    keeping its original name.
    """
    try:
        frame_path = data.pull(str(frame_list[frame_index]), str(frame_list[frame_index].name))

    except Exception as ex:
        raise RuntimeError(f"Failed to pull frame (error: {ex})") from ex

    return frame_path


def _get_ci_frame_list(
    data: DataRepository,
    *,
    tds_path: str,
    product_name: str,
    only_first_n_frames: int | None = None,
) -> list[Path]:
    """
    Get a list of frames in the data TDS.

    Parameters
    ----------
    data: DataRepository
        CI dataset.

    tds_path: str
       Path of the tds, relative to the root of the CI dataset.

    product_name: str
        Name of a product inside the tds.

    Raises
    ------
    RuntimeError

    Return
    ------
    list[Path]
        A list of first n, or all frames found at the tds_path in the CI dataset.

    """
    try:
        frame_list = sorted(
            [p for p in data.fetch_data(str(Path(tds_path).joinpath(product_name))).parent.parent.glob("BIO_*")]
        )

    except Exception as ex:
        raise RuntimeError(f"Failed to create list of frames (error: {ex})") from ex

    if only_first_n_frames is not None:
        frame_list = frame_list[:only_first_n_frames]

    return frame_list


def _set_toi(
    job_order_xml: xml.Document,
    *,
    time_begin: PreciseDateTime,
    time_end: PreciseDateTime,
):
    """Add a TOI to a job order."""
    # Add the element.
    job_order_xml.add("//Request", name="TOI")
    job_order_xml.add("//TOI", name="Start")
    job_order_xml.add("//TOI", name="Stop")

    # Populate the TOI element.
    job_order_xml.set("//Request/TOI/Start", time_begin.isoformat(timespec="microseconds"))
    job_order_xml.set("//Request/TOI/Stop", time_end.isoformat(timespec="microseconds"))
