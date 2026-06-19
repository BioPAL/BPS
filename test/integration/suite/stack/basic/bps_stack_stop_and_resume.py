# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Processor - Stop and Resume Tests
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
import signal
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable

import arepyextras.xml as xml
from arepyextras.test import DataRepository, Environment, TestSession
from aux_pps_utils import initialize_aux_pps
from bps.common import __version__
from bps.common.bps_logger import get_version_in_logger_format
from bps.common.common import retrieve_aux_product_data_single_content
from bps.stack_processor.execution.stack_coreg_execution_manager import (
    STOP_AND_RESUME_PATH as COREG_PROCESSOR_COMPLETED_PATH,
)
from bps.stack_processor.execution.stack_preproc_execution_manager import (
    STOP_AND_RESUME_PATH as PRE_PROCESSOR_COMPLETED_PATH,
)
from bps.stack_processor.interface.internal.intermediates import (
    STACK_COREG_PROC_INTERMEDIATE_FOLDER,
    STACK_PRE_PROC_INTERMEDIATE_FOLDER,
)
from bps.transcoder.sarproduct.biomass_stackproduct_reader import (
    BIOMASSStackProductReader,
)

CURRENT_VERSION = get_version_in_logger_format(__version__)

OUTPUT_DIR = "od"  # Just to keep it short for Windows tests.
WORKER_THREADS = 1
FRAMES_INT_TEST = 3

# Default print formats.
FAIL_FORMAT_STR = "  [ FAIL ] {:s}"
PASS_FORMAT_STR = "  [  OK  ] {:s}"

# Procuct names L1a frames, used to find TDSs in the testplan dataset.
TOM_REDUCED_FRAME_NAME = "BIO_S3_SCS__1S_20170213T015412_20170213T015417_T_G03_M03_C03_T000_F001_01_DNNMT7"
INT_FRAME_NAME = "BIO_S3_SCS__1S_20170225T015929_20170225T015940_I_G03_M03_C03_T000_F001_01_DNQUD0"


def test_bps_sta_001_stop_and_resume_with_working_dir(session: TestSession, env: Environment, data: DataRepository):
    """Stop and resume test with working dir on two frames (8400 x 900)."""
    aux_pps, job_order_filename = _initialize_stack_int_nocal_test(data)
    working_dir = Path(tempfile.mkdtemp(dir=env.root))
    _run_stop_and_resume_test(
        session=session,
        env=env,
        data=data,
        aux_pps=aux_pps,
        job_order_filename=job_order_filename,
        working_dir=working_dir,
        call_command=[
            "bps_stack_processor",
            "--working-dir",
            working_dir,
            job_order_filename,
        ],
    )


def test_bps_sta_002_stop_and_resume_no_working_dir(session: TestSession, env: Environment, data: DataRepository):
    """Stop and resume test without working dir on two frames (8400 x 900)."""
    aux_pps, job_order_filename = _initialize_stack_int_nocal_test(data)
    _run_stop_and_resume_test(
        session=session,
        env=env,
        data=data,
        aux_pps=aux_pps,
        job_order_filename=job_order_filename,
        working_dir=_get_working_dir_from_job_order(job_order_filename),
        call_command=[
            "bps_stack_processor",
            job_order_filename,
        ],
    )


def _run_stop_and_resume_test(
    *,
    session: TestSession,
    env: Environment,
    data: DataRepository,
    aux_pps: xml.Document,
    job_order_filename: xml.Document,
    working_dir: Path,
    call_command: list[str],
):
    """
    Run the steps:
       #1 Start excecution and stop when pre-processor completes.
       #2 Resume and stop when coregistration completes.
       #3 Resume and run to the end, with no stack modules.
       #4 Enable SKP module and resume.
           Check that pre-processor and coregistration are skipped.
       #5 Verify that execution fails, when AUX-PPSs are not compatible.
       #6 Verify that execution fails, when job orders are not compatible.

    """
    # Step 1. Start excecution and stop when pre-processor completes.
    process = run_sta_p(call_command=call_command, wait=False)

    # Stop when pre-processor completes.
    stack_coreg_complete_path = working_dir / STACK_PRE_PROC_INTERMEDIATE_FOLDER / PRE_PROCESSOR_COMPLETED_PATH
    _kill_on_condition(
        pid=process.pid,
        condition_fn=lambda: stack_coreg_complete_path.is_file(),
    )

    # Step 2. Resume and stop when coregistration completes.
    process = run_sta_p(call_command=call_command, wait=False)
    stack_coreg_complete_path = working_dir / STACK_COREG_PROC_INTERMEDIATE_FOLDER / COREG_PROCESSOR_COMPLETED_PATH
    _kill_on_condition(
        pid=process.pid,
        condition_fn=lambda: stack_coreg_complete_path.is_file(),
    )

    # Step 3. Resume and run to the end, with no stack modules.
    process = run_sta_p(call_command=call_command)
    # Check returncode 0
    session.expect_true(
        not _check_sta_process_returncode(process),
        fatal=True,
    )

    # Step 4. Enable SKP module and resume. Check that pre-processor and
    # coregistration are skipped.
    aux_pps.set("//skpPhaseEstimationFlag/text()", "true")
    process = run_sta_p(call_command=call_command)
    # Check returncode 0
    session.expect_true(
        not _check_sta_process_returncode(process),
        fatal=True,
    )
    # Check stdout for expected messages.
    session.expect_true(
        _check_for_str_in_stdout(
            process=process,
            log_message="Coregistration inputs already available",
        ),
        fatal=True,
    )
    session.expect_true(
        _check_for_str_in_stdout(
            process=process,
            log_message="Coregistration products already available",
        ),
        fatal=True,
    )
    session.expect_true(
        _check_for_str_in_stdout(
            process=process,
            log_message="skpPhaseCalibration completed",
        ),
        fatal=True,
    )

    _read_l1c(env.root)

    # Step 5. Verify that execution fails, when AUX-PPSs are not compatible.
    aux_pps.set("//coregistrationMethod/text()", "Geometry and Data")
    process = run_sta_p(call_command=call_command)
    # Check returncode different form 0, silence error print by expect_fail True.
    session.expect_true(
        _check_sta_process_returncode(process, expect_fail=True),
        fatal=True,
    )

    # Reset coregistrationMethod in AUX-PPS
    aux_pps.set("//coregistrationMethod/text()", "Geometry")

    # Step 6. Verify that execution fails, when job orders are not compatible.
    new_frame_index = 3
    xml.Document(job_order_filename).set(
        "//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[2]/text()",
        _pull_wrap(
            data,
            _get_ci_frame_list(
                data,
                tds_path="product/tom_ci_data_reduced/l1a/",
                product_name=TOM_REDUCED_FRAME_NAME,
            ),
            new_frame_index,
        ),
    )
    process = run_sta_p(call_command=call_command)
    # Check returncode different form 0, silence error print by expect_fail True.
    session.expect_true(
        _check_sta_process_returncode(process, expect_fail=True),
        fatal=True,
    )


def run_sta_p(
    *,
    call_command: list[str],
    stdout: int = subprocess.PIPE,
    stderr: int = subprocess.PIPE,
    wait: bool = True,
):
    """
    Run stack processor.

    Parameters
    ----------
    call_command: List[str]
        Commands used to call STA_P.

    stdout: int = subprocess.PIPE
        Silence but store stdout from the process call.
        Use None to print on termainal.

    stderr: int = subprocess.PIPE
        Silence but store stderr from the process call.

    wait: bool = True
        Whether to wait for the process to complete or not.

    Return
    ------
    process: subprocess.Popen

    """
    process = subprocess.Popen(call_command, stdout=stdout, stderr=stderr)
    if wait:
        process.wait()

    return process


def _check_sta_process_returncode(process: subprocess.Popen, expect_fail: bool = False) -> int:
    """Check returncode from the process, print stderr if not 0."""
    if process.returncode != 0 and not expect_fail:
        process_stderr = str(process.communicate()[1])
        assert process_stderr is not None
        print(process_stderr)
    return process.returncode


def _check_for_str_in_stdout(process: subprocess.Popen, log_message: str) -> bool:
    """Check if a log message is found in stdout of the process."""
    process_stdout = str(process.communicate()[0])
    assert process_stdout is not None
    return log_message in process_stdout


def _kill_on_condition(
    pid: int,
    condition_fn: Callable,
    timeout: int = 600,
):
    """
    Kill process of pid, when condition_fn True or after timeout [s].

    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_fn():
            _kill_process_by_pid(pid)
            return
        time.sleep(1)

    os.kill(pid, signal.SIGKILL)
    raise RuntimeError(f"Pooling for breakpoint reached timeout limit of {timeout} seconds")


def _kill_process_by_pid(process_pid: int):
    """Kill the process sending a SIGKILL signal."""
    try:
        os.kill(process_pid, signal.SIGKILL)
    except OSError:
        print(f"Could not kill STA_P (PID={process_pid})")


def _initialize_stack_int_nocal_test(data):
    """Initialize INT phase test with no stack calibration."""
    # Fetch the JobOrder.
    job_order_filename = data.pull("input/sta_int/joborder")
    job_order = xml.Document(job_order_filename)

    # Initialize the AUX-PPS.
    aux_pps_product_dir = data.pull("configurations/sta/CONF-BPS-PPS-001")
    aux_pps_path = retrieve_aux_product_data_single_content(aux_pps_product_dir)
    initialize_aux_pps(aux_pps_path)
    aux_pps = xml.Document(aux_pps_path)

    # Get list of frames in the TDS. Here we only use the first
    # three as this is an INT test.
    frame_list = _get_ci_frame_list(
        data,
        tds_path="product/sta_int/l1a",
        product_name=INT_FRAME_NAME,
        only_first_n_frames=FRAMES_INT_TEST,
    )

    # Fill the job order.
    _fill_ci_job_order_in_place(
        data,
        job_order=job_order,
        aux_pps_filename=aux_pps_product_dir,
        coreg_primary_index=1,
        Nimg=FRAMES_INT_TEST,
        frame_list=frame_list,
        bps_version=CURRENT_VERSION,
        output_dir=OUTPUT_DIR,
        worker_threads=WORKER_THREADS,
    )

    # Disable all stack modules.
    _disable_all_stack_modules(aux_pps)

    # Reset coregistration method in AUX-PPS.
    aux_pps.set("//coregistrationMethod/text()", "Geometry")

    return aux_pps, job_order_filename


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
            stack_product.read_lut_annotation()
        except:  # noqa: E722
            errors.append(FAIL_FORMAT_STR.format(f"Ivalid L1c product LUTS {l1c_product_path}"))
            traceback.print_exc()

    if errors:
        raise RuntimeError("\n".join(errors))


def _disable_all_stack_modules(aux_pps: xml.Document):
    """Set all stack modules to false in AUX-PPS file."""
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")


def _get_working_dir_from_job_order(job_order_filename: Path) -> Path:
    return Path(xml.Document(job_order_filename).get("//Intermediate_Output_File/text()"))


def _fill_ci_job_order_in_place(
    data: DataRepository,
    job_order: xml.Document,
    aux_pps_filename: Path,
    *,
    coreg_primary_index: int,
    Nimg: int,
    frame_list: list[Path],
    bps_version: str,
    output_dir: str,
    worker_threads: int,
    max_ram_usage_MB: int = 64000,
    max_disk_usage_MB: int = 35000,
):
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

    max_disk_usage_MB: int = 64000 [MB]
        The amout of disk usage (in Megabytes) to write in the job order.

    """
    # Pulling primary image and writing it to two fields in the joborder.
    primary_image = _pull_wrap(data, frame_list, coreg_primary_index)
    job_order.set(
        "//Proc_Parameter[Name/text()='primary_image']/Value",
        primary_image,
    )
    job_order.set(
        f"//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[{coreg_primary_index + 1}]/text()",
        primary_image,
    )

    # Pulling and writing remaining images to the joborder.
    for i in (n for n in range(0, Nimg) if n != coreg_primary_index):
        job_order.set(
            f"//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[{i + 1}]/text()",
            _pull_wrap(data, frame_list, i),
        )

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
