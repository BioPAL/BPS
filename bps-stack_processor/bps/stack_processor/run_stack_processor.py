# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Processor Runner
----------------------
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from timeit import default_timer

from bps.common import bps_logger
from bps.common.common import retrieve_aux_product_data_single_content
from bps.common.fnf_utils import retrieve_fnf_content
from bps.common.processor_init import get_intermediate_data_dir_name, working_directory
from bps.stack_processor import BPS_STACK_PROCESSOR_NAME
from bps.stack_processor import __version__ as VERSION
from bps.stack_processor.execution.stack_cal_execution_manager import (
    StackCalExecutionManager,
)
from bps.stack_processor.execution.stack_coreg_execution_manager import (
    StackCoregProcessorExecutionManager,
)
from bps.stack_processor.execution.stack_preproc_execution_manager import (
    StackPreProcessorExecutionManager,
)
from bps.stack_processor.interface.external.aux_pps import (
    log_aux_pps_summary,
    validate_aux_pps_parameters,
    validate_aux_pps_xsd_schema,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.external.l1c_export import export_l1c_products
from bps.stack_processor.interface.external.parsing import (
    parse_aux_pps,
    parse_stack_job_order,
)
from bps.stack_processor.interface.external.translate_job_order import (
    InvalidStackJobOrder,
)
from bps.stack_processor.interface.external.utils import (
    get_bps_logger_level,
    parse_max_num_worker_threads,
)
from bps.stack_processor.interface.internal.intermediates import (
    INTERMEDIATE_FOLDER,
    INTERMEDIATE_FOLDER_TAG,
    STACK_CAL_PROC_INTERMEDIATE_FOLDER,
    STACK_COREG_PROC_INTERMEDIATE_FOLDER,
    STACK_PRE_PROC_INTERMEDIATE_FOLDER,
    CoregistrationOutputProducts,
    StackPreProcessorOutputProducts,
)
from bps.stack_processor.interface.internal.working_directory import (
    raise_if_aux_pps_is_not_compatible_for_resume,
    raise_if_job_order_is_not_compatible_for_resume,
    remove_intermediate_outputs,
    store_current_aux_pps,
    store_current_job_order,
)
from bps.stack_processor.utils.memorycheck import (
    raise_if_disk_memory_allocated_is_insufficient,
    raise_if_memory_allocated_is_insufficient,
)
from bps.stack_processor.utils.numba_utils import precompile_numba


def stack_processor_main(
    job_order_file: str,
    working_dir: str | None,
):
    """
    The stack processor main executor.

    Parameters
    ----------
    job_order_file: Path
        Path to the (existing) job order file.

    working_dir: Path
        Path to the working directory.

    """
    # Prepare the working directory.
    working_dir_path, keep_intermediates, job_order_path, job_order = stack_processor_setup(job_order_file, working_dir)
    if not working_dir_path.is_dir():
        raise RuntimeError(f"Invalid working directory {working_dir_path}")

    # Kick off the processor.
    with working_directory(working_dir_path):
        run_stack_processor(working_dir_path, keep_intermediates, job_order_path, job_order)


def stack_processor_setup(job_order_file: str, working_dir: str | None) -> tuple[Path, bool, Path, StackJobOrder]:
    """
    Setup the STA_P workspace. The intermediate working directory is
    defined as follows:

    * If JobOrder specifies Intermediate_Output_Enabled=true:
      - If --working-dir is passed via command line, use that.
      - Otherwise, use path specified by the IntermediateDataDir tag in
        the JobOrder.
    * If JobOrder specifies Intermediate_Output_Enabled=false:
      - If --working-dir is passed via command line, use that.
      - Otherwise, the working dir (with a pre-defined name) will be created
        in the same folder of the JobOrder.

    Parameters
    ----------
    job_order_file: str
        The job order path.

    working_dir: str | None
        Optionally, a working directory.

    Raises
    ------
    ValueError, InsufficientDiskSpaceAllocationError

    Return
    ------
    working_dir: Path
        Path to the working directory

    keep_intermediates: bool
        Whether the intermediates should be kept or not.

    job_order_path: Path
        Path to the Job order.

    job_order: StackJobOrder
        The parsed stack job order object.

    """
    # Setup the logger first.
    bps_logger.init_logger(processor="STA_P", task="STA_P", version=VERSION)
    bps_logger.enable_console_logging()

    # The Job order path.
    job_order_path = Path(job_order_file).absolute()
    if not job_order_path.exists():
        raise ValueError(f"invalid job order file {job_order_file}")

    # Parse the Job Order.
    #
    # NOTE: Job order output directories should be defined either by absolute paths
    # or relatively to the job-order parent path.
    job_order = parse_stack_job_order(job_order_path.read_text(encoding="utf-8"))
    job_order.output_path.output_directory = _resolve_path(
        job_order.output_path.output_directory,
        parent=job_order_path.parent,
    )

    # Verify that the allocated disk space is enough to write data in
    # intermediate breakpoint directory (and output as well).
    raise_if_disk_memory_allocated_is_insufficient(job_order)

    # Intermediates has to be kept only if specified by the flag and in the
    # list of intermediate files.
    keep_intermediates = (
        job_order.processor_configuration.keep_intermediate and INTERMEDIATE_FOLDER_TAG in job_order.intermediate_files
    )

    # The intermediate directory specified in the JobOrder. If the user has
    # requested to enable the intermediate directory and listed the
    # intermediate files, they cannot leave the Intermediate_Output_Dir blank.
    job_order_intermediates_dir = job_order.intermediate_files.get(INTERMEDIATE_FOLDER_TAG, INTERMEDIATE_FOLDER)
    if keep_intermediates and job_order_intermediates_dir == "":
        raise InvalidStackJobOrder(
            "<Intermediate_Output_Enable> is set to 'true' but <Intermediate_Output_Dir> is blank."
        )

    # Implement the working directory's generation policy.
    #
    #  - If Intermediate_Output_Enable=True, use what specified by
    #    Intermediate_Output_Dir, unelss specified by the user via command
    #    line argument.
    #  - If Intermediate_Output_Enable=False, use a default directory,
    #    unless specified by the user via command line argument.
    #
    if job_order.processor_configuration.keep_intermediate:
        if working_dir is None:
            working_dir = _resolve_path(
                Path(job_order_intermediates_dir),
                parent=job_order_path.parent,
            )
    elif working_dir is None:
        working_dir = get_intermediate_data_dir_name(
            job_order_path,
            add_data_dir=True,
        )

    # Setup the working directory and the log file.
    working_dir_path = Path(working_dir).absolute()
    working_dir_path.mkdir(exist_ok=True, parents=True)
    bps_logger.enable_file_logging(working_dir_path)

    # Report the log.
    bps_logger.info("%s started", BPS_STACK_PROCESSOR_NAME)
    bps_logger.info("Working directory: %s", working_dir_path)
    bps_logger.info("Job order file: %s", job_order_path)

    return working_dir_path, keep_intermediates, job_order_path, job_order


def run_stack_processor(
    working_dir: Path,
    keep_intermediates: bool,
    job_order_path: Path,
    job_order: StackJobOrder,
):
    """
    Run the stack processor as prescribed in the job order.

    Parameters
    ----------
    working_dir: Path
        Path to the working directory.

    keep_intermediates: bool
        Keep or not the intermediates products.

    job_order_path: Path
        Path to the (existing) job order file.

    job_order: StackJobOrder
        The parsed stack job order object.

    Raises
    ------
    InvalidAuxFnFProductError, InvalidAuxProduct, InsufficientMemoryAllocationError

    """
    # Run the stack processor.
    stack_start_time = default_timer()
    bps_logger.info("%s started", BPS_STACK_PROCESSOR_NAME)

    bps_logger.update_logger(
        loglevel=get_bps_logger_level(
            job_order.processor_configuration.stdout_log_level,
            job_order.processor_configuration.stderr_log_level,
        )
    )

    # If the AUX-PPS product is empty, this will raise an error.
    aux_pps_path = retrieve_aux_product_data_single_content(
        _resolve_path(job_order.auxiliary_files, parent=job_order_path.parent)
    )
    validate_aux_pps_xsd_schema(aux_pps_path)

    # Parse the AUX-PPS and log a summary.
    aux_pps = parse_aux_pps(aux_pps_path.read_text(encoding="utf-8"))
    validate_aux_pps_parameters(aux_pps)
    log_aux_pps_summary(aux_pps)

    # Kick-off the Numba compilation.
    numba_compilation_task = precompile_numba(aux_pps)

    # Verify that the allocated RAM memory is compatible with the current
    # configuration.
    raise_if_memory_allocated_is_insufficient(job_order, aux_pps)

    # Verify that a possible previous job order in the current working directory
    # is compatible with the current one.
    raise_if_job_order_is_not_compatible_for_resume(current_job_order=job_order, working_dir=working_dir)
    # Store current job order for next run.
    store_current_job_order(current_job_order_path=job_order_path, working_dir=working_dir)

    # Verify that a possible previous AUX-PPS in the current working directory
    # is compatible with the current one.
    raise_if_aux_pps_is_not_compatible_for_resume(current_aux_pps=aux_pps, working_dir=working_dir)
    # Store current AUX-PPS for next run.
    store_current_aux_pps(current_aux_pps_path=aux_pps_path, working_dir=working_dir)

    # Possibly retrieve the FNF mask.
    fnf_mask_file = None
    if job_order.external_products.fnf_database_entry_point is not None:
        fnf_mask_file = retrieve_fnf_content(
            _resolve_path(
                job_order.external_products.fnf_database_entry_point,
                parent=job_order_path.parent,
            )
        )
        if fnf_mask_file is None:
            bps_logger.warning(
                "Could not retrieve any FNF mask from %s",
                job_order.external_products.fnf_database_entry_point,
            )
        else:
            bps_logger.info(f"FNF mask file: {fnf_mask_file}")

    # Paths initialization for the pre-processor's breakpoints.
    stack_pre_proc_brk_folder = working_dir / STACK_PRE_PROC_INTERMEDIATE_FOLDER
    stack_pre_proc_brk_folder.mkdir(exist_ok=True)

    # Paths initialization for the coregistrator's breakpoints.
    stack_coreg_proc_brk_folder = working_dir / STACK_COREG_PROC_INTERMEDIATE_FOLDER
    stack_coreg_proc_brk_folder.mkdir(exist_ok=True)

    # Path initialization to the calibrator's breakpoints.
    stack_cal_proc_brk_folder = working_dir / STACK_CAL_PROC_INTERMEDIATE_FOLDER
    stack_cal_proc_brk_folder.mkdir(exist_ok=True)

    # Number of worker threads.
    num_worker_threads = parse_max_num_worker_threads(aux_pps, job_order.device_resources)
    bps_logger.info("Stack processor running with %d worker threads", num_worker_threads)

    # STA_P Step 1: Run the pre-processor.
    #  * Pack the stack data,
    #  * Compute RFI degradation,
    #  * Compute Faraday Rotation,
    #  * Cross-pol merging,
    #  * Compute primary image index,
    #  * Export DEM and data product folders.

    preprocessor_start_time = default_timer()
    bps_logger.info("Run Pre-Processor step")

    stack_pre_proc_output_products = list(
        StackPreProcessorOutputProducts.from_intermediate_dir(
            stack_pre_proc_brk_folder,
            product.name,
            index,
        )
        for index, product in enumerate(job_order.input_stack)
    )

    stack_pre_processor_execution_manager = StackPreProcessorExecutionManager(
        job_order=job_order,
        aux_pps=aux_pps,
    )
    stack_pre_proc_exec_products = stack_pre_processor_execution_manager.run(
        stack_pre_proc_output_products=stack_pre_proc_output_products,
        breakpoint_dir=stack_pre_proc_brk_folder,
        num_worker_threads=num_worker_threads,
    )

    preprocessor_end_time = default_timer()
    bps_logger.info(
        "Pre-processor successfully completed. Elapsed time [h:mm:ss]: %s",
        timedelta(seconds=preprocessor_end_time - preprocessor_start_time),
    )

    # STA_P step 2: Run the coregistration module.
    #  * Coregister data,
    #  * Shift LUTs.

    coregistration_start_time = default_timer()
    bps_logger.info("Run the Coregistration/LUT-shifting module")
    bps_logger.info(
        "Coreg primary product geocoded onto %s",
        stack_pre_proc_exec_products["l1a_product_data"][
            stack_pre_proc_exec_products["coreg_primary_image_index"]
        ].height_model.value.name,
    )

    stack_coreg_proc_output_products = list(
        CoregistrationOutputProducts.from_intermediate_dir(
            stack_coreg_proc_brk_folder,
            product.name,
            index,
        )
        for index, product in enumerate(job_order.input_stack)
    )

    stack_coreg_processor_execution_manager = StackCoregProcessorExecutionManager(
        job_order=job_order,
        aux_pps=aux_pps,
        breakpoint_dir=stack_coreg_proc_brk_folder,
    )
    stack_coreg_exec_products = stack_coreg_processor_execution_manager.run_coregistration(
        stack_pre_proc_output_products=stack_pre_proc_output_products,
        stack_coreg_proc_output_products=stack_coreg_proc_output_products,
        stack_pre_proc_exec_products=stack_pre_proc_exec_products,
        num_worker_threads=num_worker_threads,
    )
    lut_shift_exec_products = stack_coreg_processor_execution_manager.run_lut_shifting(
        stack_coreg_proc_output_products=stack_coreg_proc_output_products,
        stack_coreg_proc_exec_products=stack_coreg_exec_products,
        stack_pre_proc_exec_products=stack_pre_proc_exec_products,
    )

    # Removing the pre-processor intermediates output, unless requested by user.
    if not keep_intermediates:
        bps_logger.info("Removing pre-processor's intermediate outputs")
        remove_intermediate_outputs(stack_pre_proc_brk_dir=stack_pre_proc_brk_folder)

    coregistration_end_time = default_timer()
    bps_logger.info(
        "Coregistration successfully completed. Elapsed time [h:mm:ss]: %s",
        timedelta(seconds=coregistration_end_time - coregistration_start_time),
    )

    # STA_P step 3: Calibration.
    #  * azimuthSpectralFilter,
    #  * slowIonosphereRemoval,
    #  * inSarCalibration,
    #  * skpCalibration (i.e. Sum-of-Kronecker-Products).

    calibration_start_time = default_timer()
    bps_logger.info("Run the Calibration module")

    # Make sure that the numba pre-compilation is complete.
    numba_compilation_task.join()
    bps_logger.info("Numba code successfully compiled")

    stack_cal_execution_manager = StackCalExecutionManager(
        job_order=job_order,
        aux_pps=aux_pps,
        breakpoint_dir=stack_cal_proc_brk_folder,
        fnf_mask_path=fnf_mask_file,
    )
    stack_cal_proc_exec_products = stack_cal_execution_manager.run(
        stack_pre_proc_output_products=stack_pre_proc_output_products,
        stack_pre_proc_exec_products=stack_pre_proc_exec_products,
        stack_coreg_proc_output_products=stack_coreg_proc_output_products,
        stack_coreg_proc_exec_products=stack_coreg_exec_products,
        lut_shift_exec_products=lut_shift_exec_products,
        num_worker_threads=num_worker_threads,
    )

    calibration_end_time = default_timer()
    bps_logger.info(
        "Calibration successfully completed. Elapsed time [h:mm:ss]: %s",
        timedelta(seconds=calibration_end_time - calibration_start_time),
    )

    # Clean the pre-processor working directory.
    if not keep_intermediates:
        bps_logger.info("Removing coreg processor's intermediate outputs")
        remove_intermediate_outputs(stack_coreg_proc_brk_dir=stack_coreg_proc_brk_folder)

    # STA_P step 4: Export the L1c products.
    l1c_writing_start_time = default_timer()
    bps_logger.info("Exporting L1c products to %s", job_order.output_path.output_directory)

    export_l1c_products(
        job_order=job_order,
        aux_pps=aux_pps,
        stack_pre_proc_exec_products=stack_pre_proc_exec_products,
        stack_coreg_proc_output_products=stack_coreg_proc_output_products,
        stack_coreg_exec_products=stack_coreg_exec_products,
        lut_shift_exec_products=lut_shift_exec_products,
        stack_cal_proc_exec_products=stack_cal_proc_exec_products,
        fnf_mask_file=fnf_mask_file,
        gdal_num_threads=num_worker_threads,
    )

    l1c_writing_end_time = default_timer()
    bps_logger.info(
        "Successfully exported L1c products. Elapsed time [h:mm:ss]: %s",
        timedelta(seconds=l1c_writing_end_time - l1c_writing_start_time),
    )

    # Removing the intermediates output, unless requested by user.
    if not keep_intermediates:
        bps_logger.info("Removing the intermediate outputs")
        remove_intermediate_outputs(stack_working_dir=working_dir)

    stack_end_time = default_timer()
    bps_logger.info(
        "%s completed [h:mm:ss]: %s",
        BPS_STACK_PROCESSOR_NAME,
        timedelta(seconds=stack_end_time - stack_start_time),
    )


def _resolve_path(path: Path, *, parent: Path | None = None) -> Path:
    """Resolve a path relatively from an optional path."""
    if path.is_absolute():
        return path

    parent_path = parent if parent is not None else Path()
    return (parent_path / path).resolve()
