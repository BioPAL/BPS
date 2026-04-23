# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Translation
-----------
"""

from pathlib import Path

from bps.common.io import joborder_models
from bps.common.l2_joborder_tags import L2B_OUTPUT_PRODUCT_FD
from bps.common.translate_job_order import (
    flatten_configuration_file,
    flatten_input_products_allow_multiple_products,
    flatten_output_products,
    flatten_processing_params,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_task,
)
from bps.l2b_fh_processor.core.joborder_l2b_fh import (
    L2bFHJobOrder,
    L2BFHProcessingParameters,
)


class InvalidJobOrder(ValueError):
    """Raised when failing to translate a joborder"""


EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass L2b FH processor"""

EXPECTED_PROCESSOR_NAME = "L2B_FH_P"
"""Processor name for Biomass L2b FH processor"""

EXPECTED_PROCESSOR_ALIAS_NO_FD = "L2B_FH_P_withoutFD"
"""Alternative alias for Biomass L2b FH processor, without optional L2b FD in input"""

EXPECTED_PROCESSOR_ALIAS_YES_FD = "L2B_FH_P_withFD"
"""Alternative alias for Biomass L2b FH processor, with optional L2b FD in input"""

EXPECTED_PROCESSOR_NAMES_LIST = [
    EXPECTED_PROCESSOR_NAME,
    EXPECTED_PROCESSOR_ALIAS_NO_FD,
    EXPECTED_PROCESSOR_ALIAS_YES_FD,
]
"""All possible alias for Biomass L2b FH processors"""

EXPECTED_PROCESSOR_VERSION = "04.44"
"""Processor version for Biomass L2b FH processor"""

EXPECTED_TASK_NAME = EXPECTED_PROCESSOR_NAME
"""Task name for Biomass L2b FH processor"""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass L2b FH processor"""

L2B_OUTPUT_PRODUCT_FH = "FP_FH__L2B"

L2B_OUTPUT_PRODUCT_TFH = "FP_TFH_L2B"

L2A_PRODUCT_FH = "FP_FH__L2A"

L2A_PRODUCT_TFH = "FP_TFH_L2A"

L2B_PRODUCT_FD = "FP_FD__L2B"

AUX_PP_INPUT = "AUX_PP2_FH"

L2B_FH_INPUT_ID_LIST = [L2A_PRODUCT_FH, L2A_PRODUCT_TFH, AUX_PP_INPUT, L2B_PRODUCT_FD]

AUX_PP2B_FH_PRODUCT = "AUX_PP2_FH"

CONFIGURATION_FILES_L2BFHPCONF = "L2B_FH_P_Conf"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_L2BFHPCONF,
]

PROCESSING_PARAMS_TILE_ID = "tile_id"

PROCESSING_PARAMS_ID_LIST = [PROCESSING_PARAMS_TILE_ID]

ALIAS_EXPECTED_FD_INPUT: dict[str, bool] = {
    EXPECTED_PROCESSOR_ALIAS_NO_FD: False,
    EXPECTED_PROCESSOR_ALIAS_YES_FD: True,
}
"""Maps each processor alias to whether an optional L2B FD input is expected."""


class InvalidL2bFHJobOrder(ValueError):
    """Raised when failing to translate a joborder meant for the L2b FH Processor"""


def translate_l2b_fh_list_of_inputs(
    input_products_list: list[joborder_models.JoInputType],
) -> tuple[list[Path], Path, Path | None]:
    """Retrieve, from the input products section, paths of L1c stack acquisitions,
    aux_pp2_fh file and optionally the FH L2a product.

    Parameters
    ----------
    input_products_list : List[joborder_models.JoInputType]
        list of input products tags

    Returns
    -------
    Tuple[Tuple[Path,...], Path]:
        four outputs, one tuple for the input stack products, one for direct MPH files paths
        (for fast reading in L2a pre-processing),
        one single path for the AUX PP2 FH Configuration and for the FH L2b product.

    Raises
    ------
    InvalidL2bFHJobOrder
        in case of unexpected input products identifiers, missing required input products
    """

    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products:
        if file_id not in L2B_FH_INPUT_ID_LIST:
            raise InvalidL2bFHJobOrder(f"Unexpected input identifier: {file_id}")

    if AUX_PP_INPUT not in input_products:
        raise InvalidL2bFHJobOrder(f"Missing required input: {AUX_PP_INPUT}")
    if L2A_PRODUCT_FH not in input_products and L2A_PRODUCT_TFH not in input_products:
        raise InvalidL2bFHJobOrder(f"Missing required input: one of {L2A_PRODUCT_FH} or {L2A_PRODUCT_TFH}")

    input_l2a_products = [
        Path(p)
        for p in (
            input_products.pop(L2A_PRODUCT_FH)
            if L2A_PRODUCT_FH in input_products
            else input_products.pop(L2A_PRODUCT_TFH)
        )
    ]

    aux_pp2_fh_path = Path(input_products.pop(AUX_PP_INPUT)[0])

    # input_l2b_fd_product is optional
    input_l2b_fd_product = None
    if len(input_products) > 0 and L2B_PRODUCT_FD in input_products:
        input_l2b_fd_product = Path(input_products.pop(L2B_PRODUCT_FD)[0])

    assert len(input_products) == 0

    return input_l2a_products, aux_pp2_fh_path, input_l2b_fd_product


def retrieve_configuration_files(
    configuration_files_list: list[joborder_models.CfgFileType],
) -> Path | None:
    """Retrieve configuration files from the section

    Parameters
    ----------
    configuration_files_list : List[joborder_models.CfgFileType]
        list of configuration files tag

    Returns
    -------
    Path
        dem directory, l2b_fh configuration file

    Raises
    ------
    InvalidL2bFHJobOrder
        unexpected configuration files id
    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id not in CONFIGURATION_FILES_ID_LIST:
            raise InvalidL2bFHJobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    l2b_p_conf_raw = configuration_files.pop(CONFIGURATION_FILES_L2BFHPCONF, None)
    l2b_p_conf = Path(l2b_p_conf_raw) if l2b_p_conf_raw is not None else None

    assert len(configuration_files) == 0

    return l2b_p_conf


def retrieve_l2b_fh_processing_parameters(
    metadata_parameters: list[joborder_models.ParameterType],
) -> L2BFHProcessingParameters:
    """Retrieve Proc parameters from the section

    Parameters
    ----------
    proc_parameters_list : List[joborder_models.ParameterType]
        list of processing parameters

    Returns
    -------
    L2BFHProcessingParameters
        the struct containing the processing parameters

    Raises
    ------
    InvalidL2bFHJobOrder
        unexpected configuration files id
    """
    parameters = flatten_processing_params(metadata_parameters)

    for param_id in parameters:
        if param_id not in PROCESSING_PARAMS_ID_LIST:
            raise InvalidL2bFHJobOrder(f"Unexpected input processing parameter identifier: {param_id}")

    l2b_fh_processing_parameters = L2BFHProcessingParameters()

    tile_id = parameters.pop(PROCESSING_PARAMS_TILE_ID, None)
    if tile_id:
        l2b_fh_processing_parameters.tile_id = tile_id

    assert len(parameters) == 0

    return l2b_fh_processing_parameters


def retrieve_l2b_fh_output_directory(
    output_products_list: list[joborder_models.JoOutputType],
) -> tuple[Path, str, int]:
    """Retrieve output products directory from the output products section

    Parameters
    ----------
    output_products_list : List[joborder_models.JoOutputType]
        output products tags

    Returns
    -------
    Tuple[Path, List[str]]
        Output products common directory and list of enabled output products.

    Raises
    ------
    InvalidL2bFHJobOrder
        in case of unexpected output products identifiers, missing required output products or mismatches in the swath
    """

    output_products, output_directory, output_baseline = flatten_output_products(output_products_list)

    if len(output_products) == 0:
        raise InvalidL2bFHJobOrder("No output products specified: exactly one is required.")
    if len(output_products) > 1:
        raise InvalidL2bFHJobOrder("Too many output products specified: just one is requested.")

    output_product = output_products[0]
    if output_product not in [L2B_OUTPUT_PRODUCT_FH, L2B_OUTPUT_PRODUCT_TFH]:
        raise InvalidL2bFHJobOrder(f"Unexpected output product identifier: {output_product}")

    return Path(output_directory), output_product, int(output_baseline)


def translate_model_to_l2b_fh_job_order(
    job_order: joborder_models.JobOrder,
) -> L2bFHJobOrder:
    """Translate the job order model into a L2b fh processor job order object.

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        the object representing the input xml as provided by the XML parser.

    Returns
    -------
    L2bFHJobOrder
        Object containing the job order for the L2b fh processor task.

    Raises
    ------
    InvalidL2bFHJobOrder
        If the job_order_content is not compatible with a L2b FH Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidL2bFHJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    processor_configuration = retrieve_configuration_params(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAMES_LIST,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    input_ids = [p.input_id.value for p in task.list_of_inputs.input]
    found_optional_l2b_fd = L2B_OUTPUT_PRODUCT_FD in input_ids

    proc_name = job_order.processor_configuration.processor_name.value

    if proc_name in ALIAS_EXPECTED_FD_INPUT:
        expected_fd = ALIAS_EXPECTED_FD_INPUT[proc_name]
        if found_optional_l2b_fd != expected_fd:
            fd_status = "PRESENT" if found_optional_l2b_fd else "ABSENT"
            raise InvalidJobOrder(f"Invalid processor name: {proc_name} when optional L2B FD is '{fd_status}'")

    device_resources = retrieve_device_resources(task)

    l2b_fh_processing_parameters = retrieve_l2b_fh_processing_parameters(task.list_of_proc_parameters.proc_parameter)

    l2b_fh_p_conf = retrieve_configuration_files(task.list_of_cfg_files.cfg_file)

    (
        input_l2a_products,
        aux_pp2_fh_path,
        input_l2b_fd_product,
    ) = translate_l2b_fh_list_of_inputs(task.list_of_inputs.input)

    (
        output_directory,
        output_product,
        output_baseline,
    ) = retrieve_l2b_fh_output_directory(task.list_of_outputs.output)

    return L2bFHJobOrder(
        input_l2a_products,
        output_directory,
        output_product,
        aux_pp2_fh_path,
        device_resources,
        processor_configuration,
        l2b_fh_processing_parameters,
        input_l2b_fd_product=input_l2b_fd_product,
        l2b_fh_p_conf=l2b_fh_p_conf,
        output_baseline=output_baseline,
    )
