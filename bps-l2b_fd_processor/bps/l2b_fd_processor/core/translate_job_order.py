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
from bps.common.translate_job_order import (
    InvalidJobOrder,
    flatten_input_products_allow_multiple_products,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_optional_configuration_file,
    retrieve_single_output_product,
    retrieve_task,
    retrieve_tile_processing_parameters,
)
from bps.l2b_fd_processor.core.joborder_l2b_fd import (
    L2bFDJobOrder,
)

EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass L2b FD processor"""

EXPECTED_PROCESSOR_NAME = "L2B_FD_P"
"""Processor name for Biomass L2b FD processor"""

EXPECTED_PROCESSOR_VERSION = "04.44"
"""Processor version for Biomass L2b FD processor"""

EXPECTED_TASK_NAME = EXPECTED_PROCESSOR_NAME
"""Task name for Biomass L2b FD processor"""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass L2b FD processor"""

L2B_OUTPUT_PRODUCT_FD = "FP_FD__L2B"

L2A_PRODUCT_FD = "FP_FD__L2A"

AUX_PP_INPUT = "AUX_PP2_FD"

L2B_FD_INPUT_ID_LIST = [L2A_PRODUCT_FD, AUX_PP_INPUT]

AUX_PP2B_FD_PRODUCT = AUX_PP_INPUT

CONFIGURATION_FILES_L2BFDPCONF = "L2B_FD_P_Conf"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_L2BFDPCONF,
]

PROCESSING_PARAMS_TILE_ID = "tile_id"

PROCESSING_PARAMS_ID_LIST = [PROCESSING_PARAMS_TILE_ID]


def translate_l2b_fd_list_of_inputs(
    input_products_list: list[joborder_models.JoInputType],
) -> tuple[list[Path], Path]:
    """Retrieve, from the input products section, paths of L1c stack acquisitions,
    aux_pp2_fd file and optionally the FD L2a product.

    Parameters
    ----------
    input_products_list : List[joborder_models.JoInputType]
        list of input products tags

    Returns
    -------
    Tuple[Tuple[Path], Path]:
        four outputs, one tuple for the input stack products, one for direct MPH files paths
        (for fast reading in L2a pre-processing),
        one single path for the AUX PP2 FD Configuration and for the FD L2b product.

    Raises
    ------
    InvalidJobOrder
        in case of unexpected input products identifiers, missing required input products
    """

    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products:
        if file_id not in L2B_FD_INPUT_ID_LIST:
            raise InvalidJobOrder(f"Unexpected input identifier: {file_id}")

    if L2A_PRODUCT_FD not in input_products:
        raise InvalidJobOrder(f"Missing required input: {L2A_PRODUCT_FD}")
    if AUX_PP_INPUT not in input_products:
        raise InvalidJobOrder(f"Missing required input: {AUX_PP_INPUT}")

    input_l2a_products = [Path(input_l2a_path) for input_l2a_path in input_products.pop(L2A_PRODUCT_FD)]

    aux_pp2_fd_path = Path(input_products.pop(AUX_PP_INPUT)[0])

    assert len(input_products) == 0

    return input_l2a_products, aux_pp2_fd_path


def translate_model_to_l2b_fd_job_order(
    job_order: joborder_models.JobOrder,
) -> L2bFDJobOrder:
    """Translate the job order model into a L2b fd processor job order object.

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        the object representing the input xml as provided by the XML parser.

    Returns
    -------
    L2bFDJobOrder
        Object containing the job order for the L2b fd processor task.

    Raises
    ------
    InvalidJobOrder
        If the job_order_content is not compatible with a L2b FD Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    processor_configuration = retrieve_configuration_params(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAME,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    device_resources = retrieve_device_resources(task)

    l2b_fh_processing_parameters = retrieve_tile_processing_parameters(
        task.list_of_proc_parameters.proc_parameter, PROCESSING_PARAMS_TILE_ID
    )

    l2b_fd_p_conf = retrieve_optional_configuration_file(
        task.list_of_cfg_files.cfg_file, CONFIGURATION_FILES_L2BFDPCONF
    )

    (
        input_l2a_products,
        aux_pp2_fd_path,
    ) = translate_l2b_fd_list_of_inputs(task.list_of_inputs.input)

    (
        output_directory,
        output_product,
        output_baseline,
    ) = retrieve_single_output_product(task.list_of_outputs.output, [L2B_OUTPUT_PRODUCT_FD])

    return L2bFDJobOrder(
        input_l2a_products,
        output_directory,
        output_product,
        aux_pp2_fd_path,
        device_resources,
        processor_configuration,
        l2b_fh_processing_parameters,
        l2b_fd_p_conf=l2b_fd_p_conf,
        output_baseline=output_baseline,
    )
