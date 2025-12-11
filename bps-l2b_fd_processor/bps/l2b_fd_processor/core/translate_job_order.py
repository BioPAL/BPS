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

import numpy as np
from bps.common.io import joborder_models
from bps.common.translate_job_order import (
    flatten_configuration_file,
    flatten_input_products_allow_multiple_products,
    flatten_output_products,
    flatten_processing_params,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_task,
)
from bps.l2b_fd_processor.core.joborder_l2b_fd import (
    L2bFDJobOrder,
    L2BFDProcessingParameters,
)

EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass L2b FD processor"""

EXPECTED_PROCESSOR_NAME = "L2B_FD_P"
"""Processor name for Biomass L2b FD processor"""

EXPECTED_PROCESSOR_VERSION = "04.22"
"""Processor version for Biomass L2b FD processor"""

EXPECTED_TASK_NAME = EXPECTED_PROCESSOR_NAME
"""Task name for Biomass L2b FD processor"""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass L2b FD processor"""

L2B_OUTPUT_PRODUCT_FD = "FP_FD__L2B"

L2A_PRODUCT_FD = "FP_FD__L2A"

AUX_PP_INPUT = "AUX_PP2_FD"

L2B_FD_INPUT_ID_LIST = [L2A_PRODUCT_FD, AUX_PP_INPUT]

AUX_PP2B_FD_PRODUCT = "AUX_PP2_FD"

CONFIGURATION_FILES_L2BFDPCONF = "L2B_FD_P_Conf"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_L2BFDPCONF,
]

PROCESSING_PARAMS_TILE_ID = "tile_id"

PROCESSING_PARAMS_ID_LIST = [PROCESSING_PARAMS_TILE_ID]


class InvalidL2bFDJobOrder(ValueError):
    """Raised when failing to translate a joborder meant for the L2b FD Processor"""


def translate_l2b_fd_list_of_inputs(
    input_products_list: list[joborder_models.JoInputType],
) -> tuple[tuple[Path], Path]:
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
    InvalidL2bFDJobOrder
        in case of unexpected input products identifiers, missing required input products
    """

    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products:
        if file_id not in L2B_FD_INPUT_ID_LIST:
            raise InvalidL2bFDJobOrder(f"Unexpected input identifier: {file_id}")

    input_l2a_products = input_products.pop(L2A_PRODUCT_FD)
    input_l2a_products = tuple(Path(input_l2a_path) for input_l2a_path in input_l2a_products)

    aux_pp2_fd_path = Path(input_products.pop(AUX_PP_INPUT)[0])

    if len(input_products) > 0:
        raise InvalidL2bFDJobOrder(f"Unexpected input products: {input_products}")

    return input_l2a_products, aux_pp2_fd_path


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
        dem directory, l2b_fd configuration file

    Raises
    ------
    InvalidL2bFDJobOrder
        unexpected configuration files id
    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id not in CONFIGURATION_FILES_ID_LIST:
            raise InvalidL2bFDJobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    l2b_p_conf = configuration_files.pop(CONFIGURATION_FILES_L2BFDPCONF, None)
    if l2b_p_conf is not None:
        l2b_p_conf = Path(l2b_p_conf)

    if len(configuration_files) > 0:
        raise InvalidL2bFDJobOrder(f"Unexpected configuration files: {configuration_files}")

    return l2b_p_conf


def retrieve_l2b_fd_processing_parameters(
    metadata_parameters: list[joborder_models.ParameterType],
) -> L2BFDProcessingParameters:
    """Retrieve Proc parameters from the section

    Parameters
    ----------
    proc_parameters_list : List[joborder_models.ParameterType]
        list of processing parameters

    Returns
    -------
    L2BFDProcessingParameters
        the struct containing the processing parameters

    Raises
    ------
    InvalidL2bFDJobOrder
        unexpected configuration files id
    """
    parameters = flatten_processing_params(metadata_parameters)

    for param_id in parameters:
        if param_id not in PROCESSING_PARAMS_ID_LIST:
            raise InvalidL2bFDJobOrder(f"Unexpected input processing parameter identifier: {param_id}")

    l2b_fd_processing_parameters = L2BFDProcessingParameters()

    tile_id = parameters.pop(PROCESSING_PARAMS_TILE_ID, None)
    if tile_id:
        l2b_fd_processing_parameters.tile_id = tile_id

    if len(parameters) > 0:
        raise InvalidL2bFDJobOrder(f"Unexpected processing parameters: {parameters}")

    return l2b_fd_processing_parameters


def retrieve_l2b_fd_output_directory(
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
    InvalidL2bFDJobOrder
        in case of unexpected output products identifiers, missing required output products or mismatches in the swath
    """

    output_products, output_directory, output_baselines = flatten_output_products(output_products_list)
    output_baselines = list(np.ones(len(output_products)).astype(int) * int(output_baselines))

    if len(output_products) > 1:
        raise InvalidL2bFDJobOrder("Too many output products specified: just one is requested.")

    output_baseline = output_baselines[0]

    file_id = output_products[0]
    if file_id not in L2B_OUTPUT_PRODUCT_FD:
        raise InvalidL2bFDJobOrder(f"Unexpected output product identifier: {file_id}")

    if L2B_OUTPUT_PRODUCT_FD not in output_products[0]:
        raise InvalidL2bFDJobOrder("Required output product is not an L2B FD")

    output_products, _, _ = flatten_output_products(output_products_list)
    output_product = output_products[0]

    return Path(output_directory), output_product, output_baseline


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
    InvalidL2bFDJobOrder
        If the job_order_content is not compatible with a L2b FD Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidL2bFDJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    assert job_order.processor_configuration is not None
    processor_configuration = retrieve_configuration_params(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAME,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    device_resources = retrieve_device_resources(task)

    assert task.list_of_proc_parameters is not None
    l2b_fh_processing_parameters = retrieve_l2b_fd_processing_parameters(task.list_of_proc_parameters.proc_parameter)

    assert task.list_of_cfg_files is not None
    l2b_fd_p_conf = retrieve_configuration_files(task.list_of_cfg_files.cfg_file)

    assert task.list_of_inputs is not None
    (
        input_l2a_products,
        aux_pp2_fd_path,
    ) = translate_l2b_fd_list_of_inputs(task.list_of_inputs.input)

    assert task.list_of_outputs is not None
    (
        output_directory,
        output_product,
        output_baseline,
    ) = retrieve_l2b_fd_output_directory(task.list_of_outputs.output)

    return L2bFDJobOrder(
        input_l2a_products,
        output_directory,
        output_product,
        aux_pp2_fd_path,
        device_resources,
        processor_configuration,
        l2b_fh_processing_parameters,
        l2b_fd_p_conf=l2b_fd_p_conf if l2b_fd_p_conf else None,
        output_baseline=output_baseline,
    )
