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
from bps.common import Swath
from bps.common.io import joborder_models
from bps.common.l2_joborder_tags import (
    L2A_OUTPUT_PRODUCT_FD,
    L2A_OUTPUT_PRODUCT_FH,
    L2A_OUTPUT_PRODUCT_GN,
    L2A_OUTPUT_PRODUCT_TFH,
)
from bps.common.translate_job_order import (
    flatten_configuration_file,
    flatten_input_products_allow_multiple_products,
    flatten_output_products,
    retrieve_configuration_params_l2,
    retrieve_device_resources,
    retrieve_swath_from_products_identifiers,
    retrieve_task,
)
from bps.l2a_processor.core.joborder_l2a import L2aJobOrder

EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass L2a processor"""

EXPECTED_PROCESSOR_NAME = "L2A_P"
"""Processor name for Biomass L2a processor, valid for FD, FH, GN and TOMO FH"""

EXPECTED_PROCESSOR_ALIAS_FD = "L2A_P_FD"
"""Alternative alias for Biomass L2a FD processor only"""

EXPECTED_PROCESSOR_ALIAS_FH = "L2A_P_FH"
"""Alternative alias for Biomass L2a FH processor only"""

EXPECTED_PROCESSOR_ALIAS_GN = "L2A_P_GN"
"""Alternative alias for Biomass L2a GN processor only"""

EXPECTED_PROCESSOR_ALIAS_TOMO_FH = "L2A_P_TFH"
"""Alternative alias for Biomass L2a TOMO FH processor only"""

EXPECTED_PROCESSOR_NAMES_LIST = [
    EXPECTED_PROCESSOR_NAME,
    EXPECTED_PROCESSOR_ALIAS_FD,
    EXPECTED_PROCESSOR_ALIAS_FH,
    EXPECTED_PROCESSOR_ALIAS_GN,
    EXPECTED_PROCESSOR_ALIAS_TOMO_FH,
]
"""All possible alias for Biomass L2a processors"""

EXPECTED_PROCESSOR_VERSION = "04.22"
"""Processor version for Biomass L2a processor"""

EXPECTED_TASK_NAME = "L2A_P"
"""Task name for Biomass L2a processor"""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass L2a processor"""

L2A_OUTPUT_PRODUCTS_ID_LIST = [
    L2A_OUTPUT_PRODUCT_FD,
    L2A_OUTPUT_PRODUCT_FH,
    L2A_OUTPUT_PRODUCT_GN,
    L2A_OUTPUT_PRODUCT_TFH,
]


STA_PRODUCT_S1 = "S1_STA__1S"
STA_PRODUCT_S2 = "S2_STA__1S"
STA_PRODUCT_S3 = "S3_STA__1S"
FD_PRODUCT_INPUT = L2A_OUTPUT_PRODUCT_FD
AUX_PP_INPUT = "AUX_PP2_2A"

STA_PRODUCT_MAP = {
    Swath.S1: STA_PRODUCT_S1,
    Swath.S2: STA_PRODUCT_S2,
    Swath.S3: STA_PRODUCT_S3,
}

L2a_INPUT_ID_LIST = list(STA_PRODUCT_MAP.values()) + [FD_PRODUCT_INPUT] + [AUX_PP_INPUT]

AUX_PP2A_PRODUCT = "AUX_PP2_2A"

CONFIGURATION_FILES_L2APCONF = "L2A_P_Conf"
CONFIGURATION_FILES_FNF_DIR = "FNF"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_L2APCONF,
    CONFIGURATION_FILES_FNF_DIR,
]


class InvalidL2aJobOrder(ValueError):
    """Raised when failing to translate a joborder meant for the L2a Processor"""


def translate_l2a_list_of_inputs(
    input_products_list: list[joborder_models.JoInputType],
    processing_swath: Swath,
) -> tuple[tuple[Path], tuple[Path], Path, Path | None]:
    """Retrieve, from the input products section, paths of L1c stack acquisitions,
    aux_pp2_2a file and optionally the FD L2a product.

    Parameters
    ----------
    input_products_list : List[joborder_models.JoInputType]
        list of input products tags
    processing_swath : Swath
        which swath to process

    Returns
    -------
    Tuple[Tuple[Path], Tuple[Path], Path, Optional[Path]]:
        four outputs, one tuple for the input stack products, one for direct MPH files paths
        (for fast reading in L2a pre-processing),
        one single path for the AUX PP2 2A Configuration and for the FD L2a product.

    Raises
    ------
    InvalidL2aJobOrder
        in case of unexpected input products identifiers, missing required input products
    """

    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products:
        if file_id not in L2a_INPUT_ID_LIST:
            raise InvalidL2aJobOrder(f"Unexpected input identifier: {file_id}")

    input_stack = input_products.pop(STA_PRODUCT_MAP[processing_swath])
    input_stack = tuple(Path(input_lic_path) for input_lic_path in input_stack)

    if len(input_stack) < 2 or len(input_stack) > 8:
        raise InvalidL2aJobOrder("Wrong number of input Sx_STA__1S file names: should be >2 and <8")

    input_stack_mph_files = []
    for input_acquisition in input_stack:
        name = Path(str(input_acquisition.name).lower() + ".xml")
        input_stack_mph_files.append(input_acquisition.joinpath(name))
    input_stack_mph_files = tuple(input_stack_mph_files)

    aux_pp2_2a_path = Path(input_products.pop(AUX_PP_INPUT)[0])

    if len(input_products) > 0:
        # input_l2a_fd_product is optional
        input_l2a_fd_product = Path(input_products.pop(FD_PRODUCT_INPUT)[0])
    else:
        input_l2a_fd_product = None

    if len(input_products) > 0:
        raise InvalidL2aJobOrder(f"Unexpected input products: {input_products}")

    return input_stack, input_stack_mph_files, aux_pp2_2a_path, input_l2a_fd_product


def retrieve_configuration_files(
    configuration_files_list: list[joborder_models.CfgFileType],
) -> tuple[Path, Path | None]:
    """Retrieve configuration files from the section

    Parameters
    ----------
    configuration_files_list : List[joborder_models.CfgFileType]
        list of configuration files tag

    Returns
    -------
    Tuple[Path, Optional[Path]]
        dem directory, l1p configuration file

    Raises
    ------
    InvalidL2aJobOrder
        unexpected configuration files id
    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id not in CONFIGURATION_FILES_ID_LIST:
            raise InvalidL2aJobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    l2a_p_conf = configuration_files.pop(CONFIGURATION_FILES_L2APCONF, None)
    if l2a_p_conf is not None:
        l2a_p_conf = Path(l2a_p_conf)

    fnf_dir = Path(configuration_files.pop(CONFIGURATION_FILES_FNF_DIR))

    if len(configuration_files) > 0:
        raise InvalidL2aJobOrder(f"Unexpected configuration files: {configuration_files}")

    return fnf_dir, l2a_p_conf


def retrieve_l2a_output_directory(
    output_products_list: list[joborder_models.JoOutputType],
    processor_name: str,
) -> tuple[Path, list[str], int | list[int]]:
    """Retrieve output products directory from the output products section

    Parameters
    ----------
    output_products_list : List[joborder_models.JoOutputType]
        output products tags
    processor_name: str
        processor name specified in job order

    Returns
    -------
    Tuple[Path, List[str]]
        Output products common directory and list of enabled output products.

    Raises
    ------
    InvalidL2aJobOrder
        in case of unexpected output products identifiers, missing required output products or mismatches in processor_name and output products
    """

    output_products, output_directory, output_baselines = flatten_output_products(output_products_list)

    if processor_name == EXPECTED_PROCESSOR_NAME:
        valid_output_products = L2A_OUTPUT_PRODUCTS_ID_LIST
    elif processor_name == EXPECTED_PROCESSOR_ALIAS_FD:
        valid_output_products = L2A_OUTPUT_PRODUCT_FD
    elif processor_name == EXPECTED_PROCESSOR_ALIAS_FH:
        valid_output_products = L2A_OUTPUT_PRODUCT_FH
    elif processor_name == EXPECTED_PROCESSOR_ALIAS_GN:
        valid_output_products = L2A_OUTPUT_PRODUCT_GN
    elif processor_name == EXPECTED_PROCESSOR_ALIAS_TOMO_FH:
        valid_output_products = L2A_OUTPUT_PRODUCT_TFH
    if (
        processor_name
        in [
            EXPECTED_PROCESSOR_ALIAS_FD,
            EXPECTED_PROCESSOR_ALIAS_FH,
            EXPECTED_PROCESSOR_ALIAS_GN,
        ]
        and len(output_products) != 1
        or output_products[0] not in valid_output_products
    ):
        raise InvalidL2aJobOrder(
            f"when processor name is {processor_name}, the only admitted output is {valid_output_products}; found instead {output_products[0] if len(output_products) == 1 else output_products}"
        )

    output_baselines = list(np.ones(len(output_products)).astype(int) * int(output_baselines))

    for file_id in output_products:
        if file_id not in L2A_OUTPUT_PRODUCTS_ID_LIST:
            raise InvalidL2aJobOrder(f"Unexpected output product identifier: {file_id}")

    if not any([prod_string in output_products for prod_string in L2A_OUTPUT_PRODUCTS_ID_LIST]):
        raise InvalidL2aJobOrder(
            f"Missing required output product (at least one of three is necessary): {L2A_OUTPUT_PRODUCTS_ID_LIST}"
        )
    for prod in L2A_OUTPUT_PRODUCTS_ID_LIST:
        if prod in output_products:
            output_products.remove(prod)

    if len(output_products) > 0:
        raise InvalidL2aJobOrder(f"Unexpected output products: {output_products}")

    output_products, _, _ = flatten_output_products(output_products_list)

    return Path(output_directory), output_products, output_baselines


def translate_model_to_l2a_job_order(
    job_order: joborder_models.JobOrder,
) -> L2aJobOrder:
    """Translate the job order model into a L2a processor job order object.

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        the object representing the input xml as provided by the XML parser.

    Returns
    -------
    L2aJobOrder
        Object containing the job order for the L2a processor task.

    Raises
    ------
    InvalidL2aJobOrder
        If the job_order_content is not compatible with a L2a Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidL2aJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    assert job_order.processor_configuration is not None
    processor_configuration = retrieve_configuration_params_l2(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAMES_LIST,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    device_resources = retrieve_device_resources(task)

    assert task.list_of_cfg_files is not None
    fnf_dir, l2a_p_conf = retrieve_configuration_files(task.list_of_cfg_files.cfg_file)

    assert task.list_of_inputs is not None
    input_products = flatten_input_products_allow_multiple_products(task.list_of_inputs.input)

    processing_swath = retrieve_swath_from_products_identifiers(
        list(input_products.keys()),
        STA_PRODUCT_MAP,
    )
    if processing_swath is None:
        raise InvalidL2aJobOrder("Missing L1c product")

    (
        input_stack_acquisitions,
        input_stack_mph_files,
        aux_pp2_2a_path,
        input_l2a_fd_product,
    ) = translate_l2a_list_of_inputs(task.list_of_inputs.input, processing_swath)

    assert task.list_of_outputs is not None
    assert job_order.processor_configuration.processor_name is not None
    output_directory, output_products, output_baselines = retrieve_l2a_output_directory(
        task.list_of_outputs.output,
        job_order.processor_configuration.processor_name.value,
    )

    return L2aJobOrder(
        input_stack_acquisitions,
        input_stack_mph_files,
        output_directory,
        output_products,
        aux_pp2_2a_path,
        fnf_dir,
        device_resources,
        processor_configuration,
        input_l2a_fd_product=input_l2a_fd_product if input_l2a_fd_product else None,
        l2a_p_conf=l2a_p_conf if l2a_p_conf else None,
        output_baselines=output_baselines,
    )
