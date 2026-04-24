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

from bps.common import Swath
from bps.common.io import joborder_models
from bps.common.io.mph import get_mph_path
from bps.common.l2_joborder_tags import (
    L2A_OUTPUT_PRODUCT_FD,
    L2A_OUTPUT_PRODUCT_FH,
    L2A_OUTPUT_PRODUCT_GN,
    L2A_OUTPUT_PRODUCT_TFH,
)
from bps.common.translate_job_order import (
    InvalidJobOrder,
    flatten_configuration_file,
    flatten_input_products_allow_multiple_products,
    flatten_output_products,
    retrieve_configuration_params,
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

EXPECTED_PROCESSOR_VERSION = "04.44"
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

ALIAS_EXPECTED_OUTPUT: dict[str, str] = {
    EXPECTED_PROCESSOR_ALIAS_FD: L2A_OUTPUT_PRODUCT_FD,
    EXPECTED_PROCESSOR_ALIAS_FH: L2A_OUTPUT_PRODUCT_FH,
    EXPECTED_PROCESSOR_ALIAS_GN: L2A_OUTPUT_PRODUCT_GN,
    EXPECTED_PROCESSOR_ALIAS_TOMO_FH: L2A_OUTPUT_PRODUCT_TFH,
}
"""Maps each single-output processor alias to its expected output product."""


def translate_l2a_list_of_inputs(
    input_products_list: list[joborder_models.JoInputType],
    processing_swath: Swath,
) -> tuple[list[Path], list[Path], Path, Path | None]:
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
    InvalidJobOrder
        in case of unexpected input products identifiers, missing required input products
    """

    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products:
        if file_id not in L2a_INPUT_ID_LIST:
            raise InvalidJobOrder(f"Unexpected input identifier: {file_id}")

    input_stack = [Path(input_lic_path) for input_lic_path in input_products.pop(STA_PRODUCT_MAP[processing_swath])]

    if len(input_stack) < 2 or len(input_stack) > 8:
        raise InvalidJobOrder("Wrong number of input Sx_STA__1S file names: should be >2 and <8")

    input_stack_mph_files = [get_mph_path(acquisition) for acquisition in input_stack]

    aux_pp2_2a_path = Path(input_products.pop(AUX_PP_INPUT)[0])

    if len(input_products) > 0:
        # input_l2a_fd_product is optional
        input_l2a_fd_product = Path(input_products.pop(FD_PRODUCT_INPUT)[0])
    else:
        input_l2a_fd_product = None

    if len(input_products) > 0:
        raise InvalidJobOrder(f"Unexpected input products: {input_products}")

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
    InvalidJobOrder
        unexpected configuration files id
    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id not in CONFIGURATION_FILES_ID_LIST:
            raise InvalidJobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    l2a_p_conf = configuration_files.pop(CONFIGURATION_FILES_L2APCONF, None)
    if l2a_p_conf is not None:
        l2a_p_conf = Path(l2a_p_conf)

    fnf_dir = Path(configuration_files.pop(CONFIGURATION_FILES_FNF_DIR))

    if len(configuration_files) > 0:
        raise InvalidJobOrder(f"Unexpected configuration files: {configuration_files}")

    return fnf_dir, l2a_p_conf


def retrieve_l2a_output_directory(
    output_products_list: list[joborder_models.JoOutputType],
    processor_name: str,
) -> tuple[Path, list[str], list[int]]:
    """Retrieve output products directory from the output products section

    Parameters
    ----------
    output_products_list : List[joborder_models.JoOutputType]
        output products tags
    processor_name: str
        processor name specified in job order

    Returns
    -------
    Tuple[Path, List[str], List[int]]
        Output products common directory, list of enabled output products, and list of output baselines.

    Raises
    ------
    InvalidJobOrder
        in case of unexpected output products identifiers, missing required output products or mismatches in processor_name and output products
    """

    output_products, output_directory, output_baselines = flatten_output_products(output_products_list)

    if processor_name in ALIAS_EXPECTED_OUTPUT:
        expected_output = ALIAS_EXPECTED_OUTPUT[processor_name]
        if len(output_products) != 1 or output_products[0] != expected_output:
            raise InvalidJobOrder(
                f"when processor name is {processor_name}, the only admitted output is {expected_output}; "
                f"found instead {output_products[0] if len(output_products) == 1 else output_products}"
            )

    output_baselines = [int(output_baselines)] * len(output_products)

    for file_id in output_products:
        if file_id not in L2A_OUTPUT_PRODUCTS_ID_LIST:
            raise InvalidJobOrder(f"Unexpected output product identifier: {file_id}")

    if not any(prod_string in output_products for prod_string in L2A_OUTPUT_PRODUCTS_ID_LIST):
        raise InvalidJobOrder(
            f"Missing required output product (at least one of three is necessary): {L2A_OUTPUT_PRODUCTS_ID_LIST}"
        )

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
    InvalidJobOrder
        If the job_order_content is not compatible with a L2a Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    assert job_order.processor_configuration is not None
    processor_configuration = retrieve_configuration_params(
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
        raise InvalidJobOrder("Missing L1c product")

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
        input_l2a_fd_product=input_l2a_fd_product,
        l2a_p_conf=l2a_p_conf,
        output_baselines=output_baselines,
    )
