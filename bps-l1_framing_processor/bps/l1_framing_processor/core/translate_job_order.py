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

import enum
from pathlib import Path

from bps.common import Swath
from bps.common.io import joborder_models
from bps.common.translate_job_order import (
    flatten_input_products,
    flatten_intermediate_outputs,
    flatten_output_products,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_swath_from_products_identifiers,
    retrieve_task,
)
from bps.l1_framing_processor.core.joborder_l1f import (
    L1AuxiliaryProducts,
    L1FJobOrder,
    L1StripmapInputProducts,
    L1StripmapProducts,
    L1VirtualFrameOutputProducts,
)

EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass L1 framing processor"""

EXPECTED_PROCESSOR_NAME = "L1F_P"
"""Processor name for Biomass L1 framing processor"""

EXPECTED_PROCESSOR_VERSION = "04.30"
"""Processor version for Biomass L1 framing processor"""

EXPECTED_TASK_NAME = EXPECTED_PROCESSOR_NAME
"""Task name for Biomass L1 framing processor"""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass L1 framing processor"""


class ProductsLevel(enum.Enum):
    """Different product levels"""

    L0_STANDARD_RAW = enum.auto()
    L0_MONITORING_RAW = enum.auto()
    L1_VIRTUAL_FRAME = enum.auto()


L0_STANDARD_PRODUCT_RAW_S1 = "S1_RAW__0S"
L0_STANDARD_PRODUCT_RAW_S2 = "S2_RAW__0S"
L0_STANDARD_PRODUCT_RAW_S3 = "S3_RAW__0S"
L0_STANDARD_PRODUCT_RAW_MAP = {
    Swath.S1: L0_STANDARD_PRODUCT_RAW_S1,
    Swath.S2: L0_STANDARD_PRODUCT_RAW_S2,
    Swath.S3: L0_STANDARD_PRODUCT_RAW_S3,
}

L0_MONITORING_PRODUCT_RAW_S1 = "S1_RAW__0M"
L0_MONITORING_PRODUCT_RAW_S2 = "S2_RAW__0M"
L0_MONITORING_PRODUCT_RAW_S3 = "S3_RAW__0M"
L0_MONITORING_PRODUCT_RAW_MAP = {
    Swath.S1: L0_MONITORING_PRODUCT_RAW_S1,
    Swath.S2: L0_MONITORING_PRODUCT_RAW_S2,
    Swath.S3: L0_MONITORING_PRODUCT_RAW_S3,
}

L0_INPUT_PRODUCTS_ID_LIST = list(L0_STANDARD_PRODUCT_RAW_MAP.values()) + list(L0_MONITORING_PRODUCT_RAW_MAP.values())

L1_VIRTUAL_FRAME = "CPF_L1VFRA"
L1_VIRTUAL_FRAME_MAP = {
    Swath.S1: L1_VIRTUAL_FRAME,
    Swath.S2: L1_VIRTUAL_FRAME,
    Swath.S3: L1_VIRTUAL_FRAME,
}
L1_OUTPUT_PRODUCTS_ID_LIST = list(L1_VIRTUAL_FRAME_MAP.values())

PRODUCT_LEVEL_TO_ID_MAP = {
    ProductsLevel.L0_STANDARD_RAW: L0_STANDARD_PRODUCT_RAW_MAP,
    ProductsLevel.L0_MONITORING_RAW: L0_MONITORING_PRODUCT_RAW_MAP,
    ProductsLevel.L1_VIRTUAL_FRAME: L1_VIRTUAL_FRAME_MAP,
}

AUX_ORB_PRODUCT = "AUX_ORB___"

AUX_PRODUCTS_ID_LIST = [
    AUX_ORB_PRODUCT,
]

CONFIGURATION_FILES_L1FPCONF = "L1F_P_Conf"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_L1FPCONF,
]


class InvalidL1FJobOrder(ValueError):
    """Raised when failing to translate a joborder meant for the L1 Framing Processor"""


def fill_l1_auxiliary_products(products: dict[str, str]) -> L1AuxiliaryProducts:
    """Fill the structure with the input products found in the input dict

    when found, items are removed from the products list

    Parameters
    ----------
    products : Dict[str, str]
        input/output, list of products; updated by the function

    Returns
    -------
    L1AuxiliaryProducts
        structure
    """
    aux_orb_product = products.pop(AUX_ORB_PRODUCT)

    return L1AuxiliaryProducts(Path(aux_orb_product))


def retrieve_l1_input_and_aux_products(
    input_products_list: list[joborder_models.JoInputType],
    processing_swath: Swath,
) -> tuple[L1StripmapInputProducts, L1AuxiliaryProducts]:
    """Retrieve input and auxiliary products from the input products section

    Parameters
    ----------
    input_products_list : List[joborder_models.JoInputType]
        list of input products tags
    processing_swath : Swath
        which swath to process

    Returns
    -------
    Tuple[L1StripmapInputProducts, L1AuxiliaryProducts]
        two structures, one for the input products and one for the auxiliary products

    Raises
    ------
    InvalidL1FJobOrder
        in case of unexpected input products identifiers, missing required input products or mismatches in the swath
    """
    input_products = flatten_input_products(input_products_list)

    for file_id in input_products:
        if file_id not in L0_INPUT_PRODUCTS_ID_LIST + AUX_PRODUCTS_ID_LIST:
            raise InvalidL1FJobOrder(f"Unexpected input product identifier: {file_id}")

    input_standard_product = input_products.pop(L0_STANDARD_PRODUCT_RAW_MAP[processing_swath])

    monitoring_product_swath = retrieve_swath_from_products_identifiers(
        list(input_products.keys()),
        PRODUCT_LEVEL_TO_ID_MAP[ProductsLevel.L0_MONITORING_RAW],
    )

    if monitoring_product_swath is None:
        raise InvalidL1FJobOrder("Missing L0 monitoring product")

    if monitoring_product_swath != processing_swath:
        raise InvalidL1FJobOrder(
            f"Invalid input L0 monitoring product swath: {monitoring_product_swath} != {processing_swath}"
        )

    input_monitoring_product = input_products.pop(L0_MONITORING_PRODUCT_RAW_MAP[monitoring_product_swath])

    l1_input_products = L1StripmapInputProducts(
        input_standard=Path(input_standard_product),
        input_monitoring=Path(input_monitoring_product),
    )

    aux_products = fill_l1_auxiliary_products(input_products)

    if len(input_products) > 0:
        raise InvalidL1FJobOrder(f"Unexpected input products: {input_products}")

    return l1_input_products, aux_products


def retrieve_l1_output_products(
    output_products_list: list[joborder_models.JoOutputType],
    processing_swath: Swath,
) -> L1VirtualFrameOutputProducts:
    """Retrieve output products from the output products section

    Parameters
    ----------
    output_products_list : List[joborder_models.JoOutputType]
        output products tags
    processing_swath : Swath
        which swath to process

    Returns
    -------
    L1VirtualFrameOutputProducts
        Structure containing the output products

    Raises
    ------
    InvalidL1FJobOrder
        in case of unexpected output products identifiers, missing required output products or mismatches in the swath
    """
    output_products, output_directory, output_baseline = flatten_output_products(output_products_list)

    for file_id in output_products:
        if file_id not in L1_OUTPUT_PRODUCTS_ID_LIST:
            raise InvalidL1FJobOrder(f"Unexpected output product identifier: {file_id}")

    if L1_VIRTUAL_FRAME_MAP[processing_swath] not in output_products:
        raise InvalidL1FJobOrder(f"Missing required output product: {L1_VIRTUAL_FRAME_MAP[processing_swath]}")
    output_products.remove(L1_VIRTUAL_FRAME_MAP[processing_swath])

    l1_output_products = L1VirtualFrameOutputProducts(
        output_directory=Path(output_directory),
        output_baseline=output_baseline,
        vfra_standard_required=True,
    )

    if len(output_products) > 0:
        raise InvalidL1FJobOrder(f"Unexpected output products: {output_products}")

    return l1_output_products


def translate_model_to_l1f_job_order(
    job_order: joborder_models.JobOrder,
) -> L1FJobOrder:
    """Translate the job order model into a L1 framing processor job order object.

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        the object representing the input xml as provided by the XML parser.

    Returns
    -------
    L1FJobOrder
        Object containing the job order for the L1 framing processor task.

    Raises
    ------
    InvalidL1FJobOrder
        If the job_order_content is not compatible with a L1 Framing Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidL1FJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    assert job_order.processor_configuration is not None
    processor_configuration = retrieve_configuration_params(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAME,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    device_resources = retrieve_device_resources(task)

    assert task.list_of_inputs is not None
    input_products = flatten_input_products(task.list_of_inputs.input)

    processing_swath = retrieve_swath_from_products_identifiers(
        list(input_products.keys()),
        PRODUCT_LEVEL_TO_ID_MAP[ProductsLevel.L0_STANDARD_RAW],
    )
    if processing_swath is None:
        raise InvalidL1FJobOrder("Missing L0 product")

    l1_input_products, auxiliary_products = retrieve_l1_input_and_aux_products(
        task.list_of_inputs.input, processing_swath
    )

    assert task.list_of_outputs is not None
    l1_output_products = retrieve_l1_output_products(task.list_of_outputs.output, processing_swath)

    assert task.list_of_intermediate_outputs is not None
    intermediate_outputs = flatten_intermediate_outputs(task.list_of_intermediate_outputs.intermediate_output)

    assert isinstance(l1_input_products, L1StripmapInputProducts)
    assert isinstance(l1_output_products, L1VirtualFrameOutputProducts)
    io_products = L1StripmapProducts(input=l1_input_products, output=l1_output_products)

    return L1FJobOrder(
        io_products=io_products,
        auxiliary_files=auxiliary_products,
        device_resources=device_resources,
        processor_configuration=processor_configuration,
        intermediate_files=intermediate_outputs,
    )
