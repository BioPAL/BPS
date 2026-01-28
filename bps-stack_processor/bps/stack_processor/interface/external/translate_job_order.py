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

from bps.common import STRIPMAP_SWATHS, Swath
from bps.common.io import joborder_models
from bps.common.io.parsing import ParsingError
from bps.common.translate_job_order import (
    flatten_configuration_file,
    flatten_input_products_allow_multiple_products,
    flatten_intermediate_outputs,
    flatten_output_products,
    flatten_processing_params,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_swath_from_products_identifiers,
    retrieve_task,
)
from bps.stack_processor.interface.external.joborder_stack import (
    StackExternalsProducts,
    StackJobOrder,
    StackOutputProducts,
    StackProcessingParameters,
)

EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass Stack processor."""

EXPECTED_PROCESSOR_NAME = "STA_P"
"""Processor name for Biomass Stack processor."""

EXPECTED_PROCESSOR_VERSION = "04.30"
"""Processor version for Biomass Stack processor."""

EXPECTED_TASK_NAME = EXPECTED_PROCESSOR_NAME
"""Task name for Biomass Stack processor."""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass Stack processor."""


class StackJobOrderParsingError(ParsingError):
    """Handle errors while parsing the Stack Job Order."""


class ProductsLevel(enum.Enum):
    """Different product levels."""

    L1_STANDARD_SCS = enum.auto()
    STACK_STANDARD_STA = enum.auto()
    STACK_MONITORING_STA = enum.auto()


L1_STANDARD_PRODUCT_SCS_S1 = "S1_SCS__1S"
L1_STANDARD_PRODUCT_SCS_S2 = "S2_SCS__1S"
L1_STANDARD_PRODUCT_SCS_S3 = "S3_SCS__1S"
L1_STANDARD_PRODUCT_SCS_MAP = {
    Swath.S1: L1_STANDARD_PRODUCT_SCS_S1,
    Swath.S2: L1_STANDARD_PRODUCT_SCS_S2,
    Swath.S3: L1_STANDARD_PRODUCT_SCS_S3,
}

STACK_STANDARD_PRODUCT_STA_S1 = "S1_STA__1S"
STACK_STANDARD_PRODUCT_STA_S2 = "S2_STA__1S"
STACK_STANDARD_PRODUCT_STA_S3 = "S3_STA__1S"
STACK_STANDARD_PRODUCT_STA_MAP = {
    Swath.S1: STACK_STANDARD_PRODUCT_STA_S1,
    Swath.S2: STACK_STANDARD_PRODUCT_STA_S2,
    Swath.S3: STACK_STANDARD_PRODUCT_STA_S3,
}

STACK_MONITORING_PRODUCT_STA_S1 = "S1_STA__1M"
STACK_MONITORING_PRODUCT_STA_S2 = "S2_STA__1M"
STACK_MONITORING_PRODUCT_STA_S3 = "S3_STA__1M"
STACK_MONITORING_PRODUCT_STA_MAP = {
    Swath.S1: STACK_MONITORING_PRODUCT_STA_S1,
    Swath.S2: STACK_MONITORING_PRODUCT_STA_S2,
    Swath.S3: STACK_MONITORING_PRODUCT_STA_S3,
}


STACK_INPUT_PRODUCTS_ID_LIST = list(L1_STANDARD_PRODUCT_SCS_MAP.values())

STACK_OUTPUT_PRODUCTS_ID_LIST = list(STACK_STANDARD_PRODUCT_STA_MAP.values()) + list(
    STACK_MONITORING_PRODUCT_STA_MAP.values()
)

PRODUCT_LEVEL_TO_ID_MAP = {
    ProductsLevel.L1_STANDARD_SCS: L1_STANDARD_PRODUCT_SCS_MAP,
    ProductsLevel.STACK_STANDARD_STA: STACK_STANDARD_PRODUCT_STA_MAP,
    ProductsLevel.STACK_MONITORING_STA: STACK_MONITORING_PRODUCT_STA_MAP,
}


AUX_PPS_PRODUCT = "AUX_PPS___"

AUX_PRODUCTS_ID_LIST = [
    AUX_PPS_PRODUCT,
]

PROCESSING_PARAMS_PRIMARY_COREG = "primary_image"
PROCESSING_PARAMS_PRIMARY_CAL = "calibration_primary_image"
PROCESSING_PARAMS_START_RANGE = "range_start_time"
PROCESSING_PARAMS_STOP_RANGE = "range_stop_time"
PROCESSING_PARAMS_ID_LIST = [
    PROCESSING_PARAMS_PRIMARY_COREG,
    PROCESSING_PARAMS_PRIMARY_CAL,
    PROCESSING_PARAMS_START_RANGE,
    PROCESSING_PARAMS_STOP_RANGE,
]

CONFIGURATION_FILES_STAPCONF = "STA_P_Conf"
CONFIGURATION_FILES_DEM_DIR = "DEM"
CONFIGURATION_FILES_FNF_DIR = "FNF"
CONFIGURATION_FILES_LCM_DIR = "LCM"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_STAPCONF,
    CONFIGURATION_FILES_DEM_DIR,
    CONFIGURATION_FILES_FNF_DIR,
    CONFIGURATION_FILES_LCM_DIR,
]


class InvalidStackJobOrder(ValueError):
    """Handle invalid STA_P Job Orders."""


def fill_stack_auxiliary_products(products: dict[str, list]) -> Path:
    """
    Fill the structure with the input products found in the input dict

    when found, items are removed from the products list.

    Parameters
    ----------
    products : dict[str, str]
        input/output, list of products; updated by the function.

    Return
    ------
    StackAuxiliaryProducts
        structure.

    """
    aux_pps_product = products.pop(AUX_PPS_PRODUCT)
    assert len(aux_pps_product) == 1
    aux_pps_product = aux_pps_product[0]

    return Path(aux_pps_product)


def retrieve_stack_input_and_aux_products(
    input_products_list: list[joborder_models.JoInputType],
    processing_swath: Swath,
) -> tuple[tuple[Path, ...], Path]:
    """
    Retrieve input and auxiliary products from the input products section.

    Parameters
    ----------
    input_products_list : list[joborder_models.JoInputType]
        list of input products tags.
    processing_swath : Swath
        which swath to process.

    Raises
    ------
    InvalidStackJobOrder
        in case of unexpected input products identifiers, missing required input products or mismatches in the swath.

    Return
    ------
    tuple[Union[L1StripmapInputProducts, L1RXOnlyInputProducts], L1AuxiliaryProducts]
        two structures, one for the input products and one for the auxiliary products.

    """
    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    l1_input_products = []
    for file_id in input_products:
        if file_id not in STACK_INPUT_PRODUCTS_ID_LIST + AUX_PRODUCTS_ID_LIST:
            raise InvalidStackJobOrder(f"Unexpected input product identifier: {file_id}")

    input_standard_product = input_products.pop(L1_STANDARD_PRODUCT_SCS_MAP[processing_swath])

    if processing_swath in STRIPMAP_SWATHS:
        l1_input_products = (Path(path) for path in input_standard_product)

    aux_products = fill_stack_auxiliary_products(input_products)

    if len(input_products) > 0:
        raise InvalidStackJobOrder(f"Unexpected input products: {input_products}")

    l1_input_products = tuple(l1_input_products)

    return l1_input_products, aux_products


def retrieve_stack_output_products(
    output_products_list: list[joborder_models.JoOutputType],
    processing_swath: Swath,
) -> StackOutputProducts:
    """
    Retrieve output products from the output products section.

    Parameters
    ----------
    output_products_list : list[joborder_models.JoOutputType]
        output products tags.
    processing_swath : Swath
        which swath to process.

    Raises
    ------
    InvalidStackJobOrder
        in case of missing required outputs.

    Return
    ------
    StackOutputProducts
        Structure containing the output products.

    """
    output_products, output_directory, output_baseline = flatten_output_products(output_products_list)

    for file_id in output_products:
        if file_id not in STACK_OUTPUT_PRODUCTS_ID_LIST:
            raise InvalidStackJobOrder(f"Unexpected output product identifier: {file_id}")

    if STACK_STANDARD_PRODUCT_STA_MAP[processing_swath] not in output_products:
        raise InvalidStackJobOrder(
            "Missing required standard output product: {:s}".format(STACK_STANDARD_PRODUCT_STA_MAP[processing_swath]),
        )
    if STACK_MONITORING_PRODUCT_STA_MAP[processing_swath] not in output_products:
        raise InvalidStackJobOrder(
            "Missing required monitoring output product: {:s}".format(
                STACK_MONITORING_PRODUCT_STA_MAP[processing_swath]
            ),
        )

    sta_output_products = StackOutputProducts(
        output_directory=Path(output_directory),
        sta_standard_required=True,
        sta_monitoring_required=True,
        product_baseline=output_baseline,
    )

    return sta_output_products


def retrieve_stack_processing_parameters(
    metadata_parameters: list[joborder_models.ParameterType],
) -> StackProcessingParameters:
    """
    Retrieve L1 processing parameters from the parameters section.

    Parameters
    ----------
    metadata_parameters : list[joborder_models.ParameterType]
        list of processing parameters tags.

    Raises
    ------
    InvalidStackJobOrder
        in case of unexpected processing parameters id or missing required parameters.

    Return
    ------
    StackProcessingParameters
        the struct containing the processing parameters.

    """
    parameters = flatten_processing_params(metadata_parameters)

    for param_id in parameters:
        if param_id not in PROCESSING_PARAMS_ID_LIST:
            raise InvalidStackJobOrder(f"Unexpected input processing parameter identifier: {param_id}")

    sta_processing_parameters = StackProcessingParameters()

    primary_coreg = parameters.pop(PROCESSING_PARAMS_PRIMARY_COREG, None)
    if primary_coreg:
        sta_processing_parameters.primary_image = Path(primary_coreg)

    primary_call = parameters.pop(PROCESSING_PARAMS_PRIMARY_CAL, None)
    if primary_call:
        sta_processing_parameters.calibration_primary_image = Path(primary_call)

    start_range = parameters.pop(PROCESSING_PARAMS_START_RANGE, None)
    stop_range = parameters.pop(PROCESSING_PARAMS_STOP_RANGE, None)
    if start_range and stop_range:
        sta_processing_parameters.range_interval = (
            float(start_range),
            float(stop_range),
        )
    elif start_range or stop_range:
        raise InvalidStackJobOrder(
            "Invalid input processing parameter section:"
            + f" {PROCESSING_PARAMS_START_RANGE} and {PROCESSING_PARAMS_STOP_RANGE} must be specified together"
        )

    if len(parameters) > 0:
        raise InvalidStackJobOrder(f"Unexpected processing parameters: {parameters}")

    return sta_processing_parameters


def retrieve_configuration_files(
    configuration_files_list: list[joborder_models.CfgFileType],
) -> tuple[StackExternalsProducts, str | None]:
    """Retrieve configuration files from the section.

    Parameters
    ----------
    configuration_files_list : list[joborder_models.CfgFileType]
        list of configuration files tag.

    Raises
    ------
    InvalidStackJobOrder
        unexpected configuration files id.

    Return
    ------
    external_products : StackExternalProducts
        The STA_P external inputs (e.g. DEM, FNF)

    stap_conf : str | None
        Possibly, the STA_P configuration file (i.e. AUX_PPS)

    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id not in CONFIGURATION_FILES_ID_LIST:
            raise InvalidStackJobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    stap_conf = configuration_files.pop(CONFIGURATION_FILES_STAPCONF, None)
    dem_dir = configuration_files.pop(CONFIGURATION_FILES_DEM_DIR, None)
    fnf_dir = configuration_files.pop(CONFIGURATION_FILES_FNF_DIR, None)
    lcm_dir = configuration_files.pop(CONFIGURATION_FILES_LCM_DIR, None)

    external_products = StackExternalsProducts(
        dem_database_entry_point=_optional_path(dem_dir),
        fnf_database_entry_point=_optional_path(fnf_dir),
        lcm_database_entry_point=_optional_path(lcm_dir),
    )

    if len(configuration_files) > 0:
        raise InvalidStackJobOrder(f"Unexpected configuration files: {configuration_files}")

    return external_products, stap_conf


def translate_model_to_stack_job_order(
    job_order: joborder_models.JobOrder,
) -> StackJobOrder:
    """Translate the job order model into a Stack processor job order object.

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        the object representing the input xml as provided by the XML parser.

    Raises
    ------
    InvalidStackJobOrder
        If the job_order_content is not compatible with a L1 Processor job order.

    Return
    ------
    StackJobOrder
        Object containing the job order for the Stack processor task.

    """
    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidStackJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    assert job_order.processor_configuration is not None
    processor_configuration = retrieve_configuration_params(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAME,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    device_resources = retrieve_device_resources(task)

    assert task.list_of_proc_parameters is not None
    stack_processing_parameters = retrieve_stack_processing_parameters(task.list_of_proc_parameters.proc_parameter)

    assert task.list_of_cfg_files is not None
    external_products, stap_conf = retrieve_configuration_files(task.list_of_cfg_files.cfg_file)

    assert task.list_of_inputs is not None
    input_products = flatten_input_products_allow_multiple_products(task.list_of_inputs.input)

    processing_swath = retrieve_swath_from_products_identifiers(
        list(input_products.keys()),
        PRODUCT_LEVEL_TO_ID_MAP[ProductsLevel.L1_STANDARD_SCS],
    )
    if processing_swath is None:
        raise InvalidStackJobOrder("Missing L1 stack")

    stack_input_products, auxiliary_products = retrieve_stack_input_and_aux_products(
        task.list_of_inputs.input, processing_swath
    )
    assert len(stack_input_products) > 1
    assert task.list_of_outputs is not None

    stack_output_products = retrieve_stack_output_products(task.list_of_outputs.output, processing_swath)

    intermediate_outputs = None
    if task.list_of_intermediate_outputs is not None:
        intermediate_outputs = flatten_intermediate_outputs(task.list_of_intermediate_outputs.intermediate_output)

    assert isinstance(stack_input_products, tuple)
    i_products = stack_input_products

    return StackJobOrder(
        input_stack=i_products,
        output_path=stack_output_products,
        auxiliary_files=auxiliary_products,
        processing_parameters=stack_processing_parameters,
        external_products=external_products,
        device_resources=device_resources,
        processor_configuration=processor_configuration,
        config_file=_optional_path(stap_conf),
        intermediate_files=intermediate_outputs,
    )


def _optional_path(path: str | None) -> Path | None:
    """Instantiate a Path object if path is not None, or return None."""
    if path is None or path == "":
        return None
    return Path(path)
