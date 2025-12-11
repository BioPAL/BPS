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
from bps.common.translate_job_order import (
    flatten_configuration_file,
    flatten_input_products_allow_multiple_products,
    flatten_intermediate_outputs,
    flatten_output_products,
    flatten_processing_params,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_single_input_products,
    retrieve_swath_from_products_identifiers,
    retrieve_task,
)
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1AuxiliaryProducts,
    L1JobOrder,
    L1ProcessingParameters,
    L1RXOnlyInputProducts,
    L1RXOnlyOutputProducts,
    L1RXOnlyProducts,
    L1StripmapInputProducts,
    L1StripmapOutputProducts,
    L1StripmapProducts,
)

EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass L1 processor"""

EXPECTED_PROCESSOR_NAME = "L1_P"
"""Processor name for Biomass L1 processor"""

EXPECTED_PROCESSOR_VERSION = "04.22"
"""Processor version for Biomass L1 processor"""

EXPECTED_TASK_NAME = EXPECTED_PROCESSOR_NAME
"""Task name for Biomass L1 processor"""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass L1 processor"""


class ProductsLevel(enum.Enum):
    """Different product levels"""

    L0_STANDARD_RAW = enum.auto()
    L0_MONITORING_RAW = enum.auto()
    L1_STANDARD_SCS = enum.auto()
    L1_STANDARD_DGM = enum.auto()
    L1_MONITORING_SCS = enum.auto()
    L1_STANDARD_PARC_SCS = enum.auto()


L0_STANDARD_PRODUCT_RAW_S1 = "S1_RAW__0S"
L0_STANDARD_PRODUCT_RAW_S2 = "S2_RAW__0S"
L0_STANDARD_PRODUCT_RAW_S3 = "S3_RAW__0S"
L0_STANDARD_PRODUCT_RAW_RO = "RO_RAW__0S"
L0_STANDARD_PRODUCT_RAW_MAP = {
    Swath.S1: L0_STANDARD_PRODUCT_RAW_S1,
    Swath.S2: L0_STANDARD_PRODUCT_RAW_S2,
    Swath.S3: L0_STANDARD_PRODUCT_RAW_S3,
    Swath.RO: L0_STANDARD_PRODUCT_RAW_RO,
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

L1_STANDARD_PRODUCT_SCS_S1 = "S1_SCS__1S"
L1_STANDARD_PRODUCT_SCS_S2 = "S2_SCS__1S"
L1_STANDARD_PRODUCT_SCS_S3 = "S3_SCS__1S"
L1_STANDARD_PRODUCT_SCS_RO = "RO_SCS__1S"
L1_STANDARD_PRODUCT_SCS_MAP = {
    Swath.S1: L1_STANDARD_PRODUCT_SCS_S1,
    Swath.S2: L1_STANDARD_PRODUCT_SCS_S2,
    Swath.S3: L1_STANDARD_PRODUCT_SCS_S3,
    Swath.RO: L1_STANDARD_PRODUCT_SCS_RO,
}


L1_STANDARD_PRODUCT_DGM_S1 = "S1_DGM__1S"
L1_STANDARD_PRODUCT_DGM_S2 = "S2_DGM__1S"
L1_STANDARD_PRODUCT_DGM_S3 = "S3_DGM__1S"
L1_STANDARD_PRODUCT_DGM_MAP = {
    Swath.S1: L1_STANDARD_PRODUCT_DGM_S1,
    Swath.S2: L1_STANDARD_PRODUCT_DGM_S2,
    Swath.S3: L1_STANDARD_PRODUCT_DGM_S3,
}

L1_MONITORING_PRODUCT_SCS_S1 = "S1_SCS__1M"
L1_MONITORING_PRODUCT_SCS_S2 = "S2_SCS__1M"
L1_MONITORING_PRODUCT_SCS_S3 = "S3_SCS__1M"
L1_MONITORING_PRODUCT_SCS_MAP = {
    Swath.S1: L1_MONITORING_PRODUCT_SCS_S1,
    Swath.S2: L1_MONITORING_PRODUCT_SCS_S2,
    Swath.S3: L1_MONITORING_PRODUCT_SCS_S3,
}

L1_STANDARD_PRODUCT_SCS_PARC_S1 = "S1_SCSc_1S"
L1_STANDARD_PRODUCT_SCS_PARC_S2 = "S2_SCSc_1S"
L1_STANDARD_PRODUCT_SCS_PARC_S3 = "S3_SCSc_1S"
L1_STANDARD_PRODUCT_SCS_PARC_MAP = {
    Swath.S1: L1_STANDARD_PRODUCT_SCS_PARC_S1,
    Swath.S2: L1_STANDARD_PRODUCT_SCS_PARC_S2,
    Swath.S3: L1_STANDARD_PRODUCT_SCS_PARC_S3,
}

L1_OUTPUT_PRODUCTS_ID_LIST = (
    list(L1_STANDARD_PRODUCT_SCS_MAP.values())
    + list(L1_STANDARD_PRODUCT_DGM_MAP.values())
    + list(L1_MONITORING_PRODUCT_SCS_MAP.values())
    + list(L1_STANDARD_PRODUCT_SCS_PARC_MAP.values())
)

PRODUCT_LEVEL_TO_ID_MAP = {
    ProductsLevel.L0_STANDARD_RAW: L0_STANDARD_PRODUCT_RAW_MAP,
    ProductsLevel.L0_MONITORING_RAW: L0_MONITORING_PRODUCT_RAW_MAP,
    ProductsLevel.L1_STANDARD_SCS: L1_STANDARD_PRODUCT_SCS_MAP,
    ProductsLevel.L1_STANDARD_DGM: L1_STANDARD_PRODUCT_DGM_MAP,
    ProductsLevel.L1_MONITORING_SCS: L1_MONITORING_PRODUCT_SCS_MAP,
    ProductsLevel.L1_STANDARD_PARC_SCS: L1_STANDARD_PRODUCT_SCS_PARC_MAP,
}

AUX_ORB_PRODUCT = "AUX_ORB___"
AUX_ATT_PRODUCT = "AUX_ATT___"
AUX_TEC_PRODUCT = "AUX_TEC___"
AUX_INS_PRODUCT = "AUX_INS___"
AUX_PP1_PRODUCT = "AUX_PP1___"
AUX_PARC_INFO_PRODUCT = "PARC_INFO_"

AUX_PRODUCTS_ID_LIST = [
    AUX_ORB_PRODUCT,
    AUX_ATT_PRODUCT,
    AUX_TEC_PRODUCT,
    AUX_INS_PRODUCT,
    AUX_PP1_PRODUCT,
    AUX_PARC_INFO_PRODUCT,
]

PROCESSING_PARAMS_FRAMEID = "frame_id"
PROCESSING_PARAMS_FRAMESTATUS = "frame_status"
PROCESSING_PARAMS_STARTRANGE = "range_start_time"
PROCESSING_PARAMS_STOPRANGE = "range_stop_time"
PROCESSING_PARAMS_RFIFLAG = "rfi_mitigation_flag"
PROCESSING_PARAMS_ID_LIST = [
    PROCESSING_PARAMS_FRAMEID,
    PROCESSING_PARAMS_FRAMESTATUS,
    PROCESSING_PARAMS_STARTRANGE,
    PROCESSING_PARAMS_STOPRANGE,
    PROCESSING_PARAMS_RFIFLAG,
]

CONFIGURATION_FILES_DEM_DIR = "DEM"
CONFIGURATION_FILES_GMF_DIR = "GMF"
CONFIGURATION_FILES_IRI_DIR = "IRI"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_DEM_DIR,
    CONFIGURATION_FILES_GMF_DIR,
    CONFIGURATION_FILES_IRI_DIR,
]

INTERMEDIATE_DATA_DIR = "IntermediateDataDir"


class InvalidL1JobOrder(ValueError):
    """Raised when failing to translate a joborder meant for the L1 Processor"""


def fill_l1_auxiliary_products(products: dict[str, str], aux_tec_products: list[str]) -> L1AuxiliaryProducts:
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
    try:
        aux_orb_product = products.pop(AUX_ORB_PRODUCT)
        aux_att_product = products.pop(AUX_ATT_PRODUCT)
        aux_ins_product = products.pop(AUX_INS_PRODUCT)
        aux_pp1_product = products.pop(AUX_PP1_PRODUCT)
    except KeyError as exc:
        raise RuntimeError(f"Aux product '{exc.args[0]}' section not found in JobOrder") from exc

    parc_info_product = products.pop(AUX_PARC_INFO_PRODUCT, None)

    return L1AuxiliaryProducts(
        Path(aux_orb_product),
        Path(aux_att_product),
        [Path(p) for p in aux_tec_products],
        Path(aux_ins_product),
        Path(aux_pp1_product),
        calibration_site_information=(Path(parc_info_product) if parc_info_product else None),
    )


def retrieve_l1_input_and_aux_products(
    input_products_list: list[joborder_models.JoInputType],
    processing_swath: Swath,
) -> tuple[L1StripmapInputProducts | L1RXOnlyInputProducts, L1AuxiliaryProducts]:
    """Retrieve input and auxiliary products from the input products section

    Parameters
    ----------
    input_products_list : List[joborder_models.JoInputType]
        list of input products tags
    processing_swath : Swath
        which swath to process

    Returns
    -------
    Tuple[Union[L1StripmapInputProducts, L1RXOnlyInputProducts], L1AuxiliaryProducts]
        two structures, one for the input products and one for the auxiliary products

    Raises
    ------
    InvalidL1JobOrder
        in case of unexpected input products identifiers, missing required input products or mismatches in the swath
    """
    input_products_multiple_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products_multiple_products:
        if file_id not in L0_INPUT_PRODUCTS_ID_LIST + AUX_PRODUCTS_ID_LIST:
            raise InvalidL1JobOrder(f"Unexpected input product identifier: {file_id}")

    aux_tec_products = input_products_multiple_products.pop(AUX_TEC_PRODUCT, [])

    input_products = retrieve_single_input_products(input_products_multiple_products)

    input_standard_product = input_products.pop(L0_STANDARD_PRODUCT_RAW_MAP[processing_swath])

    if processing_swath in STRIPMAP_SWATHS:
        monitoring_product_swath = retrieve_swath_from_products_identifiers(
            list(input_products.keys()),
            PRODUCT_LEVEL_TO_ID_MAP[ProductsLevel.L0_MONITORING_RAW],
        )

        if monitoring_product_swath is None:
            raise InvalidL1JobOrder("Missing L0 monitoring product")

        if monitoring_product_swath != processing_swath:
            raise InvalidL1JobOrder(
                f"Invalid input L0 monitoring product swath: {monitoring_product_swath} != {processing_swath}"
            )

        input_monitoring_product = input_products.pop(L0_MONITORING_PRODUCT_RAW_MAP[monitoring_product_swath])

        l1_input_products = L1StripmapInputProducts(
            input_standard=Path(input_standard_product),
            input_monitoring=(Path(input_monitoring_product) if input_monitoring_product else None),
        )
    else:
        assert processing_swath == Swath.RO
        l1_input_products = L1RXOnlyInputProducts(input_standard=Path(input_standard_product))

    aux_products = fill_l1_auxiliary_products(input_products, aux_tec_products)

    if len(input_products) > 0:
        raise InvalidL1JobOrder(f"Unexpected input products: {input_products}")

    return l1_input_products, aux_products


def retrieve_l1_output_products(
    output_products_list: list[joborder_models.JoOutputType],
    processing_swath: Swath,
) -> L1StripmapOutputProducts | L1RXOnlyOutputProducts:
    """Retrieve output products from the output products section

    Parameters
    ----------
    output_products_list : List[joborder_models.JoOutputType]
        output products tags
    processing_swath : Swath
        which swath to process

    Returns
    -------
    Union[L1StripmapOutputProducts, L1RXOnlyOutputProducts]
        Structure containing the output products

    Raises
    ------
    InvalidL1JobOrder
        in case of missing required output products
    """
    output_products, output_directory, output_baseline = flatten_output_products(output_products_list)

    for file_id in output_products:
        if file_id not in L1_OUTPUT_PRODUCTS_ID_LIST:
            raise InvalidL1JobOrder(f"Unexpected output product identifier: {file_id}")

    if L1_STANDARD_PRODUCT_SCS_MAP[processing_swath] not in output_products:
        raise InvalidL1JobOrder(f"Missing required output product: {L1_STANDARD_PRODUCT_SCS_MAP[processing_swath]}")

    if processing_swath in STRIPMAP_SWATHS:
        if L1_MONITORING_PRODUCT_SCS_MAP[processing_swath] not in output_products:
            raise InvalidL1JobOrder("Missing L1 monitoring SCS product")

        l1_output_products = L1StripmapOutputProducts(
            output_directory=Path(output_directory),
            output_baseline=output_baseline,
            scs_standard_required=True,
            scs_monitoring_required=True,
            dgm_standard_required=L1_STANDARD_PRODUCT_DGM_MAP[processing_swath] in output_products,
        )
    else:
        assert processing_swath == Swath.RO
        l1_output_products = L1RXOnlyOutputProducts(
            output_directory=Path(output_directory),
            output_baseline=output_baseline,
            scs_standard_required=True,
        )

    return l1_output_products


def retrieve_l1_processing_parameters(
    metadata_parameters: list[joborder_models.ParameterType],
) -> L1ProcessingParameters:
    """Retrieve L1 processing parameters from the parameters section

    Parameters
    ----------
    metadata_parameters : List[joborder_models.ParameterType]
        list of processing parameters tags

    Returns
    -------
    L1ProcessingParameters
        the struct containing the processing parameters

    Raises
    ------
    InvalidL1JobOrder
        in case of unexpected processing parameters id or missing required parameters
    """
    parameters = flatten_processing_params(metadata_parameters)

    for param_id in parameters:
        if param_id not in PROCESSING_PARAMS_ID_LIST:
            raise InvalidL1JobOrder(f"Unexpected input processing parameter identifier: {param_id}")

    l1_processing_parameters = L1ProcessingParameters()

    frame_id = parameters.pop(PROCESSING_PARAMS_FRAMEID, None)
    if frame_id:
        if frame_id == "___":
            l1_processing_parameters.frame_id = 0
        else:
            l1_processing_parameters.frame_id = int(frame_id)

    frame_status = parameters.pop(PROCESSING_PARAMS_FRAMESTATUS, None)
    if frame_status:
        l1_processing_parameters.frame_status = frame_status

    start_range = parameters.pop(PROCESSING_PARAMS_STARTRANGE, None)
    stop_range = parameters.pop(PROCESSING_PARAMS_STOPRANGE, None)
    if start_range and stop_range:
        l1_processing_parameters.range_interval = (
            float(start_range),
            float(stop_range),
        )
    elif start_range or stop_range:
        raise InvalidL1JobOrder(
            "Invalid input processing parameter section:"
            + f" {PROCESSING_PARAMS_STARTRANGE} and {PROCESSING_PARAMS_STOPRANGE} must be specified togheter"
        )

    rfi_flag = parameters.pop(PROCESSING_PARAMS_RFIFLAG, None)
    if rfi_flag:
        if rfi_flag.lower() in ["true", "false"]:
            l1_processing_parameters.rfi_mitigation_enabled = rfi_flag == "true"
        else:
            raise InvalidL1JobOrder(f"Invalid input {PROCESSING_PARAMS_RFIFLAG}: cannot convert {rfi_flag} to bool")

    if len(parameters) > 0:
        raise InvalidL1JobOrder(f"Unexpected processing parameters: {parameters}")

    return l1_processing_parameters


def retrieve_configuration_files(
    configuration_files_list: list[joborder_models.CfgFileType],
) -> tuple[str | None, str | None, str | None]:
    """Retrieve configuration files from the section

    Parameters
    ----------
    configuration_files_list : List[joborder_models.CfgFileType]
        list of configuration files tag

    Returns
    -------
    Tuple[Optional[str], Optional[str], Optional[str]]
        dem directory, gmf product, iri directory

    Raises
    ------
    InvalidL1JobOrder
        unexpected configuration files id
    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id not in CONFIGURATION_FILES_ID_LIST:
            raise InvalidL1JobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    dem_dir = configuration_files.pop(CONFIGURATION_FILES_DEM_DIR, None)
    gmf_product = configuration_files.pop(CONFIGURATION_FILES_GMF_DIR, None)
    iri_dir = configuration_files.pop(CONFIGURATION_FILES_IRI_DIR, None)

    if len(configuration_files) > 0:
        raise InvalidL1JobOrder(f"Unexpected configuration files: {configuration_files}")

    return dem_dir, gmf_product, iri_dir


def translate_model_to_l1_job_order(job_order: joborder_models.JobOrder) -> L1JobOrder:
    """Translate the job order model into a L1 processor job order object.

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        the object representing the input xml as provided by the XML parser.

    Returns
    -------
    L1JobOrder
        Object containing the job order for the L1 processor task.

    Raises
    ------
    InvalidL1JobOrder
        If the job_order_content is not compatible with a L1 Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidL1JobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    assert job_order.processor_configuration is not None
    processor_configuration = retrieve_configuration_params(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAME,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    device_resources = retrieve_device_resources(task)

    assert task.list_of_proc_parameters is not None
    l1_processing_parameters = retrieve_l1_processing_parameters(task.list_of_proc_parameters.proc_parameter)

    assert task.list_of_cfg_files is not None
    dem_dir, gmf_product, iri_dir = retrieve_configuration_files(task.list_of_cfg_files.cfg_file)

    assert task.list_of_inputs is not None
    input_products_identifiers = list(flatten_input_products_allow_multiple_products(task.list_of_inputs.input).keys())

    processing_swath = retrieve_swath_from_products_identifiers(
        input_products_identifiers,
        PRODUCT_LEVEL_TO_ID_MAP[ProductsLevel.L0_STANDARD_RAW],
    )
    if processing_swath is None:
        raise InvalidL1JobOrder("Missing L0 product")

    l1_input_products, auxiliary_products = retrieve_l1_input_and_aux_products(
        task.list_of_inputs.input, processing_swath
    )

    assert task.list_of_outputs is not None
    l1_output_products = retrieve_l1_output_products(task.list_of_outputs.output, processing_swath)

    assert task.list_of_intermediate_outputs is not None
    intermediate_outputs = flatten_intermediate_outputs(task.list_of_intermediate_outputs.intermediate_output)

    intermediate_data_dir = intermediate_outputs.pop(INTERMEDIATE_DATA_DIR, None)
    if len(intermediate_outputs) > 0:
        raise InvalidL1JobOrder(
            f"Unexpected values in list of intermediate outputs: {set(intermediate_outputs.keys())}"
        )

    if processing_swath in STRIPMAP_SWATHS:
        assert isinstance(l1_input_products, L1StripmapInputProducts)
        assert isinstance(l1_output_products, L1StripmapOutputProducts)
        io_products = L1StripmapProducts(input=l1_input_products, output=l1_output_products)
    else:
        assert isinstance(l1_input_products, L1RXOnlyInputProducts)
        assert isinstance(l1_output_products, L1RXOnlyOutputProducts)
        io_products = L1RXOnlyProducts(input=l1_input_products, output=l1_output_products)
        assert processing_swath == Swath.RO

    return L1JobOrder(
        io_products=io_products,
        auxiliary_files=auxiliary_products,
        processing_parameters=l1_processing_parameters,
        dem_database_entry_point=Path(dem_dir) if dem_dir is not None else None,
        geomagnetic_field=Path(gmf_product) if gmf_product is not None else None,
        iri_data_folder=Path(iri_dir) if iri_dir is not None else None,
        device_resources=device_resources,
        processor_configuration=processor_configuration,
        intermediate_data_dir=(Path(intermediate_data_dir) if intermediate_data_dir is not None else None),
    )
