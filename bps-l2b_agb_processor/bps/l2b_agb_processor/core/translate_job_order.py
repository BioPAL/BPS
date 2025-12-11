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
from bps.common.l2_joborder_tags import L2B_OUTPUT_PRODUCT_AGB, L2B_OUTPUT_PRODUCT_FD
from bps.common.translate_job_order import (
    flatten_configuration_file,
    flatten_input_products_allow_multiple_products,
    flatten_output_products,
    flatten_processing_params,
    retrieve_configuration_params_l2,
    retrieve_device_resources,
    retrieve_task,
)
from bps.l2b_agb_processor.core.joborder_l2b_agb import (
    L2bAGBJobOrder,
    L2BAGBProcessingParameters,
)


class InvalidJobOrder(ValueError):
    """Raised when failing to translate a joborder"""


EXPECTED_SCHEMA_NAME = r"BIOMASS CPF-Processor ICD"
"""Schema name for Biomass L2b AGB processor"""

EXPECTED_PROCESSOR_NAME = "L2B_AGB_P"
"""Processor name for Biomass L2b AGB processor"""

EXPECTED_PROCESSOR_ALIAS_YES_FD_NO_AGB = "L2B_AGB_P_withFD_withoutAGB"
"""Alternative alias for Biomass L2b AGB processor"""

EXPECTED_PROCESSOR_ALIAS_NO_FD_NO_AGB = "L2B_AGB_P_withoutFD_withoutAGB"
"""Alternative alias for Biomass L2b AGB processor"""

EXPECTED_PROCESSOR_ALIAS_NO_FD_YES_AGB = "L2B_AGB_P_withoutFD_withAGB"
"""Alternative alias for Biomass L2b AGB processor"""

EXPECTED_PROCESSOR_ALIAS_YES_FD_YES_AGB = "L2B_AGB_P_withFD_withAGB"
"""Alternative alias for Biomass L2b AGB processor"""

EXPECTED_PROCESSOR_NAMES_LIST = [
    EXPECTED_PROCESSOR_NAME,
    EXPECTED_PROCESSOR_ALIAS_YES_FD_NO_AGB,
    EXPECTED_PROCESSOR_ALIAS_NO_FD_NO_AGB,
    EXPECTED_PROCESSOR_ALIAS_NO_FD_YES_AGB,
    EXPECTED_PROCESSOR_ALIAS_YES_FD_YES_AGB,
]
"""All possible alias for Biomass L2b FH processors"""

EXPECTED_PROCESSOR_VERSION = "04.22"
"""Processor version for Biomass L2b AGB processor"""

EXPECTED_TASK_NAME = EXPECTED_PROCESSOR_NAME
"""Task name for Biomass L2b AGB processor"""

EXPECTED_TASK_VERSION = EXPECTED_PROCESSOR_VERSION
"""Task version for Biomass L2b AGB processor"""

L2A_PRODUCT_GN = "FP_GN__L2A"

AUX_PP_INPUT = "AUX_PP2_AB"

L2B_PRODUCT_FD = "FP_FD__L2B"

L2B_PRODUCT_AGB = "FP_AGB_L2B"

L2B_AGB_INPUT_ID_LIST = [
    L2A_PRODUCT_GN,
    AUX_PP_INPUT,
    L2B_PRODUCT_FD,
    L2B_PRODUCT_AGB,
]

AUX_PP2B_AGB_PRODUCT = "AUX_PP2_AB"

CONFIGURATION_FILES_L2BAGBPCONF = "L2B_AGB_P_Conf"
CONFIGURATION_FILES_LCM_DIR = "LCM"
CONFIGURATION_FILES_CAL_AB_DIR = "CAL_AB"
CONFIGURATION_FILES_ID_LIST = [
    CONFIGURATION_FILES_L2BAGBPCONF,
    CONFIGURATION_FILES_LCM_DIR,
    CONFIGURATION_FILES_CAL_AB_DIR,
]
PROCESSING_PARAMS_TILE_ID = "central_tile_id"

PROCESSING_PARAMS_ID_LIST = [PROCESSING_PARAMS_TILE_ID]


class InvalidL2bAGBJobOrder(ValueError):
    """Raised when failing to translate a joborder meant for the L2b AGB Processor"""


def translate_l2b_agb_list_of_inputs(
    input_products_list: list[joborder_models.JoInputType],
) -> tuple[tuple[Path, ...], Path, Path, Path, Path, Path]:
    """Retrieve, from the input products section, paths of L1c stack acquisitions,
    aux_pp2_agb file and optionally the AGB L2a product.

    Parameters
    ----------
    input_products_list : List[joborder_models.JoInputType]
        list of input products tags

    Returns
    -------
    Tuple[Tuple[Path,...], Path, Path, Path, Path, Path]:
        one tuple of Paths for the input stack products
        one single Path for
          AUX PP2 AGB Configuration
          input_l2b_fd_products (can be "None")
          input_l2b_agb_products (can be "None")

    Raises
    ------
    InvalidL2bAGBJobOrder
        in case of unexpected input products identifiers, missing required input products
    """

    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products:
        if file_id not in L2B_AGB_INPUT_ID_LIST:
            raise InvalidL2bAGBJobOrder(f"Unexpected input identifier: {file_id}")

    input_l2a_products = input_products.pop(L2A_PRODUCT_GN)
    input_l2a_products = tuple(Path(input_l2a_path) for input_l2a_path in input_l2a_products)

    input_l2a_mph_files = []
    for input_acquisition in input_l2a_products:
        name = Path(str(input_acquisition.name).lower() + ".xml")
        input_l2a_mph_files.append(input_acquisition.joinpath(name))
    input_l2a_mph_files = tuple(input_l2a_mph_files)

    aux_pp2_agb_path = Path(input_products.pop(AUX_PP_INPUT)[0])

    if len(input_products) > 0 and L2B_PRODUCT_FD in input_products:
        # input_l2b_fd_products is optional
        paths_string = input_products.pop(L2B_PRODUCT_FD)

        input_l2b_fd_products = [Path(pp) for pp in paths_string]
    else:
        input_l2b_fd_products = None

    if len(input_products) > 0 and L2B_PRODUCT_AGB in input_products:
        # input_l2b_agb_products is optional
        paths_string = input_products.pop(L2B_PRODUCT_AGB)

        input_l2b_agb_products = [Path(pp) for pp in paths_string]

    else:
        input_l2b_agb_products = None

    if len(input_products) > 0:
        raise InvalidL2bAGBJobOrder(f"Unexpected input products: {input_products}")

    return (
        input_l2a_products,
        input_l2a_mph_files,
        aux_pp2_agb_path,
        input_l2b_fd_products,
        input_l2b_agb_products,
    )


def retrieve_configuration_files(
    configuration_files_list: list[joborder_models.CfgFileType],
) -> tuple[Path | None, Path | None, Path | None]:
    """Retrieve configuration files from the section

    Parameters
    ----------
    configuration_files_list : List[joborder_models.CfgFileType]
        list of configuration files tag

    Returns
    -------
    Path
        dem directory, l2b agb configuration file

    Raises
    ------
    InvalidL2bAGBJobOrder
        unexpected configuration files id
    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id not in CONFIGURATION_FILES_ID_LIST:
            raise InvalidL2bAGBJobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    l2b_p_conf = configuration_files.pop(CONFIGURATION_FILES_L2BAGBPCONF, None)
    if l2b_p_conf is not None:
        l2b_p_conf = Path(l2b_p_conf)

    lcm_dir = configuration_files.pop(CONFIGURATION_FILES_LCM_DIR, None)
    if lcm_dir is not None:
        lcm_dir = Path(lcm_dir)

    cal_ab_dir = configuration_files.pop(CONFIGURATION_FILES_CAL_AB_DIR, None)
    if cal_ab_dir is not None:
        cal_ab_dir = Path(cal_ab_dir)

    if len(configuration_files) > 0:
        raise InvalidL2bAGBJobOrder(f"Unexpected configuration files: {configuration_files}")

    return lcm_dir, cal_ab_dir, l2b_p_conf


def retrieve_l2b_agb_processing_parameters(
    metadata_parameters: list[joborder_models.ParameterType],
) -> L2BAGBProcessingParameters:
    """Retrieve Proc parameters from the section

    Parameters
    ----------
    proc_parameters_list : List[joborder_models.ParameterType]
        list of processing parameters

    Returns
    -------
    L2BAGBProcessingParameters
        the struct containing the processing parameters

    Raises
    ------
    InvalidL2bAGBJobOrder
        unexpected configuration files id
    """
    parameters = flatten_processing_params(metadata_parameters)

    for param_id in parameters:
        if param_id not in PROCESSING_PARAMS_ID_LIST:
            raise InvalidL2bAGBJobOrder(f"Unexpected input processing parameter identifier: {param_id}")

    l2b_agb_processing_parameters = L2BAGBProcessingParameters()

    tile_id = parameters.pop(PROCESSING_PARAMS_TILE_ID, None)
    if tile_id:
        l2b_agb_processing_parameters.tile_id = tile_id

    if len(parameters) > 0:
        raise InvalidL2bAGBJobOrder(f"Unexpected processing parameters: {parameters}")

    return l2b_agb_processing_parameters


def retrieve_l2b_agb_output_directory(
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
    InvalidL2bAGBJobOrder
        in case of unexpected output products identifiers, missing required output products or mismatches in the swath
    """

    output_products, output_directory, output_baselines = flatten_output_products(output_products_list)
    output_baselines = list(np.ones(len(output_products)).astype(int) * int(output_baselines))

    if len(output_products) > 1:
        raise InvalidL2bAGBJobOrder("Too many output products specified: just one is requested.")

    output_baseline = output_baselines[0]

    file_id = output_products[0]
    if file_id not in L2B_OUTPUT_PRODUCT_AGB:
        raise InvalidL2bAGBJobOrder(f"Unexpected output product identifier: {file_id}")

    if L2B_OUTPUT_PRODUCT_AGB not in output_products[0]:
        raise InvalidL2bAGBJobOrder("Required output product is not an L2B AGB")

    output_products, _, _ = flatten_output_products(output_products_list)
    output_product = output_products[0]

    return Path(output_directory), output_product, output_baseline


def translate_model_to_l2b_agb_job_order(
    job_order: joborder_models.JobOrder,
) -> L2bAGBJobOrder:
    """Translate the job order model into a L2B AGB processor job order object.

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        the object representing the input xml as provided by the XML parser.

    Returns
    -------
    L2bAGBJobOrder
        Object containing the job order for the L2B AGB processor task.

    Raises
    ------
    InvalidL2bAGBJobOrder
        If the job_order_content is not compatible with a L2B AGB Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidL2bAGBJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

    assert job_order.processor_configuration is not None
    processor_configuration = retrieve_configuration_params_l2(
        job_order.processor_configuration,
        EXPECTED_PROCESSOR_NAMES_LIST,
        EXPECTED_PROCESSOR_VERSION,
    )

    task = retrieve_task(job_order, EXPECTED_TASK_NAME, EXPECTED_TASK_VERSION)

    found_optional_l2b_fd = False
    found_optional_l2b_agb = False
    optional_l2b_fd_string = "ABSENT"
    optional_l2b_agb_string = "ABSENT"
    assert job_order.list_of_tasks is not None
    assert job_order.list_of_tasks.task[0].list_of_inputs is not None
    for input_product in job_order.list_of_tasks.task[0].list_of_inputs.input:
        assert input_product.input_id is not None
        if input_product.input_id.value == L2B_OUTPUT_PRODUCT_FD:
            found_optional_l2b_fd = True
            optional_l2b_fd_string = "PRESENT"
        if input_product.input_id.value == L2B_OUTPUT_PRODUCT_AGB:
            found_optional_l2b_agb = True
            optional_l2b_agb_string = "PRESENT"

    assert job_order.processor_configuration.processor_name is not None
    if job_order.processor_configuration.processor_name.value == EXPECTED_PROCESSOR_ALIAS_YES_FD_YES_AGB:
        if not found_optional_l2b_fd or not found_optional_l2b_agb:
            raise InvalidJobOrder(
                f"Invalid processor name: {job_order.processor_configuration.processor_name.value} when optional L2B FD is '{optional_l2b_fd_string}' and optional L2B AGB is '{optional_l2b_agb_string}'"
            )

    assert job_order.processor_configuration.processor_name is not None
    if job_order.processor_configuration.processor_name.value == EXPECTED_PROCESSOR_ALIAS_NO_FD_NO_AGB:
        if found_optional_l2b_fd or found_optional_l2b_agb:
            raise InvalidJobOrder(
                f"Invalid processor name: {job_order.processor_configuration.processor_name.value} when optional L2B FD is '{optional_l2b_fd_string}' and optional L2B AGB is '{optional_l2b_agb_string}'"
            )

    assert job_order.processor_configuration.processor_name is not None
    if job_order.processor_configuration.processor_name.value == EXPECTED_PROCESSOR_ALIAS_NO_FD_YES_AGB:
        if found_optional_l2b_fd or not found_optional_l2b_agb:
            raise InvalidJobOrder(
                f"Invalid processor name: {job_order.processor_configuration.processor_name.value} when optional L2B FD is '{optional_l2b_fd_string}' and optional L2B AGB is '{optional_l2b_agb_string}'"
            )

    assert job_order.processor_configuration.processor_name is not None
    if job_order.processor_configuration.processor_name.value == EXPECTED_PROCESSOR_ALIAS_YES_FD_NO_AGB:
        if not found_optional_l2b_fd or found_optional_l2b_agb:
            raise InvalidJobOrder(
                f"Invalid processor name: {job_order.processor_configuration.processor_name.value} when optional L2B FD is '{optional_l2b_fd_string}' and optional L2B AGB is '{optional_l2b_agb_string}'"
            )

    device_resources = retrieve_device_resources(task)

    assert task.list_of_proc_parameters is not None
    l2b_agb_processing_parameters = retrieve_l2b_agb_processing_parameters(task.list_of_proc_parameters.proc_parameter)

    assert task.list_of_cfg_files is not None
    lcm_product, cal_ab_product, l2b_agb_p_conf = retrieve_configuration_files(task.list_of_cfg_files.cfg_file)

    assert task.list_of_inputs is not None
    (
        input_l2a_products,
        input_l2a_mph_files,
        aux_pp2_agb_path,
        input_l2b_fd_products,
        input_l2b_agb_products,
    ) = translate_l2b_agb_list_of_inputs(task.list_of_inputs.input)

    assert task.list_of_outputs is not None
    (
        output_directory,
        output_product,
        output_baseline,
    ) = retrieve_l2b_agb_output_directory(task.list_of_outputs.output)

    return L2bAGBJobOrder(
        input_l2a_products,
        input_l2a_mph_files,
        output_directory,
        output_product,
        aux_pp2_agb_path,
        lcm_product,
        cal_ab_product,
        device_resources,
        processor_configuration,
        l2b_agb_processing_parameters,
        input_l2b_fd_products=input_l2b_fd_products if input_l2b_fd_products else None,
        input_l2b_agb_products=(input_l2b_agb_products if input_l2b_agb_products else None),
        l2b_agb_p_conf=l2b_agb_p_conf if l2b_agb_p_conf else None,
        output_baseline=output_baseline,
    )
