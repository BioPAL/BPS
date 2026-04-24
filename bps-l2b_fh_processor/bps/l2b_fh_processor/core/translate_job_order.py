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
    InvalidJobOrder,
    flatten_input_products_allow_multiple_products,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_optional_configuration_file,
    retrieve_single_output_product,
    retrieve_task,
    retrieve_tile_processing_parameters,
)
from bps.l2b_fh_processor.core.joborder_l2b_fh import (
    L2bFHJobOrder,
)

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

AUX_PP2B_FH_PRODUCT = AUX_PP_INPUT

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
    InvalidJobOrder
        in case of unexpected input products identifiers, missing required input products
    """

    input_products = flatten_input_products_allow_multiple_products(input_products_list)

    for file_id in input_products:
        if file_id not in L2B_FH_INPUT_ID_LIST:
            raise InvalidJobOrder(f"Unexpected input identifier: {file_id}")

    if AUX_PP_INPUT not in input_products:
        raise InvalidJobOrder(f"Missing required input: {AUX_PP_INPUT}")
    if L2A_PRODUCT_FH not in input_products and L2A_PRODUCT_TFH not in input_products:
        raise InvalidJobOrder(f"Missing required input: one of {L2A_PRODUCT_FH} or {L2A_PRODUCT_TFH}")

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
    InvalidJobOrder
        If the job_order_content is not compatible with a L2b FH Processor job order.
    """

    if job_order.schema_name != EXPECTED_SCHEMA_NAME:
        raise InvalidJobOrder(f"Invalid schema name: {job_order.schema_name} != {EXPECTED_SCHEMA_NAME}")

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

    l2b_fh_processing_parameters = retrieve_tile_processing_parameters(
        task.list_of_proc_parameters.proc_parameter, PROCESSING_PARAMS_TILE_ID
    )

    l2b_fh_p_conf = retrieve_optional_configuration_file(
        task.list_of_cfg_files.cfg_file, CONFIGURATION_FILES_L2BFHPCONF
    )

    (
        input_l2a_products,
        aux_pp2_fh_path,
        input_l2b_fd_product,
    ) = translate_l2b_fh_list_of_inputs(task.list_of_inputs.input)

    (
        output_directory,
        output_product,
        output_baseline,
    ) = retrieve_single_output_product(task.list_of_outputs.output, [L2B_OUTPUT_PRODUCT_FH, L2B_OUTPUT_PRODUCT_TFH])

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
