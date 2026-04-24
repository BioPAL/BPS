# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
General Job Order translation functions
---------------------------------------
"""

import logging
from pathlib import Path

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import Swath
from bps.common.io import joborder_models
from bps.common.joborder import DeviceResources, ProcessorConfiguration, TileProcessingParameters


class InvalidJobOrder(ValueError):
    """Raised when failing to translate a joborder"""


def retrieve_configuration_params(
    configuration: joborder_models.ProcessorConfigurationType,
    expected_processor_name: str | list[str],
    expected_processor_version: str,
) -> ProcessorConfiguration:
    """Retrieve configuration parameters from the processor configuration section

    Parameters
    ----------
    configuration : joborder_models.ProcessorConfigurationType
        processor configuration section
    expected_processor_name : str | list[str]
        expected processor name, or list of accepted names
    expected_processor_version : str
        expected version of processor

    Returns
    -------
    ProcessorConfiguration
        processor configuration

    Raises
    ------
    InvalidJobOrder
        in case of unexpected tags content
    """
    file_class = configuration.file_class

    accepted_names = [expected_processor_name] if isinstance(expected_processor_name, str) else expected_processor_name
    if configuration.processor_name.value not in accepted_names:
        raise InvalidJobOrder(f"Invalid processor name: {configuration.processor_name.value} not in {accepted_names}")

    if configuration.processor_version.value != expected_processor_version:
        raise InvalidJobOrder(
            f"Invalid processor version: {configuration.processor_version.value} != {expected_processor_version}"
        )

    if len(configuration.list_of_stdout_log_levels.stdout_log_level) != 1:
        raise InvalidJobOrder("Unexpected number of stdout log level")
    stdout_log_level = configuration.list_of_stdout_log_levels.stdout_log_level[0]

    if len(configuration.list_of_stderr_log_levels.stderr_log_level) != 1:
        raise InvalidJobOrder("Unexpected number of stderr log level")
    stderr_log_level = configuration.list_of_stderr_log_levels.stderr_log_level[0]

    intermediate_output_enabled = configuration.intermediate_output_enable

    azimuth_interval = None
    if configuration.request.toi:
        toi_start = str(configuration.request.toi.start)
        toi_stop = str(configuration.request.toi.stop)
        if toi_start and toi_stop:
            azimuth_interval = (
                PreciseDateTime.fromisoformat(toi_start),
                PreciseDateTime.fromisoformat(toi_stop),
            )
        elif toi_start or toi_stop:
            raise InvalidJobOrder(
                "Invalid metadata parameter request section:" + " TOI start and stop times must be specified together"
            )

    return ProcessorConfiguration(
        file_class=file_class.value,
        stdout_log_level=ProcessorConfiguration.LogLevel(stdout_log_level.value),
        stderr_log_level=ProcessorConfiguration.LogLevel(stderr_log_level.value),
        keep_intermediate=intermediate_output_enabled,
        azimuth_interval=azimuth_interval,
    )


def retrieve_task(
    job_order: joborder_models.JobOrder,
    expected_task_name: str,
    expected_task_version: str,
) -> joborder_models.JoTaskType:
    """Get task from job order object

    Parameters
    ----------
    job_order : joborder_models.JobOrder
        job order model
    expected_task_name : str
        expected name of the task
    expected_task_version : str
        expected version of the task

    Returns
    -------
    joborder_models.JoTaskType
        the task model

    Raises
    ------
    InvalidJobOrder
        In case of unexpected joborder content
    """
    if len(job_order.list_of_tasks.task) != 1:
        raise InvalidJobOrder("Unexpected number of tasks")

    task = job_order.list_of_tasks.task[0]

    if task.task_name.value != expected_task_name:
        raise InvalidJobOrder(f"Invalid task name: {task.task_name.value} != {expected_task_name}")

    if task.task_version.value != expected_task_version:
        raise InvalidJobOrder(f"Invalid task version: {task.task_version.value} != {expected_task_version}")
    return task


def retrieve_device_resources(task: joborder_models.JoTaskType) -> DeviceResources:
    """Retrieve device resources

    Parameters
    ----------
    task : joborder_models.JoTaskType
        job order task

    Returns
    -------
    DeviceResources
        host available resources
    """
    ramdisk_amount = None
    ramdisk_mount_point = None
    if task.list_of_ramdisks is not None:
        if len(task.list_of_ramdisks.ramdisk) > 1:
            raise RuntimeError("More than one ramdisk found in the JobOrder")

        if len(task.list_of_ramdisks.ramdisk) == 1:
            ramdisk = task.list_of_ramdisks.ramdisk[0]
            ramdisk_amount = ramdisk.amount
            ramdisk_mount_point = Path(ramdisk.mount_path)

    return DeviceResources(
        num_threads=int(task.number_of_cpu_cores),
        available_ram=task.amount_of_ram,
        available_space=task.disk_space.value,
        ramdisk_size=ramdisk_amount,
        ramdisk_mount_point=ramdisk_mount_point,
    )


def retrieve_swath_from_products_identifiers(
    products_identifiers: list[str], swath_to_product_id_map: dict[Swath, str]
) -> Swath | None:
    """Retrieve the swath id from a list of products identifiers.

    Parameters
    ----------
    products_identifiers : List[str]
        list of product identifiers
    swath_to_product_id_map : Dict[Swath, str]
        map swath to product identifier

    Returns
    -------
    Optional[Swath]
        Swath, if exactly one product of the specified level was found

    Raises
    ------
    InvalidJobOrder
        if multiple products of the same level were found
    """
    swaths = [swath for swath, product_id in swath_to_product_id_map.items() if product_id in products_identifiers]

    if len(swaths) == 1:
        return swaths[0]

    if len(swaths) == 0:
        return None

    product_idx = [swath_to_product_id_map[swath] for swath in swaths]
    raise InvalidJobOrder(f"Cannot retrieve swath: multiple products of the same level found {product_idx}")


def flatten_processing_params(
    metadata_parameters: list[joborder_models.ParameterType],
) -> dict[str, str]:
    """Convert processing parameters section to a dictionary

    Parameters
    ----------
    metadata_parameters : List[joborder_models.ParameterType]
        list of processing parameters tags

    Returns
    -------
    Dict[str, str]
        id to string map

    Raises
    ------
    InvalidJobOrder
        in case of duplicated parameters entry
    """
    parameters: dict[str, str] = {}
    for parameter in metadata_parameters:
        if parameter.name in parameters:
            raise InvalidJobOrder(f"Duplicated {parameter.name} parameter entry")

        parameters[parameter.name] = parameter.value

    return parameters


def flatten_configuration_file(
    configuration_files: list[joborder_models.CfgFileType],
) -> dict[str, str]:
    """Convert configuration file section to a dictionary

    Parameters
    ----------
    configuration_files : List[models.CfgFileType]
        list of configuration files tag

    Returns
    -------
    Dict[str, str]
        id to string map

    Raises
    ------
    InvalidJobOrder
        in case of duplicated configuration file entry
    """
    files: dict[str, str] = {}
    for file in configuration_files:
        if file.cfg_id.value in files:
            raise InvalidJobOrder(f"Duplicated {file.cfg_id} configuration file entry")
        files[file.cfg_id.value] = file.cfg_file_name

    return files


def retrieve_single_input_products(inputs: dict[str, list[str]]) -> dict[str, str]:
    """Simplified inputs assuming there is only one product per type"""
    single_products: dict[str, str] = {}
    for file_id, product_list in inputs.items():
        if len(product_list) > 1:
            raise InvalidJobOrder(f"Unexpected multiple input products found for id: {file_id}")

        single_products[file_id] = product_list[0]
    return single_products


def flatten_input_products(
    input_products: list[joborder_models.JoInputType],
) -> dict[str, str]:
    """Convert input file section to a dictionary

    Parameters
    ----------
    input_products : List[joborder_models.JoInputType]
        list of input products tags

    Returns
    -------
    Dict[str, str]
        id to product path map

    Raises
    ------
    InvalidJobOrder
        in case of duplicated input file id entry
    """
    flattened = _flatten_input_products_core(input_products)

    return retrieve_single_input_products(flattened)


def flatten_input_products_allow_multiple_products(
    input_products: list[joborder_models.JoInputType],
) -> dict[str, list[str]]:
    """Convert input file section to a dictionary

    Parameters
    ----------
    input_products : List[joborder_models.JoInputType]
        list of input products tags

    Returns
    -------
    Dict[str, List[str]]
        id to list of products map

    Raises
    ------
    InvalidJobOrder
        in case of duplicated input file id entry
    """
    return _flatten_input_products_core(input_products)


def _flatten_input_products_core(
    input_products: list[joborder_models.JoInputType],
) -> dict[str, list[str]]:
    inputs: dict[str, list[str]] = {}
    for input_product in input_products:
        assert len(input_product.list_of_selected_inputs.selected_input) == 1
        selected_input = input_product.list_of_selected_inputs.selected_input[0]

        assert len(selected_input.list_of_file_names.file_name) >= 1
        if selected_input.file_type.value in inputs:
            raise InvalidJobOrder(f"Duplicated {selected_input.file_type} input file entry")

        inputs[selected_input.file_type.value] = [file.value for file in selected_input.list_of_file_names.file_name]

    return inputs


def flatten_output_products(
    output_products: list[joborder_models.JoOutputType],
) -> tuple[list[str], str, int]:
    """Convert output products section to a dictionary

    Parameters
    ----------
    output_products : List[joborder_models.JoOutputType]
        list of output products tags

    Returns
    -------
    Tuple[List[str], str, int]:
        list of required output products id, output directory and product baseline

    Raises
    ------
    InvalidJobOrder
        in case of duplicated output products entry
    """
    outputs: list[str] = []
    output_directory = None
    output_baseline = None
    for output_product in output_products:
        assert output_product.file_dir is not None
        if output_product.file_type.value in outputs:
            raise InvalidJobOrder(f"Duplicated {output_product.file_type.value} output file entry")

        if output_directory:
            if output_product.file_dir.value != output_directory:
                raise InvalidJobOrder("Multiple output directories specified")
        else:
            output_directory = output_product.file_dir.value

        baseline = int(output_product.baseline.value)
        if output_baseline:
            if baseline != output_baseline:
                raise InvalidJobOrder("Multiple product baselines specified")
        else:
            output_baseline = baseline

        outputs.append(output_product.file_type.value)

    assert output_directory is not None
    assert output_baseline is not None
    return outputs, output_directory, output_baseline


def retrieve_single_output_product(
    output_products_list: list[joborder_models.JoOutputType],
    valid_products: list[str],
) -> tuple[Path, str, int]:
    """Retrieve and validate a single output product from the output products section.

    Parameters
    ----------
    output_products_list : list[joborder_models.JoOutputType]
        output products tags
    valid_products : list[str]
        accepted output product identifiers

    Returns
    -------
    tuple[Path, str, int]
        Output directory, output product identifier, output baseline.

    Raises
    ------
    InvalidJobOrder
        in case of no/too many output products or unexpected product identifier
    """
    if len(output_products_list) == 0:
        raise InvalidJobOrder("No output products specified: exactly one is required.")

    output_products, output_directory, output_baseline = flatten_output_products(output_products_list)

    if len(output_products) > 1:
        raise InvalidJobOrder("Too many output products specified: just one is requested.")

    output_product = output_products[0]
    if output_product not in valid_products:
        raise InvalidJobOrder(f"Unexpected output product identifier: {output_product}")

    return Path(output_directory), output_product, output_baseline


def retrieve_tile_processing_parameters(
    metadata_parameters: list[joborder_models.ParameterType],
    tile_id_param: str,
) -> TileProcessingParameters:
    """Retrieve tile processing parameters containing a single tile_id parameter.

    Parameters
    ----------
    metadata_parameters : list[joborder_models.ParameterType]
        list of processing parameters
    tile_id_param : str
        the parameter name for the tile id

    Returns
    -------
    TileProcessingParameters
        struct containing the processing parameters

    Raises
    ------
    InvalidJobOrder
        if unexpected parameter identifier is found
    """
    parameters = flatten_processing_params(metadata_parameters)

    for param_id in parameters:
        if param_id != tile_id_param:
            raise InvalidJobOrder(f"Unexpected input processing parameter identifier: {param_id}")

    result = TileProcessingParameters()
    tile_id = parameters.pop(tile_id_param, None)
    if tile_id:
        result.tile_id = tile_id

    assert len(parameters) == 0
    return result


def retrieve_optional_configuration_file(
    configuration_files_list: list[joborder_models.CfgFileType],
    conf_key: str,
) -> Path | None:
    """Retrieve a single optional configuration file from the section.

    Parameters
    ----------
    configuration_files_list : list[joborder_models.CfgFileType]
        list of configuration files tags
    conf_key : str
        the configuration file identifier to retrieve

    Returns
    -------
    Path | None
        the configuration file path, or None if not present

    Raises
    ------
    InvalidJobOrder
        if unexpected configuration file identifier is found
    """
    configuration_files = flatten_configuration_file(configuration_files_list)

    for conf_files_id in configuration_files:
        if conf_files_id != conf_key:
            raise InvalidJobOrder(f"Unexpected configuration file identifier: {conf_files_id}")

    conf_raw = configuration_files.pop(conf_key, None)
    assert len(configuration_files) == 0
    return Path(conf_raw) if conf_raw is not None else None


def flatten_intermediate_outputs(
    intermediate_files: list[joborder_models.JoIntermediateOutputType],
) -> dict[str, str]:
    """Convert intermediate files section to a dictionary

    Parameters
    ----------
    intermediate_files : List[joborder_models.JoIntermediateOutputType]
        list of intermediate files tags

    Returns
    -------
    Dict[str, str]
        id to string map

    Raises
    ------
    InvalidJobOrder
        in case of duplicated intermediate files entry
    """
    files: dict[str, str] = {}
    for file in intermediate_files:
        if file.intermediate_output_id.value in files:
            raise InvalidJobOrder(f"Duplicated {file.intermediate_output_id} intermediate file entry")

        files[file.intermediate_output_id.value] = file.intermediate_output_file

    return files


JOB_ORDER_LOG_LEVEL_TO_LOGGING_LEVELS = {
    ProcessorConfiguration.LogLevel.ERROR: logging.ERROR,
    ProcessorConfiguration.LogLevel.WARNING: logging.WARNING,
    ProcessorConfiguration.LogLevel.PROGRESS: logging.INFO,
    ProcessorConfiguration.LogLevel.INFO: logging.INFO,
    ProcessorConfiguration.LogLevel.DEBUG: logging.DEBUG,
}


def translate_logger_level(log_level: ProcessorConfiguration.LogLevel) -> int:
    return JOB_ORDER_LOG_LEVEL_TO_LOGGING_LEVELS[log_level]


def get_bps_logger_level(
    stdout_log_level: ProcessorConfiguration.LogLevel,
    stderr_log_level: ProcessorConfiguration.LogLevel,
) -> int:
    stdout_level = translate_logger_level(stdout_log_level)
    stderr_level = translate_logger_level(stderr_log_level)
    return max(stdout_level, stderr_level)
