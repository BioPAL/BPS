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
from bps.common.joborder import DeviceResources, ProcessorConfiguration


class InvalidJobOrder(ValueError):
    """Raised when failing to translate a joborder"""


def retrieve_configuration_params(
    configuration: joborder_models.ProcessorConfigurationType,
    expected_processor_name: str,
    expected_processor_version: str,
) -> ProcessorConfiguration:
    """Retrieve configuration parameters from the processor configuration section

    Parameters
    ----------
    configuration : joborder_models.ProcessorConfigurationType
        processor configuration section
    expected_processor_name : str
        expected name of processor
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
    assert configuration.file_class
    file_class = configuration.file_class

    assert configuration.processor_name
    if configuration.processor_name.value != expected_processor_name:
        raise InvalidJobOrder(
            f"Invalid processor name: {configuration.processor_name.value} != {expected_processor_name}"
        )

    assert configuration.processor_version
    if configuration.processor_version.value != expected_processor_version:
        raise InvalidJobOrder(
            f"Invalid processor version: {configuration.processor_version.value} != {expected_processor_version}"
        )

    assert configuration.list_of_stdout_log_levels
    if len(configuration.list_of_stdout_log_levels.stdout_log_level) != 1:
        raise InvalidJobOrder("Unexpected number of stdout log level")
    stdout_log_level = configuration.list_of_stdout_log_levels.stdout_log_level[0]

    assert configuration.list_of_stderr_log_levels
    if len(configuration.list_of_stderr_log_levels.stderr_log_level) != 1:
        raise InvalidJobOrder("Unexpected number of stderr log level")
    stderr_log_level = configuration.list_of_stderr_log_levels.stderr_log_level[0]

    assert configuration.intermediate_output_enable is not None
    intermediate_output_enabled = configuration.intermediate_output_enable

    assert configuration.request
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

    # Ignored tags
    assert configuration.processing_node is not None
    assert configuration.processing_station is not None

    return ProcessorConfiguration(
        file_class=file_class.value,
        stdout_log_level=ProcessorConfiguration.LogLevel(stdout_log_level.value),
        stderr_log_level=ProcessorConfiguration.LogLevel(stderr_log_level.value),
        keep_intermediate=intermediate_output_enabled,
        azimuth_interval=azimuth_interval,
    )


def retrieve_configuration_params_l2(
    configuration: joborder_models.ProcessorConfigurationType,
    expected_processor_names: list[str],
    expected_processor_version: str,
) -> ProcessorConfiguration:
    """Retrieve configuration parameters from the processor configuration section, l2a only

    Parameters
    ----------
    configuration : joborder_models.JobOrder.ProcessorConfiguration
        processor configuration section
    expected_processor_names : List[str]
        list of all possible expected names of L2a processors
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
    assert configuration.file_class
    file_class = configuration.file_class

    assert configuration.processor_name
    if configuration.processor_name.value not in expected_processor_names:
        raise InvalidJobOrder(
            f"Invalid processor name: {configuration.processor_name.value} not in {expected_processor_names}"
        )

    assert configuration.processor_version
    if configuration.processor_version.value != expected_processor_version:
        raise InvalidJobOrder(
            f"Invalid processor version: {configuration.processor_version.value} != {expected_processor_version}"
        )

    assert configuration.list_of_stdout_log_levels
    if len(configuration.list_of_stdout_log_levels.stdout_log_level) != 1:
        raise InvalidJobOrder("Unexpected number of stdout log level")
    stdout_log_level = configuration.list_of_stdout_log_levels.stdout_log_level[0]

    assert configuration.list_of_stderr_log_levels
    if len(configuration.list_of_stderr_log_levels.stderr_log_level) != 1:
        raise InvalidJobOrder("Unexpected number of stderr log level")
    stderr_log_level = configuration.list_of_stderr_log_levels.stderr_log_level[0]

    assert configuration.intermediate_output_enable is not None
    intermediate_output_enabled = configuration.intermediate_output_enable

    assert configuration.request
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

    # Ignored tags
    assert configuration.processing_node is not None
    assert configuration.processing_station is not None

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
    assert job_order.list_of_tasks is not None
    if len(job_order.list_of_tasks.task) != 1:
        raise InvalidJobOrder("Unexpected number of tasks")

    task = job_order.list_of_tasks.task[0]

    assert task.task_name is not None
    if task.task_name.value != expected_task_name:
        raise InvalidJobOrder(f"Invalid task name: {task.task_name.value} != {expected_task_name}")

    assert task.task_version is not None
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
    assert task.number_of_cpu_cores is not None and task.amount_of_ram is not None and task.disk_space is not None

    ramdisk_amount = None
    ramdisk_mount_point = None
    if task.list_of_ramdisks is not None:
        if len(task.list_of_ramdisks.ramdisk) > 1:
            raise RuntimeError("More than one ramdisk found in the JobOrder")

        if len(task.list_of_ramdisks.ramdisk) == 1:
            ramdisk = task.list_of_ramdisks.ramdisk[0]
            assert ramdisk.amount is not None
            assert ramdisk.mount_path is not None
            ramdisk_amount = ramdisk.amount
            ramdisk_mount_point = Path(ramdisk.mount_path)

    assert task.disk_space.value is not None
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
        assert parameter.name is not None
        assert parameter.value is not None
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
        assert file.cfg_id is not None
        assert file.cfg_file_name is not None
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
        assert input_product.list_of_selected_inputs is not None
        assert len(input_product.list_of_selected_inputs.selected_input) == 1
        selected_input = input_product.list_of_selected_inputs.selected_input[0]

        assert selected_input.list_of_file_names is not None
        assert len(selected_input.list_of_file_names.file_name) >= 1
        assert selected_input.file_type is not None
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
        assert output_product.file_type is not None
        assert output_product.file_dir is not None
        if output_product.file_type in outputs:
            raise InvalidJobOrder(f"Duplicated {output_product.file_type} output file entry")

        if output_directory:
            if output_product.file_dir.value != output_directory:
                raise InvalidJobOrder("Multiple output directories specified")
        else:
            output_directory = output_product.file_dir.value

        assert output_product.baseline is not None
        baseline = int(output_product.baseline.value)
        if output_baseline:
            if baseline != output_baseline:
                raise InvalidJobOrder("Multiple product baselines specified")
        else:
            output_baseline = baseline

        outputs.append(output_product.file_type.value)

        # unused
        assert output_product.file_name_pattern is not None

    assert output_directory is not None
    assert output_baseline is not None
    return outputs, output_directory, output_baseline


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
        assert file.intermediate_output_id is not None
        assert file.intermediate_output_file is not None
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
