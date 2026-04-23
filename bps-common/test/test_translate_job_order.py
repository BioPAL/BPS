# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Unit Tests for translate_job_order module
------------------------------------------
"""

import logging
import unittest

from bps.common import Swath
from bps.common.io.joborder_models import (
    AlternativeId,
    Baseline,
    CfgFileType,
    CfgId,
    DiskSpace,
    FileDir,
    FileName,
    FileNamePattern,
    FileType,
    InputId,
    IntermediateOutputId,
    JobOrder,
    JobOrderType,
    JoInputType,
    JoIntermediateOutputType,
    JoOutputType,
    JoTaskType,
    ListOfStderrLogLevelsStderrLogLevel,
    ListOfStdoutLogLevelsStdoutLogLevel,
    ParameterType,
    ProcessorConfigurationType,
    ProcessorName,
    ProcessorVersion,
    RamdiskType,
    RequestType,
    SelectedInputType,
    TaskName,
    TaskVersion,
    ToiType,
)
from bps.common.joborder import ProcessorConfiguration
from bps.common.translate_job_order import (
    InvalidJobOrder,
    flatten_configuration_file,
    flatten_input_products,
    flatten_input_products_allow_multiple_products,
    flatten_intermediate_outputs,
    flatten_output_products,
    flatten_processing_params,
    get_bps_logger_level,
    retrieve_configuration_params,
    retrieve_device_resources,
    retrieve_single_input_products,
    retrieve_swath_from_products_identifiers,
    retrieve_task,
    translate_logger_level,
)
from xsdata.models.datatype import XmlDateTime

_PROC_NAME = "MY_PROC"
_PROC_VERSION = "01.00"
_TASK_NAME = "MY_TASK"
_TASK_VERSION = "01.00"


def _make_proc_config(
    processor_name=_PROC_NAME,
    processor_version=_PROC_VERSION,
    stdout_level=ListOfStdoutLogLevelsStdoutLogLevel.INFO,
    stderr_level=ListOfStderrLogLevelsStderrLogLevel.ERROR,
    toi=None,
    keep_intermediate=False,
    stdout_levels=None,
    stderr_levels=None,
):
    if stdout_levels is None:
        stdout_levels = [stdout_level]
    if stderr_levels is None:
        stderr_levels = [stderr_level]
    return ProcessorConfigurationType(
        file_class=FileType(value="OPER"),
        processor_name=ProcessorName(value=processor_name),
        processor_version=ProcessorVersion(value=processor_version),
        processing_node="node",
        list_of_stdout_log_levels=ProcessorConfigurationType.ListOfStdoutLogLevels(stdout_log_level=stdout_levels),
        list_of_stderr_log_levels=ProcessorConfigurationType.ListOfStderrLogLevels(stderr_log_level=stderr_levels),
        intermediate_output_enable=keep_intermediate,
        processing_station="station",
        request=RequestType(toi=toi),
    )


def _make_input(file_type_value="INPUT_TYPE", file_path="/path/to/file"):
    return JoInputType(
        input_id=InputId(value="IN_ID"),
        alternative_id=AlternativeId(value=""),
        list_of_selected_inputs=JoInputType.ListOfSelectedInputs(
            selected_input=[
                SelectedInputType(
                    file_type=FileType(value=file_type_value),
                    list_of_file_names=SelectedInputType.ListOfFileNames(file_name=[FileName(value=file_path)]),
                )
            ]
        ),
    )


def _make_output(file_type_value="OUTPUT_TYPE", file_dir="/out", baseline="01"):
    return JoOutputType(
        file_type=FileType(value=file_type_value),
        file_name_pattern=FileNamePattern(value="*"),
        file_dir=FileDir(value=file_dir),
        baseline=Baseline(value=baseline),
    )


def _make_task(task_name=_TASK_NAME, task_version=_TASK_VERSION, ramdisks=None):
    return JoTaskType(
        task_name=TaskName(value=task_name),
        task_version=TaskVersion(value=task_version),
        number_of_cpu_cores=4,
        amount_of_ram=8000,
        disk_space=DiskSpace(value=10000),
        list_of_ramdisks=ramdisks,
        list_of_proc_parameters=JoTaskType.ListOfProcParameters(),
        list_of_cfg_files=JoTaskType.ListOfCfgFiles(),
        list_of_inputs=JoTaskType.ListOfInputs(input=[_make_input()]),
        list_of_outputs=JoTaskType.ListOfOutputs(output=[_make_output()]),
        list_of_intermediate_outputs=JoTaskType.ListOfIntermediateOutputs(),
    )


def _make_job_order(task=None, proc_config=None):
    if task is None:
        task = _make_task()
    if proc_config is None:
        proc_config = _make_proc_config()
    return JobOrder(
        processor_configuration=proc_config,
        list_of_tasks=JobOrderType.ListOfTasks(task=[task]),
    )


class TestRetrieveConfigurationParams(unittest.TestCase):
    def test_happy_path(self):
        cfg = _make_proc_config()
        result = retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)
        self.assertIsInstance(result, ProcessorConfiguration)
        self.assertEqual(result.file_class, "OPER")
        self.assertEqual(result.stdout_log_level, ProcessorConfiguration.LogLevel.INFO)
        self.assertEqual(result.stderr_log_level, ProcessorConfiguration.LogLevel.ERROR)
        self.assertFalse(result.keep_intermediate)
        self.assertIsNone(result.azimuth_interval)

    def test_keep_intermediate_true(self):
        cfg = _make_proc_config(keep_intermediate=True)
        result = retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)
        self.assertTrue(result.keep_intermediate)

    def test_with_toi(self):
        toi = ToiType(
            start=XmlDateTime(2020, 1, 1, 0, 0, 0),
            stop=XmlDateTime(2020, 12, 31, 23, 59, 59),
        )
        cfg = _make_proc_config(toi=toi)
        result = retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)
        self.assertIsNotNone(result.azimuth_interval)

    def test_toi_partial_raises(self):
        # Setting stop as empty string causes str(stop) == "" which is falsy,
        # triggering the elif branch.
        toi = ToiType(start="2020-01-01T00:00:00", stop="")  # type: ignore[arg-type]
        cfg = _make_proc_config(toi=toi)
        with self.assertRaises(InvalidJobOrder):
            retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)

    def test_wrong_processor_name_raises(self):
        cfg = _make_proc_config(processor_name="WRONG")
        with self.assertRaises(InvalidJobOrder):
            retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)

    def test_wrong_processor_version_raises(self):
        cfg = _make_proc_config(processor_version="02.00")
        with self.assertRaises(InvalidJobOrder):
            retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)

    def test_multiple_stdout_levels_raises(self):
        cfg = _make_proc_config(
            stdout_levels=[
                ListOfStdoutLogLevelsStdoutLogLevel.INFO,
                ListOfStdoutLogLevelsStdoutLogLevel.DEBUG,
            ]
        )
        with self.assertRaises(InvalidJobOrder):
            retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)

    def test_multiple_stderr_levels_raises(self):
        cfg = _make_proc_config(
            stderr_levels=[
                ListOfStderrLogLevelsStderrLogLevel.ERROR,
                ListOfStderrLogLevelsStderrLogLevel.WARNING,
            ]
        )
        with self.assertRaises(InvalidJobOrder):
            retrieve_configuration_params(cfg, _PROC_NAME, _PROC_VERSION)

    def test_list_of_names_first_match(self):
        cfg = _make_proc_config()
        result = retrieve_configuration_params(cfg, [_PROC_NAME, "ALT_PROC"], _PROC_VERSION)
        self.assertIsInstance(result, ProcessorConfiguration)

    def test_list_of_names_second_match(self):
        cfg = _make_proc_config(processor_name="ALT_PROC")
        result = retrieve_configuration_params(cfg, [_PROC_NAME, "ALT_PROC"], _PROC_VERSION)
        self.assertIsInstance(result, ProcessorConfiguration)

    def test_list_of_names_no_match_raises(self):
        cfg = _make_proc_config(processor_name="UNKNOWN")
        with self.assertRaises(InvalidJobOrder):
            retrieve_configuration_params(cfg, [_PROC_NAME, "ALT_PROC"], _PROC_VERSION)


class TestRetrieveTask(unittest.TestCase):
    def test_happy_path(self):
        jo = _make_job_order()
        task = retrieve_task(jo, _TASK_NAME, _TASK_VERSION)
        self.assertEqual(task.task_name.value, _TASK_NAME)

    def test_multiple_tasks_raises(self):
        jo = JobOrder(
            processor_configuration=_make_proc_config(),
            list_of_tasks=JobOrderType.ListOfTasks(task=[_make_task(), _make_task()]),
        )
        with self.assertRaises(InvalidJobOrder):
            retrieve_task(jo, _TASK_NAME, _TASK_VERSION)

    def test_wrong_task_name_raises(self):
        jo = _make_job_order()
        with self.assertRaises(InvalidJobOrder):
            retrieve_task(jo, "WRONG_TASK", _TASK_VERSION)

    def test_wrong_task_version_raises(self):
        jo = _make_job_order()
        with self.assertRaises(InvalidJobOrder):
            retrieve_task(jo, _TASK_NAME, "02.00")


class TestRetrieveDeviceResources(unittest.TestCase):
    def test_no_ramdisk(self):
        task = _make_task()
        result = retrieve_device_resources(task)
        self.assertEqual(result.num_threads, 4)
        self.assertEqual(result.available_ram, 8000)
        self.assertEqual(result.available_space, 10000)
        self.assertIsNone(result.ramdisk_size)
        self.assertIsNone(result.ramdisk_mount_point)

    def test_with_ramdisk(self):
        ramdisks = JoTaskType.ListOfRamdisks(ramdisk=[RamdiskType(amount=4096, mount_path="/mnt/ramdisk")])
        task = _make_task(ramdisks=ramdisks)
        result = retrieve_device_resources(task)
        self.assertEqual(result.ramdisk_size, 4096)
        self.assertIsNotNone(result.ramdisk_mount_point)

    def test_multiple_ramdisks_raises(self):
        ramdisks = JoTaskType.ListOfRamdisks(
            ramdisk=[
                RamdiskType(amount=4096, mount_path="/mnt/ramdisk1"),
                RamdiskType(amount=2048, mount_path="/mnt/ramdisk2"),
            ]
        )
        task = _make_task(ramdisks=ramdisks)
        with self.assertRaises(RuntimeError):
            retrieve_device_resources(task)

    def test_empty_ramdisk_list(self):
        ramdisks = JoTaskType.ListOfRamdisks(ramdisk=[])
        task = _make_task(ramdisks=ramdisks)
        result = retrieve_device_resources(task)
        self.assertIsNone(result.ramdisk_size)


class TestRetrieveSwathFromProductsIdentifiers(unittest.TestCase):
    _map = {
        Swath.S1: "PROD_S1",
        Swath.S2: "PROD_S2",
        Swath.S3: "PROD_S3",
    }

    def test_single_match(self):
        result = retrieve_swath_from_products_identifiers(["PROD_S1"], self._map)
        self.assertEqual(result, Swath.S1)

    def test_no_match_returns_none(self):
        result = retrieve_swath_from_products_identifiers(["PROD_RO"], self._map)
        self.assertIsNone(result)

    def test_multiple_matches_raises(self):
        with self.assertRaises(InvalidJobOrder):
            retrieve_swath_from_products_identifiers(["PROD_S1", "PROD_S2"], self._map)


class TestFlattenProcessingParams(unittest.TestCase):
    def test_empty(self):
        result = flatten_processing_params([])
        self.assertEqual(result, {})

    def test_happy_path(self):
        params = [
            ParameterType(name="KEY1", value="VAL1"),
            ParameterType(name="KEY2", value="VAL2"),
        ]
        result = flatten_processing_params(params)
        self.assertEqual(result, {"KEY1": "VAL1", "KEY2": "VAL2"})

    def test_duplicate_raises(self):
        params = [
            ParameterType(name="KEY1", value="VAL1"),
            ParameterType(name="KEY1", value="VAL2"),
        ]
        with self.assertRaises(InvalidJobOrder):
            flatten_processing_params(params)


class TestFlattenConfigurationFile(unittest.TestCase):
    def test_happy_path(self):
        files = [
            CfgFileType(cfg_id=CfgId(value="CFG1"), cfg_file_name="/path/a.xml"),
            CfgFileType(cfg_id=CfgId(value="CFG2"), cfg_file_name="/path/b.xml"),
        ]
        result = flatten_configuration_file(files)
        self.assertEqual(result, {"CFG1": "/path/a.xml", "CFG2": "/path/b.xml"})

    def test_duplicate_raises(self):
        files = [
            CfgFileType(cfg_id=CfgId(value="CFG1"), cfg_file_name="/path/a.xml"),
            CfgFileType(cfg_id=CfgId(value="CFG1"), cfg_file_name="/path/b.xml"),
        ]
        with self.assertRaises(InvalidJobOrder):
            flatten_configuration_file(files)


class TestRetrieveSingleInputProducts(unittest.TestCase):
    def test_happy_path(self):
        inputs = {"TYPE_A": ["/path/a"], "TYPE_B": ["/path/b"]}
        result = retrieve_single_input_products(inputs)
        self.assertEqual(result, {"TYPE_A": "/path/a", "TYPE_B": "/path/b"})

    def test_multiple_products_raises(self):
        inputs = {"TYPE_A": ["/path/a1", "/path/a2"]}
        with self.assertRaises(InvalidJobOrder):
            retrieve_single_input_products(inputs)


class TestFlattenInputProducts(unittest.TestCase):
    def test_happy_path(self):
        inputs = [_make_input("TYPE_A", "/path/a")]
        result = flatten_input_products(inputs)
        self.assertEqual(result, {"TYPE_A": "/path/a"})

    def test_duplicate_raises(self):
        inputs = [_make_input("TYPE_A", "/path/a"), _make_input("TYPE_A", "/path/b")]
        with self.assertRaises(InvalidJobOrder):
            flatten_input_products(inputs)


class TestFlattenInputProductsAllowMultiple(unittest.TestCase):
    def test_single_file(self):
        inputs = [_make_input("TYPE_A", "/path/a")]
        result = flatten_input_products_allow_multiple_products(inputs)
        self.assertEqual(result, {"TYPE_A": ["/path/a"]})

    def test_multiple_files(self):
        jo_input = JoInputType(
            input_id=InputId(value="IN_ID"),
            alternative_id=AlternativeId(value=""),
            list_of_selected_inputs=JoInputType.ListOfSelectedInputs(
                selected_input=[
                    SelectedInputType(
                        file_type=FileType(value="TYPE_A"),
                        list_of_file_names=SelectedInputType.ListOfFileNames(
                            file_name=[FileName(value="/path/a1"), FileName(value="/path/a2")]
                        ),
                    )
                ]
            ),
        )
        result = flatten_input_products_allow_multiple_products([jo_input])
        self.assertEqual(result, {"TYPE_A": ["/path/a1", "/path/a2"]})


class TestFlattenOutputProducts(unittest.TestCase):
    def test_happy_path(self):
        outputs = [_make_output("TYPE_A", "/out", "01")]
        result_types, result_dir, result_baseline = flatten_output_products(outputs)
        self.assertEqual(result_types, ["TYPE_A"])
        self.assertEqual(result_dir, "/out")
        self.assertEqual(result_baseline, 1)

    def test_multiple_outputs_same_dir_and_baseline(self):
        outputs = [
            _make_output("TYPE_A", "/out", "01"),
            _make_output("TYPE_B", "/out", "01"),
        ]
        result_types, result_dir, result_baseline = flatten_output_products(outputs)
        self.assertIn("TYPE_A", result_types)
        self.assertIn("TYPE_B", result_types)
        self.assertEqual(result_dir, "/out")

    def test_multiple_output_dirs_raises(self):
        outputs = [
            _make_output("TYPE_A", "/out1", "01"),
            _make_output("TYPE_B", "/out2", "01"),
        ]
        with self.assertRaises(InvalidJobOrder):
            flatten_output_products(outputs)

    def test_multiple_baselines_raises(self):
        outputs = [
            _make_output("TYPE_A", "/out", "01"),
            _make_output("TYPE_B", "/out", "02"),
        ]
        with self.assertRaises(InvalidJobOrder):
            flatten_output_products(outputs)

    def test_duplicate_file_type_raises(self):
        outputs = [
            _make_output("TYPE_A", "/out", "01"),
            _make_output("TYPE_A", "/out", "01"),
        ]
        with self.assertRaises(InvalidJobOrder):
            flatten_output_products(outputs)


class TestFlattenIntermediateOutputs(unittest.TestCase):
    def test_empty(self):
        result = flatten_intermediate_outputs([])
        self.assertEqual(result, {})

    def test_happy_path(self):
        files = [
            JoIntermediateOutputType(
                intermediate_output_id=IntermediateOutputId(value="ID1"),
                intermediate_output_file="/path/out1",
            ),
            JoIntermediateOutputType(
                intermediate_output_id=IntermediateOutputId(value="ID2"),
                intermediate_output_file="/path/out2",
            ),
        ]
        result = flatten_intermediate_outputs(files)
        self.assertEqual(result, {"ID1": "/path/out1", "ID2": "/path/out2"})

    def test_duplicate_raises(self):
        files = [
            JoIntermediateOutputType(
                intermediate_output_id=IntermediateOutputId(value="ID1"),
                intermediate_output_file="/path/out1",
            ),
            JoIntermediateOutputType(
                intermediate_output_id=IntermediateOutputId(value="ID1"),
                intermediate_output_file="/path/out2",
            ),
        ]
        with self.assertRaises(InvalidJobOrder):
            flatten_intermediate_outputs(files)


class TestTranslateLoggerLevel(unittest.TestCase):
    def test_all_levels(self):
        self.assertEqual(translate_logger_level(ProcessorConfiguration.LogLevel.ERROR), logging.ERROR)
        self.assertEqual(translate_logger_level(ProcessorConfiguration.LogLevel.WARNING), logging.WARNING)
        self.assertEqual(translate_logger_level(ProcessorConfiguration.LogLevel.PROGRESS), logging.INFO)
        self.assertEqual(translate_logger_level(ProcessorConfiguration.LogLevel.INFO), logging.INFO)
        self.assertEqual(translate_logger_level(ProcessorConfiguration.LogLevel.DEBUG), logging.DEBUG)


class TestGetBpsLoggerLevel(unittest.TestCase):
    def test_returns_max(self):
        result = get_bps_logger_level(
            ProcessorConfiguration.LogLevel.DEBUG,
            ProcessorConfiguration.LogLevel.ERROR,
        )
        self.assertEqual(result, logging.ERROR)

    def test_both_same(self):
        result = get_bps_logger_level(
            ProcessorConfiguration.LogLevel.INFO,
            ProcessorConfiguration.LogLevel.INFO,
        )
        self.assertEqual(result, logging.INFO)


if __name__ == "__main__":
    unittest.main()
