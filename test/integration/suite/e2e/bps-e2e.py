# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Biomass end to end test
-----------------------
"""

# Import the aux-pps utilities.
# fmt: off
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "stack" / "aux_pps_utils"))
# fmt: on
import os
from typing import Literal

import arepyextras.xml as xml
from arepyextras.test import DataRepository, Environment, TestSession
from aux_pps_utils import initialize_aux_pps
from bps.common import __version__, retrieve_aux_product_data_single_content
from bps.common.bps_logger import get_version_in_logger_format
from bps.transcoder.sarproduct.biomass_l1product_reader import BIOMASSL1ProductReader
from bps.transcoder.sarproduct.biomass_l2aproduct_reader import BIOMASSL2aProductReader
from bps.transcoder.sarproduct.biomass_l2bfdproduct_reader import (
    BIOMASSL2bFDProductReader,
)
from bps.transcoder.sarproduct.biomass_l2bfhproduct_reader import (
    BIOMASSL2bFHProductReader,
)
from bps.transcoder.sarproduct.biomass_stackproduct_reader import (
    BIOMASSStackProductReader,
)

CURRENT_VERSION = get_version_in_logger_format(__version__)
WORKER_THREADS = 7
MAX_RAM_USAGE = 64768
L1_OUT_DIR = "output_dir_l1"
STA_OUT_DIR = "output_dir_sta"
L2A_OUT_DIR = "output_dir_l2a"
L2B_FH_OUT_DIR = "output_dir_l2b_fh"
L2B_FD_OUT_DIR = "output_dir_l2b_fd"

L1A_PRODUCTS_COUNT = 1
L1C_PRODUCTS_COUNT = 2
L2_PRODUCTS_COUNT = 1

L1_TOI_START = "2017-01-11T05:06:14.232204000000"
L1_TOI_STOP = "2017-01-11T05:06:24.425897302191"
BLOCK_OVERLAP = 4076


def test_001_l1(session: TestSession, env: Environment, data: DataRepository):
    """Run L1 Processing using ROI."""
    # Setup the job order and AUX-PP1.

    job_order_path = data.pull("input/l1002/joborder")
    _set_l1_configuration(
        env=env,
        data=data,
        aux_pp1_path=data.pull("configurations/l1p/CONF-BPS-PP1-003"),
        job_order_path=job_order_path,
        output_directory=env.root.parent.joinpath(L1_OUT_DIR).absolute(),
    )

    # Run the processor and check exit code.
    l1_proc = env.run("bps_l1_processor", job_order_path)
    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)

    # Count L1a M products. Raise if amount found is not as expected.
    _check_products(
        root_dir=env.root.parent.joinpath(L1_OUT_DIR),
        pattern="BIO_S*_SCS__1M*",
        expected_num_files=L1A_PRODUCTS_COUNT,
    )

    # Count and read L1a S products. Raise if amount does not match or a product
    # fails to read.
    _check_products(
        root_dir=env.root.parent.joinpath(L1_OUT_DIR),
        pattern="BIO_S*_SCS__1S*",
        expected_num_files=L1A_PRODUCTS_COUNT,
        product_reader=BIOMASSL1ProductReader,
    )


def test_002_sta(session: TestSession, env: Environment, data: DataRepository):
    """Run stack processor."""
    # Setup the job order and AUX-PPS.
    job_order_path = data.pull("input/sta_int/joborder")
    _set_stack_configuration(
        env=env,
        data=data,
        aux_pps_product_dir=data.pull("configurations/sta/CONF-BPS-PPS-001"),
        job_order_path=job_order_path,
        output_directory=env.root.parent.joinpath(L1_OUT_DIR).absolute(),
    )

    # Run the processor and check exit code.
    stack_processor = env.run(
        "bps_stack_processor",
        job_order_path,
    )
    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())

    session.expect_run_successful(stack_processor, fatal=True)

    # Count L1c M products. Raise if amount found is not as expected.
    _check_products(
        root_dir=env.root.parent.joinpath(STA_OUT_DIR),
        pattern="BIO_S*_STA__1M*",
        expected_num_files=L1C_PRODUCTS_COUNT,
    )

    # Count and read L1c S products. Raise if amount does not match or a product
    # fails to read.
    _check_products(
        root_dir=env.root.parent.joinpath(STA_OUT_DIR),
        pattern="BIO_S*_STA__1S*",
        expected_num_files=L1C_PRODUCTS_COUNT,
        product_reader=BIOMASSStackProductReader,
    )


def test_003_l2a(session: TestSession, env: Environment, data: DataRepository):
    """Run L2a processor."""
    # Setup the job order and configuration file.
    job_order_path = data.pull("input/l2a/joborder")
    job_order = xml.Document(job_order_path)

    # Get L1c products. Raise if amount does not match expected.
    l1c_list = _find_products(
        root_dir=env.root.parent.joinpath(STA_OUT_DIR),
        pattern="BIO_S*_STA__1S*",
        expected_num_files=L1C_PRODUCTS_COUNT,
    )

    aux_pp2_path = data.pull("configurations/l2/CONF-BPS-PP2-2A-ALL")

    job_order.set(
        "//List_of_Tasks/Task/Amount_of_RAM/text()",
        MAX_RAM_USAGE,
    )
    job_order.set(
        "//List_of_Tasks/Task/Number_of_CPU_Cores/text()",
        WORKER_THREADS,
    )
    job_order.set(
        "//Input[Input_ID/text()='Sx_STA__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[1]/text()",
        l1c_list[0],
    )
    job_order.set(
        "//Input[Input_ID/text()='Sx_STA__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[2]/text()",
        l1c_list[1],
    )

    # Remove third image from the INT job order.
    job_order.remove(
        "//Input[Input_ID/text()='Sx_STA__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[3]"
    )

    job_order.set("//Output/File_Dir/text()", env.root.parent.joinpath(L2A_OUT_DIR).absolute())
    job_order.set(
        "//Cfg_File_Name/text()",
        data.pull(
            "configurations/l2/AUX_L2_FNF",
            "BIO_AUX_L2_FNF_00000000T000000_00000000T000000_REGEX_SAFE_NAME",
        ),
    )

    job_order.set(
        "//Selected_Input[File_Type/text()='AUX_PP2_2A']/List_of_File_Names/File_Name",
        aux_pp2_path,
    )
    job_order.set("//Processor_Version/text()", CURRENT_VERSION)
    job_order.set("//Task_Version/text()", CURRENT_VERSION)

    # Run the processor and check exit code.
    l2a_processor = env.run(
        "bps_l2a_processor",
        job_order_path,
    )
    if l2a_processor.returncode != 0:
        assert l2a_processor.stderr is not None
        print(l2a_processor.stderr.read_text())

    session.expect_run_successful(l2a_processor, fatal=True)

    # Count and read the L2a-FD products. Raise if amount does not match or a product
    # fails to read.
    _check_products(
        root_dir=env.root.parent.joinpath(L2A_OUT_DIR),
        pattern="BIO_FP_FD__L2A_*",
        expected_num_files=L2_PRODUCTS_COUNT,
        product_reader=BIOMASSL2aProductReader,
    )

    # Count and read the L2a-FH products. Raise if amount does not match or a product
    # fails to read.
    _check_products(
        root_dir=env.root.parent.joinpath(L2A_OUT_DIR),
        pattern="BIO_FP_FH__L2A_*",
        expected_num_files=L2_PRODUCTS_COUNT,
        product_reader=BIOMASSL2aProductReader,
    )

    # Count and read the L2a-GN products. Raise if amount does not match or a product
    # fails to read.
    _check_products(
        root_dir=env.root.parent.joinpath(L2A_OUT_DIR),
        pattern="BIO_FP_GN__L2A_*",
        expected_num_files=L2_PRODUCTS_COUNT,
        product_reader=BIOMASSL2aProductReader,
    )


def test_004_l2b_fh(session: TestSession, env: Environment, data: DataRepository):
    """Run L2b FH processor."""
    # Setup the job order and configuration file.
    job_order_path = data.pull("input/l2b_fh/joborder")
    job_order = xml.Document(job_order_path)

    aux_pp2_fh_path = data.pull("configurations/l2/CONF-BPS-PP2-FH-001")
    aux_pp2_fh = xml.Document(retrieve_aux_product_data_single_content(aux_pp2_fh_path))
    aux_pp2_fh.set("//minimumL2aCoverage/text()", 4.0)

    # Get L2a FH products. Raise if amount does not match expected.
    l2a_fh_list = _find_products(
        root_dir=env.root.parent.joinpath(L2A_OUT_DIR),
        pattern="BIO_FP_FH*",
        expected_num_files=L2_PRODUCTS_COUNT,
    )

    job_order.set(
        "//List_of_Tasks/Task/Amount_of_RAM/text()",
        MAX_RAM_USAGE,
    )
    job_order.set(
        "//List_of_Tasks/Task/Number_of_CPU_Cores/text()",
        WORKER_THREADS,
    )
    job_order.set(
        "//Input[Input_ID/text()='FP_FH__L2A']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        l2a_fh_list[0],
    )

    job_order.set("//Proc_Parameter/Value/text()", _get_l2a_tile_id(l2a_fh_list[0]))

    job_order.set(
        "//Input[Input_ID/text()='AUX_PP2_FH']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        aux_pp2_fh_path,
    )
    job_order.set("//Output/File_Dir/text()", env.root.parent.joinpath(L2B_FH_OUT_DIR).absolute())
    job_order.set("//Processor_Version/text()", CURRENT_VERSION)
    job_order.set("//Task_Version/text()", CURRENT_VERSION)

    # Run the processor and check exit code.
    l2b_fh_processor = env.run(
        "bps_l2b_fh_processor",
        job_order_path,
    )
    if l2b_fh_processor.returncode != 0:
        assert l2b_fh_processor.stderr is not None
        print(l2b_fh_processor.stderr.read_text())

    session.expect_run_successful(l2b_fh_processor, fatal=True)

    # Count and read the L2B-FH products. Raise if amount does not match or a product
    # fails to read.
    _check_products(
        root_dir=env.root.parent.joinpath(L2B_FH_OUT_DIR),
        pattern="BIO_FP_FH__L2B_*",
        expected_num_files=L2_PRODUCTS_COUNT,
        product_reader=BIOMASSL2bFHProductReader,
    )


def test_005_l2b_fd(session: TestSession, env: Environment, data: DataRepository):
    """Run L2b FD processor."""
    # Setup the job order and configuration file.
    job_order_path = data.pull("input/l2b_fd/joborder")
    job_order = xml.Document(job_order_path)

    aux_pp2_fd_path = data.pull("configurations/l2/CONF-BPS-PP2-FD-001")
    aux_pp2_fd = xml.Document(retrieve_aux_product_data_single_content(aux_pp2_fd_path))
    aux_pp2_fd.set("//minimumL2aCoverage/text()", 4.0)

    # Get L2a FD products. Raise if amount does not match expected.
    l2a_fd_list = _find_products(
        root_dir=env.root.parent.joinpath(L2A_OUT_DIR),
        pattern="BIO_FP_FD*",
        expected_num_files=L2_PRODUCTS_COUNT,
    )

    job_order.set(
        "//List_of_Tasks/Task/Amount_of_RAM/text()",
        MAX_RAM_USAGE,
    )
    job_order.set(
        "//List_of_Tasks/Task/Number_of_CPU_Cores/text()",
        WORKER_THREADS,
    )
    job_order.set(
        "//Input[Input_ID/text()='FP_FD__L2A']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        l2a_fd_list[0],
    )
    job_order.set("//Proc_Parameter/Value/text()", _get_l2a_tile_id(l2a_fd_list[0]))
    job_order.set(
        "//Input[Input_ID/text()='AUX_PP2_FD']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        aux_pp2_fd_path,
    )
    job_order.set("//Output/File_Dir/text()", env.root.parent.joinpath(L2B_FD_OUT_DIR).absolute())
    job_order.set("//Processor_Version/text()", CURRENT_VERSION)
    job_order.set("//Task_Version/text()", CURRENT_VERSION)

    # Run the processor and check exit code.
    l2b_fd_processor = env.run(
        "bps_l2b_fd_processor",
        job_order_path,
    )
    if l2b_fd_processor.returncode != 0:
        assert l2b_fd_processor.stderr is not None
        print(l2b_fd_processor.stderr.read_text())

    session.expect_run_successful(l2b_fd_processor, fatal=True)

    # Count and read the L2B-FD products. Raise if amount does not match or a product
    # fails to read.
    _check_products(
        root_dir=env.root.parent.joinpath(L2B_FD_OUT_DIR),
        pattern="BIO_FP_FD__L2B_*",
        expected_num_files=L2_PRODUCTS_COUNT,
        product_reader=BIOMASSL2bFDProductReader,
    )


def _set_l1_configuration(
    *,
    env: Environment,
    data: DataRepository,
    aux_pp1_path: Path,
    job_order_path: Path,
    output_directory: Path,
):
    """Setup for the L1 tests."""
    job_order = xml.Document(job_order_path)
    aux_pp1_xml_path = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_xml_path)

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    dem_dir, dem_version = _dem_setup(env)

    _update_l1_job_order(
        job_order,
        TOI_Start=L1_TOI_START,
        TOI_Stop=L1_TOI_STOP,
        Amount_of_RAM=MAX_RAM_USAGE,
        Number_of_CPU_Cores=WORKER_THREADS,
        DEM=str(dem_dir),
        S2_RAW__0S=data.pull("product/l1002/l0/S2_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1002/l0/S2_RAW__0M"),
        AUX_ORB=data.pull("product/l1002/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1002/aux_files/AUX_ATT"),
        GMF=data.pull("internal_resources/GMF_14"),
        AUX_INS=aux_ins_path,
        AUX_PP1=aux_pp1_path,
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        OUTPUT_DIR=output_directory,
    )
    _update_aux_pp1(
        aux_pp1=aux_pp1,
        raw_data_correction_flag="false",
        bias_correction_flag="false",
        antenna_pattern_correction1_flag="false",
        antenna_pattern_correction2_flag="false",
        polarimetric_correction_flag="false",
        faraday_rotation_correction_flag="false",
        ionospheric_phase_screen_correction_flag="false",
        group_delay_correction_flag="false",
        azimuth_block_overlap_lines=BLOCK_OVERLAP,
        dem_version=dem_version,
        lut_range_decimation_factor=2,
        lut_azimuth_decimation_factor=12,
        dc_method="Geometry",
    )

    aux_pp1_xml_path = retrieve_aux_product_data_single_content(aux_pp1_path)

    job_order.set(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[Name/text()='frame_status']/Value/text()",
        "PARTIAL",
    )
    job_order.add("//Task/List_of_Proc_Parameters", "Proc_Parameter", "")
    job_order.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Name",
        "range_start_time",
    )
    job_order.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Value",
        "0.005033364936743259",
    )
    job_order.add("//Task/List_of_Proc_Parameters", "Proc_Parameter", "")
    job_order.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Name",
        "range_stop_time",
    )
    job_order.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Value",
        "0.005185799419326667",
    )


def _dem_setup(env: Environment) -> tuple[Path, str]:
    """Setup dem in test enviroment."""
    copernicus_dem_folder = env.getenv("COPERNICUS_DEM_FOLDER")
    if copernicus_dem_folder is None:
        raise RuntimeError("COPERNICUS_DEM_FOLDER not found in testing environment toml file.")
    assert isinstance(copernicus_dem_folder, str)

    copernicus_version = Path(copernicus_dem_folder).name
    dem_dir = Path("DEM").absolute()
    copernicus_dir = dem_dir.joinpath("copernicus")
    copernicus_dir.mkdir(parents=True, exist_ok=True)
    link = copernicus_dir.joinpath(copernicus_version)
    os.symlink(src=copernicus_dem_folder, dst=link, target_is_directory=True)

    return dem_dir, copernicus_version


def _update_l1_job_order(
    job_order: xml.Document,
    *,
    Intermediate_Output_Enable: bool = False,
    TOI_Start: str | None = None,
    TOI_Stop: str | None = None,
    Amount_of_RAM: int | None = None,
    Number_of_CPU_Cores: int | None = None,
    DEM: Path | str | None = None,
    S1_RAW__0S: Path | str | None = None,
    S2_RAW__0S: Path | str | None = None,
    S3_RAW__0S: Path | str | None = None,
    S1_RAW__0M: Path | str | None = None,
    S2_RAW__0M: Path | str | None = None,
    S3_RAW__0M: Path | str | None = None,
    AUX_ORB: Path | str | None = None,
    AUX_ATT: Path | str | None = None,
    GMF: Path | str | None = None,
    AUX_TEC: Path | str | None = None,
    AUX_INS: Path | str | None = None,
    AUX_PP1: Path | str | None = None,
    OUTPUT_DIR: Path | str | None = None,
    IntermediateDataDir: Path | str | None = None,
):
    """Update the L1 job order file with new parameters."""
    if Intermediate_Output_Enable is not None:
        job_order.set(
            "//Processor_Configuration/Intermediate_Output_Enable/text()",
            str(Intermediate_Output_Enable).lower(),
        )
    if TOI_Start is not None:
        job_order.set(
            "//Processor_Configuration/Request/TOI/Start/text()",
            TOI_Start,
        )
    if TOI_Stop is not None:
        job_order.set(
            "//Processor_Configuration/Request/TOI/Stop/text()",
            TOI_Stop,
        )
    if Amount_of_RAM is not None:
        job_order.set(
            "//List_of_Tasks/Task/Amount_of_RAM/text()",
            MAX_RAM_USAGE,
        )
    if Number_of_CPU_Cores is not None:
        job_order.set(
            "//List_of_Tasks/Task/Number_of_CPU_Cores/text()",
            Number_of_CPU_Cores,
        )
    if DEM is not None:
        job_order.set(
            "//List_of_Cfg_Files/Cfg_File[Cfg_ID/text()='DEM']/Cfg_File_Name/text()",
            DEM,
        )
    if GMF is not None:
        job_order.set(
            "//List_of_Cfg_Files/Cfg_File[Cfg_ID/text()='GMF']/Cfg_File_Name/text()",
            GMF,
        )
    if S1_RAW__0S is not None:
        job_order.set(
            '//Selected_Input[File_Type/text()="S1_RAW__0S"]/List_of_File_Names/File_Name/text()',
            S1_RAW__0S,
        )
    if S2_RAW__0S is not None:
        job_order.set(
            '//Selected_Input[File_Type/text()="S2_RAW__0S"]/List_of_File_Names/File_Name/text()',
            S2_RAW__0S,
        )
    if S3_RAW__0S is not None:
        job_order.set(
            '//Selected_Input[File_Type/text()="S3_RAW__0S"]/List_of_File_Names/File_Name/text()',
            S3_RAW__0S,
        )
    if S1_RAW__0M is not None:
        job_order.set(
            '//Selected_Input[File_Type/text()="S1_RAW__0M"]/List_of_File_Names/File_Name/text()',
            S1_RAW__0M,
        )
    if S2_RAW__0M is not None:
        job_order.set(
            '//Selected_Input[File_Type/text()="S2_RAW__0M"]/List_of_File_Names/File_Name/text()',
            S2_RAW__0M,
        )
    if S3_RAW__0M is not None:
        job_order.set(
            '//Selected_Input[File_Type/text()="S3_RAW__0M"]/List_of_File_Names/File_Name/text()',
            S3_RAW__0M,
        )
    if AUX_ORB is not None:
        job_order.set(
            "//Input[Input_ID/text()='AUX_ORB___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_ORB,
        )
    if AUX_ATT is not None:
        job_order.set(
            "//Input[Input_ID/text()='AUX_ATT___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_ATT,
        )
    if AUX_TEC is not None:
        job_order.set(
            "//Input[Input_ID/text()='AUX_TEC___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_TEC,
        )
    if AUX_INS is not None:
        job_order.set(
            "//Input[Input_ID/text()='AUX_INS___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_INS,
        )
    if AUX_PP1 is not None:
        job_order.set(
            "//Input[Input_ID/text()='AUX_PP1___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_PP1,
        )
    if OUTPUT_DIR is not None:
        job_order.set("//Output/File_Dir/text()", OUTPUT_DIR)
    if IntermediateDataDir is not None:
        job_order.set("//List_of_Intermediate_Outputs/Intermediate_Output", "")
        job_order.set("//Intermediate_Output/Intermediate_Output_ID", "IntermediateDataDir")
        job_order.set(
            "//Intermediate_Output/Intermediate_Output_File",
            IntermediateDataDir,
        )
    job_order.set("//Processor_Version/text()", CURRENT_VERSION)
    job_order.set("//Task_Version/text()", CURRENT_VERSION)


def _update_aux_pp1(
    *,
    aux_pp1: xml.Document,
    raw_data_correction_flag: Literal["true", "false"] | None = None,
    bias_correction_flag: Literal["true", "false"] | None = None,
    antenna_pattern_correction1_flag: Literal["true", "false"] | None = None,
    antenna_pattern_correction2_flag: Literal["true", "false"] | None = None,
    polarimetric_correction_flag: Literal["true", "false"] | None = None,
    faraday_rotation_correction_flag: Literal["true", "false"] | None = None,
    ionospheric_phase_screen_correction_flag: Literal["true", "false"] | None = None,
    group_delay_correction_flag: Literal["true", "false"] | None = None,
    azimuth_block_overlap_lines: int | None = None,
    dc_method: str | None = None,
    dc_value: float | None = None,
    remove_margin_flag: Literal["true", "false"] | None = None,
    autofocus_flag: Literal["true", "false"] | None = None,
    dem_version: str | None = None,
    lut_range_decimation_factor: int | None = None,
    lut_azimuth_decimation_factor: int | None = None,
):
    """Update the aux pp1 file with new parameters."""
    if raw_data_correction_flag is not None:
        aux_pp1.set("//rawDataCorrectionFlag/text()", raw_data_correction_flag)
    if bias_correction_flag is not None:
        aux_pp1.set("//biasCorrectionFlag/text()", bias_correction_flag)
    if antenna_pattern_correction1_flag is not None:
        aux_pp1.set("//antennaPatternCorrection1Flag/text()", antenna_pattern_correction1_flag)
    if antenna_pattern_correction2_flag is not None:
        aux_pp1.set("//antennaPatternCorrection2Flag/text()", antenna_pattern_correction2_flag)
    if polarimetric_correction_flag is not None:
        aux_pp1.set("//polarimetricCorrectionFlag/text()", polarimetric_correction_flag)
    if faraday_rotation_correction_flag is not None:
        aux_pp1.set("//faradayRotationCorrectionFlag/text()", faraday_rotation_correction_flag)
    if ionospheric_phase_screen_correction_flag is not None:
        aux_pp1.set(
            "//ionosphericPhaseScreenCorrectionFlag/text()",
            ionospheric_phase_screen_correction_flag,
        )
    if group_delay_correction_flag is not None:
        aux_pp1.set("//groupDelayCorrectionFlag/text()", group_delay_correction_flag)
    if azimuth_block_overlap_lines is not None:
        aux_pp1.set("//blockOverlapLines/text()", azimuth_block_overlap_lines)
    if dc_method is not None:
        aux_pp1.set("//dcMethod/text()", dc_method)
    if dc_value is not None:
        aux_pp1.set("//dcValue/text()", dc_value)
    if remove_margin_flag is not None:
        aux_pp1.set("//azimuthFocusingMarginsRemovalFlag/text()", remove_margin_flag)
    if autofocus_flag is not None:
        aux_pp1.set("//autofocusFlag/text()", autofocus_flag)
    if dem_version is not None:
        aux_pp1.set("//heightModel/@version", dem_version)
    if lut_range_decimation_factor is not None:
        aux_pp1.set(
            "//lutRangeDecimationFactorList/lutDecimationFactor",
            lut_range_decimation_factor,
        )
    if lut_azimuth_decimation_factor is not None:
        aux_pp1.set(
            "//lutAzimuthDecimationFactorList/lutDecimationFactor",
            lut_azimuth_decimation_factor,
        )


def _set_stack_configuration(
    *,
    env: Environment,
    data: DataRepository,
    aux_pps_product_dir: Path,
    job_order_path: Path,
    output_directory: Path,
):
    """Setup for the stack test."""
    # Initialize the default nominal configurations.
    aux_pps_path = retrieve_aux_product_data_single_content(aux_pps_product_dir)
    initialize_aux_pps(aux_pps_path)

    # Update the JobOrder.
    job_order = xml.Document(job_order_path)

    # Get L1a products. Raise if amount does not match expected.
    l1a_list = _find_products(
        root_dir=env.root.parent.joinpath(L1_OUT_DIR),
        pattern="BIO_S*_SCS__1S*",
        expected_num_files=L1A_PRODUCTS_COUNT,
    )

    job_order.set(
        "//List_of_Tasks/Task/Amount_of_RAM/text()",
        MAX_RAM_USAGE,
    )
    job_order.set(
        "//List_of_Tasks/Task/Number_of_CPU_Cores/text()",
        WORKER_THREADS,
    )
    job_order.set(
        "//Proc_Parameter[Name/text()='primary_image']/Value",
        l1a_list[0],
    )
    job_order.set(
        "//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[1]/text()",
        l1a_list[0],
    )
    job_order.set(
        "//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[2]/text()",
        l1a_list[0],
    )

    # Remove third image from the INT job order.
    job_order.remove(
        "//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[3]"
    )

    job_order.set(
        "//Input[Input_ID/text()='AUX_PPS___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        aux_pps_product_dir,
    )

    job_order.set("//Output/File_Dir/text()", env.root.parent.joinpath(STA_OUT_DIR).absolute())
    job_order.set("//Cfg_File_Name/text()", data.pull("input/AUX-FNF"))

    job_order.set("//Processor_Version/text()", CURRENT_VERSION)
    job_order.set("//Task_Version/text()", CURRENT_VERSION)


def _get_l2a_tile_id(l2_product_path: Path) -> str:
    """Read the first tile ID of an L2a FH/D product."""
    annotation_file_path = list(l2_product_path.joinpath("annotation").glob("*_annot.xml"))
    if len(annotation_file_path) != 1:
        raise RuntimeError(f"One L2 FH annotation file is expected, but found {len(annotation_file_path)}")
    annotation_file = xml.Document(annotation_file_path[0])
    tile_id = annotation_file.get("//product/tileID/ID[1]")
    if len(tile_id) == 0:
        raise RuntimeError(f"No mission tile ID found in annotation file {annotation_file_path}")
    if not isinstance(tile_id, str):
        raise RuntimeError(f"tile_id is not type str, but: {type(tile_id)}")

    return tile_id


def _find_products(
    root_dir: Path,
    *,
    pattern: str,
    expected_num_files: int,
) -> list[Path]:
    """Possibly find file from root directory matching pattern."""
    found_products = list(root_dir.glob(pattern))
    if len(found_products) != expected_num_files:
        raise RuntimeError(
            f"Expected {expected_num_files} products matching pattern '{pattern}' in directory {root_dir}, but found {len(found_products)}"
        )
    return found_products


def _check_products(
    root_dir: Path,
    *,
    pattern: str,
    expected_num_files: int,
    product_reader: type[BIOMASSL1ProductReader]
    | type[BIOMASSStackProductReader]
    | type[BIOMASSL2aProductReader]
    | type[BIOMASSL2bFHProductReader]
    | type[BIOMASSL2bFDProductReader]
    | None = None,
):
    """Verify products from root directory matching pattern."""
    found_products = _find_products(
        root_dir=root_dir,
        pattern=pattern,
        expected_num_files=expected_num_files,
    )

    if product_reader is not None:
        # Test read the products.
        for product in found_products:
            assert product is not None
            try:
                product_reader(product).read()

            except Exception as ex:
                raise RuntimeError(f"Product {product.name} failed to read (error: {ex})") from ex


if __name__ == "__main__":
    from arepyextras.test.runner import Test, TestRunner
    from arepyextras.test.settings import Settings

    config_file = Path(__file__).parent.parent.parent / "config" / "local.toml"
    settings = Settings.load_toml(config_file)
    runner = TestRunner.from_settings(settings)

    runner.run_test(Test(test_002_sta.__name__, test_002_sta))
