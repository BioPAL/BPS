# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1p tests
---------
"""

import os
import shutil
import sys
from pathlib import Path

import arepyextras.xml as xml
from arepyextras.test import DataRepository, Environment, TestSession, skip, skip_if
from bps.common import __version__, retrieve_aux_product_data_single_content
from bps.common.bps_logger import get_version_in_logger_format

WIN32 = sys.platform == "win32"
PY311 = sys.version_info.major == 3 and sys.version_info.minor == 11
AUTOFOCUS_ENABLED = PY311 and not WIN32

CURRENT_VERSION = get_version_in_logger_format(__version__)


def update_joborder(
    joborder: xml.Document,
    *,
    Intermediate_Output_Enable: bool = False,
    TOI_Start: str | None = None,
    TOI_Stop: str | None = None,
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
    IRI: Path | str | None = None,
    AUX_TEC: Path | str | None = None,
    AUX_INS: Path | str | None = None,
    AUX_PP1: Path | str | None = None,
    OUTPUT_DIR: Path | str | None = None,
    IntermediateDataDir: Path | str | None = None,
):
    if Intermediate_Output_Enable:
        joborder.set(
            "//Processor_Configuration/Intermediate_Output_Enable/text()",
            str(Intermediate_Output_Enable).lower(),
        )

    if TOI_Start:
        joborder.set(
            "//Processor_Configuration/Request/TOI/Start/text()",
            TOI_Start,
        )

    if TOI_Stop:
        joborder.set(
            "//Processor_Configuration/Request/TOI/Stop/text()",
            TOI_Stop,
        )

    if DEM:
        joborder.set(
            "//List_of_Cfg_Files/Cfg_File[Cfg_ID/text()='DEM']/Cfg_File_Name/text()",
            DEM,
        )
    if GMF:
        joborder.set(
            "//List_of_Cfg_Files/Cfg_File[Cfg_ID/text()='GMF']/Cfg_File_Name/text()",
            GMF,
        )
    if IRI:
        iri_path = joborder.get(
            "//List_of_Cfg_Files/Cfg_File[Cfg_ID/text()='IRI']/Cfg_File_Name/text()",
        )
        if iri_path is None:
            joborder.add("//List_of_Cfg_Files", "Cfg_File", "")
            joborder.add("//List_of_Cfg_Files/Cfg_File[last()]", "Cfg_ID", "IRI")
            joborder.add("//List_of_Cfg_Files/Cfg_File[last()]", "Cfg_File_Name", IRI)
        else:
            joborder.set(
                "//List_of_Cfg_Files/Cfg_File[Cfg_ID/text()='IRI']/Cfg_File_Name/text()",
                IRI,
            )

    if S1_RAW__0S:
        joborder.set(
            '//Selected_Input[File_Type/text()="S1_RAW__0S"]/List_of_File_Names/File_Name/text()',
            S1_RAW__0S,
        )
    if S2_RAW__0S:
        joborder.set(
            '//Selected_Input[File_Type/text()="S2_RAW__0S"]/List_of_File_Names/File_Name/text()',
            S2_RAW__0S,
        )
    if S3_RAW__0S:
        joborder.set(
            '//Selected_Input[File_Type/text()="S3_RAW__0S"]/List_of_File_Names/File_Name/text()',
            S3_RAW__0S,
        )
    if S1_RAW__0M:
        joborder.set(
            '//Selected_Input[File_Type/text()="S1_RAW__0M"]/List_of_File_Names/File_Name/text()',
            S1_RAW__0M,
        )
    if S2_RAW__0M:
        joborder.set(
            '//Selected_Input[File_Type/text()="S2_RAW__0M"]/List_of_File_Names/File_Name/text()',
            S2_RAW__0M,
        )
    if S3_RAW__0M:
        joborder.set(
            '//Selected_Input[File_Type/text()="S3_RAW__0M"]/List_of_File_Names/File_Name/text()',
            S3_RAW__0M,
        )

    if AUX_ORB:
        joborder.set(
            "//Input[Input_ID/text()='AUX_ORB___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_ORB,
        )
    if AUX_ATT:
        joborder.set(
            "//Input[Input_ID/text()='AUX_ATT___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_ATT,
        )
    if AUX_TEC:
        joborder.set(
            "//Input[Input_ID/text()='AUX_TEC___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_TEC,
        )
    if AUX_INS:
        joborder.set(
            "//Input[Input_ID/text()='AUX_INS___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_INS,
        )
    if AUX_PP1:
        joborder.set(
            "//Input[Input_ID/text()='AUX_PP1___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
            AUX_PP1,
        )
    if OUTPUT_DIR:
        joborder.set("//Output/File_Dir/text()", OUTPUT_DIR)

    if IntermediateDataDir is not None:
        joborder.set("//List_of_Intermediate_Outputs/Intermediate_Output", "")
        joborder.set("//Intermediate_Output/Intermediate_Output_ID", "IntermediateDataDir")
        joborder.set(
            "//Intermediate_Output/Intermediate_Output_File",
            str(IntermediateDataDir),
        )

    joborder.set("//Processor_Version/text()", CURRENT_VERSION)
    joborder.set("//Task_Version/text()", CURRENT_VERSION)

    joborder.set("//Number_of_CPU_Cores", 2)
    joborder.set("//Amount_of_RAM", 15360)
    joborder.set("//Disk_Space", 26624)


def update_aux_pp1(
    aux_pp1: xml.Document,
    raw_data_correction_flag: str | None = None,
    bias_correction_flag: str | None = None,
    antenna_pattern_correction1_flag: str | None = None,
    antenna_pattern_correction2_flag: str | None = None,
    polarimetric_correction_flag: str | None = None,
    faraday_rotation_correction_flag: str | None = None,
    ionospheric_phase_screen_correction_flag: str | None = None,
    group_delay_correction_flag: str | None = None,
    azimuth_block_overlap_lines: int | None = None,
    dc_method: str | None = None,
    dc_value: float | None = None,
    remove_margin_flag: str | None = None,
    autofocus_flag: str | None = None,
    dem_version: str | None = None,
    allow_dem_fallback: str | None = None,
):
    """Update the aux pp1 file with new parameters"""
    if raw_data_correction_flag:
        aux_pp1.set("//rawDataCorrectionFlag/text()", raw_data_correction_flag)
    if bias_correction_flag:
        aux_pp1.set("//biasCorrectionFlag/text()", bias_correction_flag)
    if antenna_pattern_correction1_flag:
        aux_pp1.set("//antennaPatternCorrection1Flag/text()", antenna_pattern_correction1_flag)
    if antenna_pattern_correction2_flag:
        aux_pp1.set("//antennaPatternCorrection2Flag/text()", antenna_pattern_correction2_flag)
    if polarimetric_correction_flag:
        aux_pp1.set("//polarimetricCorrectionFlag/text()", polarimetric_correction_flag)
    if faraday_rotation_correction_flag:
        aux_pp1.set("//faradayRotationCorrectionFlag/text()", faraday_rotation_correction_flag)
    if ionospheric_phase_screen_correction_flag:
        aux_pp1.set(
            "//ionosphericPhaseScreenCorrectionFlag/text()",
            ionospheric_phase_screen_correction_flag,
        )
    if group_delay_correction_flag:
        aux_pp1.set("//groupDelayCorrectionFlag/text()", group_delay_correction_flag)
    if azimuth_block_overlap_lines:
        aux_pp1.set("//blockOverlapLines/text()", str(azimuth_block_overlap_lines))
    if dc_method:
        aux_pp1.set("//dcMethod/text()", dc_method)
    if dc_value:
        aux_pp1.set("//dcValue/text()", str(dc_value))
    if remove_margin_flag:
        aux_pp1.set("//azimuthFocusingMarginsRemovalFlag/text()", remove_margin_flag)
    if autofocus_flag:
        aux_pp1.set("//autofocusFlag/text()", autofocus_flag)
    if dem_version:
        aux_pp1.set("//heightModel/@version", dem_version)
    if allow_dem_fallback:
        aux_pp1.set("//heightModelFallbackFlag/text()", allow_dem_fallback)


def test_bps_l1_001(session: TestSession, env: Environment, _: DataRepository):
    """L1 processor installation and setup (display help)"""

    completed_process = env.run("bps_l1_processor", "--help")

    assert completed_process.stdout is not None
    session.expect_true(completed_process.stdout.read_text().startswith("Usage: bps_l1_processor"))

    if completed_process.returncode != 0:
        assert completed_process.stderr is not None
        print(completed_process.stderr.read_text())

    session.expect_run_successful(completed_process, fatal=True)


def test_bps_l1_002(session: TestSession, env: Environment, data: DataRepository):
    """Minimal L1 processor execution (Stripmap mode)"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    output_directory = env.root.parent.joinpath("l1002").absolute()

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-002")
    aux_pp1_file = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_file)
    update_aux_pp1(aux_pp1)

    update_joborder(
        joborder,
        S2_RAW__0S=data.pull("product/l1001/l0/S1_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1001/l0/S1_RAW__0M"),
        AUX_ORB=data.pull("product/l1001/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1001/aux_files/AUX_ATT"),
        AUX_INS=aux_ins_path,
        AUX_PP1=aux_pp1_path,
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        OUTPUT_DIR=output_directory,
    )

    l1_proc = env.run("bps_l1_processor", joborder_path)

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)


def test_bps_l1_002_with_error(session: TestSession, env: Environment, data: DataRepository):
    """L1 processor: proper output and exit code on wrong input joborder"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    output_directory = env.root.parent.joinpath("l1002e").absolute()

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    update_joborder(
        joborder,
        S2_RAW__0S=str(data.pull("product/l1001/l0/S1_RAW__0S")) + "__NOT_EXISTING",
        S2_RAW__0M=data.pull("product/l1001/l0/S1_RAW__0M"),
        AUX_ORB=data.pull("product/l1001/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1001/aux_files/AUX_ATT"),
        AUX_INS=aux_ins_path,
        AUX_PP1=data.pull("configurations/l1p/CONF-BPS-PP1-002"),
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        OUTPUT_DIR=output_directory,
    )

    l1_proc = env.run("bps_l1_processor", joborder_path)

    session.expect_run_failure(l1_proc, fatal=True)
    session.expect_equal(l1_proc.returncode, 128)


def dem_setup(env: Environment) -> tuple[Path, str]:
    """Setup dem in the new layout"""
    copernicus_dem_folder = env.getenv("COPERNICUS_DEM_FOLDER")
    if copernicus_dem_folder is None:
        raise RuntimeError(
            f"{test_bps_l1_003_enable_intermediates.__name__} requires the environment variable "
            + "'COPERNICUS_DEM_FOLDER': check your testing environment toml file."
        )
    assert isinstance(copernicus_dem_folder, str)

    copernicus_version = Path(copernicus_dem_folder).name
    dem_dir = Path("DEM").absolute()
    copernicus_dir = dem_dir.joinpath("copernicus")
    copernicus_dir.mkdir(parents=True, exist_ok=True)
    link = copernicus_dir.joinpath(copernicus_version)
    os.symlink(src=copernicus_dem_folder, dst=link, target_is_directory=True)

    return dem_dir, copernicus_version


def test_bps_l1_003_enable_intermediates(session: TestSession, env: Environment, data: DataRepository):
    """Nominal L1 Processor execution (Stripmap mode) with outputs analysis"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    shortest_time_of_interest = {
        "start": "2017-01-11T05:06:07.332238314942",
        "stop": "2017-01-11T05:06:24.000325410344",
    }

    dem_dir, _ = dem_setup(env)

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-003")
    aux_pp1_file = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_file)

    update_aux_pp1(
        aux_pp1=aux_pp1, azimuth_block_overlap_lines=7654, remove_margin_flag="false", allow_dem_fallback="true"
    )

    # write in the parent dir to avoid the long test dir name to be in the path
    output_directory = env.root.parent.joinpath("l1003int").absolute()

    intermediate_data_dir = Path("wd")

    update_joborder(
        joborder,
        Intermediate_Output_Enable=True,
        TOI_Start=shortest_time_of_interest["start"],
        TOI_Stop=shortest_time_of_interest["stop"],
        DEM=str(dem_dir),
        S2_RAW__0S=data.pull("product/l1002/l0/S2_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1002/l0/S2_RAW__0M"),
        AUX_ORB=data.pull("product/l1002/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1002/aux_files/AUX_ATT"),
        GMF=data.pull("internal_resources/GMF_14"),
        IRI=data.pull("internal_resources/IRI"),
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        AUX_INS=aux_ins_path,
        AUX_PP1=aux_pp1_path,
        IntermediateDataDir=str(intermediate_data_dir),
        OUTPUT_DIR=output_directory,
    )

    l1_pre_processor_output_dir_name = "l1_pre_processor_output"
    l1_core_processor_output_dir_name = "l1_core_processor_output"

    l1_proc = env.run("bps_l1_processor", joborder_path)

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)

    # Test 009: analysis of L1 Pre-Processing step outputs
    preproc_dir = intermediate_data_dir.joinpath(l1_pre_processor_output_dir_name)
    session.expect_true(preproc_dir.joinpath("iRAW").exists())
    session.expect_true(preproc_dir.joinpath("iChirp").exists())
    session.expect_true(preproc_dir.joinpath("iEstNoise").exists())
    session.expect_true(preproc_dir.joinpath("iDrift").exists())
    session.expect_true(preproc_dir.joinpath("iTxPT").exists())
    session.expect_true(preproc_dir.joinpath("iPatt2D_D1H").exists())
    session.expect_true(preproc_dir.joinpath("iPatt2D_D1V").exists())
    session.expect_true(preproc_dir.joinpath("iPatt2D_D2H").exists())
    session.expect_true(preproc_dir.joinpath("iPatt2D_D2V").exists())
    session.expect_true(preproc_dir.joinpath("iDelays.xml").is_file())
    session.expect_true(preproc_dir.joinpath("iChannelImbalance.xml").is_file())

    # Test 010: analysis of L1 Core-Processing step outputs
    core_dir = intermediate_data_dir.joinpath(l1_core_processor_output_dir_name)
    session.expect_true(core_dir.joinpath("iSLC").exists())
    session.expect_true(core_dir.joinpath("iGRD").exists())
    session.expect_true(core_dir.joinpath("iRAW_rfi_corrected").exists())
    session.expect_true(core_dir.joinpath("iRGC").exists())
    session.expect_true(core_dir.joinpath("iSRD").exists())

    # Test 011: analysis of L1 Post-Processing step outputs
    processor_output_products = [
        sub_dir.name for sub_dir in output_directory.iterdir() if sub_dir.name.startswith("BIO_")
    ]

    session.expect_true(len(processor_output_products) == 3)


def test_bps_l1_003_disable_intermediates(session: TestSession, env: Environment, data: DataRepository):
    """Nominal L1 Processor execution (Stripmap mode) with outputs analysis"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    shortest_time_of_interest = {
        "start": "2017-01-11T05:06:07.332238314942",
        "stop": "2017-01-11T05:06:24.000325410344",
    }

    dem_dir, _ = dem_setup(env)

    # write in the parent dir to avoid the long test dir name to be in the path
    output_directory = env.root.parent.joinpath("l1003noint").absolute()

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-003")
    aux_pp1_file = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_file)

    update_aux_pp1(
        aux_pp1=aux_pp1, azimuth_block_overlap_lines=7654, remove_margin_flag="false", allow_dem_fallback="true"
    )

    update_joborder(
        joborder,
        Intermediate_Output_Enable=False,
        TOI_Start=shortest_time_of_interest["start"],
        TOI_Stop=shortest_time_of_interest["stop"],
        DEM=str(dem_dir),
        S2_RAW__0S=data.pull("product/l1002/l0/S2_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1002/l0/S2_RAW__0M"),
        AUX_ORB=data.pull("product/l1002/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1002/aux_files/AUX_ATT"),
        GMF=data.pull("internal_resources/GMF_14"),
        IRI=data.pull("internal_resources/IRI"),
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        AUX_INS=aux_ins_path,
        AUX_PP1=aux_pp1_path,
        OUTPUT_DIR=output_directory,
    )

    working_dir = Path("wd")

    l1_proc = env.run(
        "bps_l1_processor",
        joborder_path,
        "--working-dir",
        working_dir,
    )

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)


@skip_if(
    not AUTOFOCUS_ENABLED,
    reason="Autofocus is available only on linux with py311",
)
def test_bps_l1_003_autofocus(session: TestSession, env: Environment, data: DataRepository):
    """Nominal L1 Processor execution (Stripmap mode) with autofocus"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    shortest_time_of_interest = {
        "start": "2017-01-11T05:06:07.332238314942",
        "stop": "2017-01-11T05:06:24.000325410344",
    }

    dem_dir, _ = dem_setup(env)

    output_directory = env.root.parent.joinpath("autofocus").absolute()

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-003")
    aux_pp1_file = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_file)
    update_aux_pp1(aux_pp1)

    update_joborder(
        joborder,
        Intermediate_Output_Enable=False,
        TOI_Start=shortest_time_of_interest["start"],
        TOI_Stop=shortest_time_of_interest["stop"],
        DEM=str(dem_dir),
        S2_RAW__0S=data.pull("product/l1002/l0/S2_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1002/l0/S2_RAW__0M"),
        AUX_ORB=data.pull("product/l1002/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1002/aux_files/AUX_ATT"),
        GMF=data.pull("internal_resources/GMF_14"),
        IRI=data.pull("internal_resources/IRI"),
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        AUX_INS=aux_ins_path,
        AUX_PP1=aux_pp1_path,
        OUTPUT_DIR=output_directory,
    )

    working_dir = Path("wd")

    l1_proc = env.run(
        "bps_l1_processor",
        joborder_path,
        "--working-dir",
        working_dir,
    )

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)


@skip()
def test_bps_l1_006(session: TestSession, env: Environment, data: DataRepository):
    """Processing of whole input L0 slice"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    # write in the parent dir to avoid the long test dir name to be in the path
    output_directory = env.root.parent.joinpath("l1006").absolute()

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    update_joborder(
        joborder,
        S2_RAW__0S=str(data.pull("product/l1001/l0/S1_RAW__0S")),
        S2_RAW__0M=data.pull("product/l1001/l0/S1_RAW__0M"),
        AUX_ORB=data.pull("product/l1001/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1001/aux_files/AUX_ATT"),
        AUX_INS=aux_ins_path,
        AUX_PP1=data.pull("configurations/l1p/CONF-BPS-PP1-002"),
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        OUTPUT_DIR=output_directory,
    )

    joborder.remove("//Processor_Configuration/Request/TOI")

    joborder.set(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[Name/text()='frame_id']/Value/text()",
        "___",
    )
    joborder.set(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[Name/text()='frame_status']/Value/text()",
        "NOT_FRAMED",
    )

    l1_proc = env.run("bps_l1_processor", joborder_path)

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)


def test_bps_l1_007(session: TestSession, env: Environment, data: DataRepository):
    """Processing of a selected ROI"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    output_directory = env.root.parent.joinpath("l1007").absolute()

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-002")
    aux_pp1_file = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_file)
    update_aux_pp1(aux_pp1)

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    update_joborder(
        joborder,
        TOI_Start="2017-01-11T05:06:07.332204000000",
        TOI_Stop="2017-01-11T05:06:32.425897302191",
        S2_RAW__0S=data.pull("product/l1002/l0/S2_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1002/l0/S2_RAW__0M"),
        AUX_ORB=data.pull("product/l1002/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1002/aux_files/AUX_ATT"),
        AUX_INS=aux_ins_path,
        AUX_PP1=aux_pp1_path,
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        OUTPUT_DIR=output_directory,
    )

    joborder.set(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[Name/text()='frame_status']/Value/text()",
        "PARTIAL",
    )

    joborder.add("//Task/List_of_Proc_Parameters", "Proc_Parameter", "")
    joborder.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Name",
        "range_start_time",
    )
    joborder.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Value",
        "0.005033364936743259",
    )

    joborder.add("//Task/List_of_Proc_Parameters", "Proc_Parameter", "")
    joborder.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Name",
        "range_stop_time",
    )
    joborder.add(
        "//Task/List_of_Proc_Parameters/Proc_Parameter[last()]",
        "Value",
        "0.005185799419326667",
    )

    l1_proc = env.run("bps_l1_processor", joborder_path)

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)


def test_bps_l1_restart_short(session: TestSession, env: Environment, data: DataRepository):
    """Nominal L1 Processor execution (Stripmap mode) with outputs analysis"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    shortest_time_of_interest = {
        "start": "2017-01-11T05:06:07.332238314942",
        "stop": "2017-01-11T05:06:24.000325410344",
    }

    dem_dir, dem_version = dem_setup(env)

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-003")
    aux_pp1_file = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_file)

    # write in the parent dir to avoid the long test dir name to be in the path
    output_directory = env.root.parent.joinpath("l1restart").absolute()

    update_joborder(
        joborder,
        TOI_Start=shortest_time_of_interest["start"],
        TOI_Stop=shortest_time_of_interest["stop"],
        DEM=str(dem_dir),
        S2_RAW__0S=data.pull("product/l1002/l0/S2_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1002/l0/S2_RAW__0M"),
        AUX_ORB=data.pull("product/l1002/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1002/aux_files/AUX_ATT"),
        GMF=data.pull("internal_resources/GMF_14"),
        IRI=data.pull("internal_resources/IRI"),
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        AUX_INS=aux_ins_path,
        AUX_PP1=aux_pp1_path,
        OUTPUT_DIR=output_directory,
    )

    update_aux_pp1(
        aux_pp1=aux_pp1,
        raw_data_correction_flag="false",
        bias_correction_flag="false",
        antenna_pattern_correction1_flag="false",
        antenna_pattern_correction2_flag="false",
        polarimetric_correction_flag="true",
        faraday_rotation_correction_flag="true",
        ionospheric_phase_screen_correction_flag="false",
        group_delay_correction_flag="false",
        azimuth_block_overlap_lines=4664,
        dem_version=dem_version,
    )

    working_dir = Path("wd")

    # First run: remove executables (to make fail the BiomassL0ImportPreProc)
    preprocessor = shutil.which("BiomassL0ImportPreProc")
    assert preprocessor is not None
    binaries_folder = str(Path(preprocessor).parent)

    path = env.getenv("PATH")
    assert path is not None and isinstance(path, list)
    path = list(filter(lambda p: p != binaries_folder, path))
    env.setenv("PATH", path)

    l1_proc = env.run("bps_l1_processor", joborder_path, "--working-dir", working_dir)

    session.expect_run_failure(l1_proc, fatal=True)
    session.expect_equal(l1_proc.returncode, 128)

    # Second run: fix binaries path
    path.insert(0, binaries_folder)
    env.setenv("PATH", path)

    # Second run: make core break by having an empty iri folder
    iri_empty = Path("empty_iri_folder").absolute()
    iri_empty.mkdir(exist_ok=True)
    update_joborder(joborder, IRI=iri_empty)

    l1_proc = env.run("bps_l1_processor", joborder_path, "--working-dir", working_dir)
    session.expect_run_failure(l1_proc, fatal=True)
    session.expect_equal(l1_proc.returncode, 128)
    session.expect_true("BiomassL0ImportPreProc completed" in l1_proc.stdout.read_text())

    # Third run
    update_joborder(joborder, IRI=data.pull("internal_resources/IRI"))
    l1_proc = env.run("bps_l1_processor", joborder_path, "--working-dir", working_dir)
    session.expect_true("L1PreProcessor already completed" in l1_proc.stdout.read_text())

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)

    # Fourth run: should do nothing
    l1_proc = env.run("bps_l1_processor", joborder_path, "--working-dir", working_dir)
    assert l1_proc.stdout is not None

    session.expect_true("L1PreProcessor already completed" in l1_proc.stdout.read_text())
    session.expect_true("L1CoreProcessor already completed" in l1_proc.stdout.read_text())
    session.expect_true("L1PostProcessor already completed" in l1_proc.stdout.read_text())

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)


def test_bps_l1_002_fixed_doppler(session: TestSession, env: Environment, data: DataRepository):
    """Nominal L1 processor execution (Stripmap mode) with fixed doppler"""

    joborder_path = data.pull("input/l1002/joborder")

    joborder = xml.Document(joborder_path)

    output_directory = env.root.parent.joinpath("l1002fixdoppler").absolute()

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-002")
    aux_pp1_file = retrieve_aux_product_data_single_content(aux_pp1_path)
    aux_pp1 = xml.Document(aux_pp1_file)

    update_joborder(
        joborder,
        S2_RAW__0S=data.pull("product/l1001/l0/S1_RAW__0S"),
        S2_RAW__0M=data.pull("product/l1001/l0/S1_RAW__0M"),
        AUX_ORB=data.pull("product/l1001/aux_files/AUX_ORB"),
        AUX_ATT=data.pull("product/l1001/aux_files/AUX_ATT"),
        AUX_INS=aux_ins_path,
        AUX_TEC=data.pull("configurations/l1p/CONF-BPS-TEC-003"),
        AUX_PP1=data.pull("configurations/l1p/CONF-BPS-PP1-002"),
        OUTPUT_DIR=output_directory,
    )

    dc_method = "Fixed"
    dc_value = 0.0

    update_aux_pp1(aux_pp1=aux_pp1, dc_method=dc_method, dc_value=dc_value)

    l1_proc = env.run("bps_l1_processor", joborder_path)

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)
