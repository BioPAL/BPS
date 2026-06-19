# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Processor Tests - Configuration Tests
-------------------------------------------
"""

# Import the aux-pps utilities.
# fmt: off
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "aux_pps_utils"))
# fmt: on
import os
import tempfile
import traceback

import arepyextras.xml as xml
from arepyextras.test import DataRepository, Environment, TestSession
from aux_pps_utils import initialize_aux_pps
from bps.common import __version__
from bps.common.bps_logger import get_version_in_logger_format
from bps.common.common import retrieve_aux_product_data_single_content
from bps.transcoder.sarproduct.biomass_stackproduct_reader import (
    BIOMASSStackProductReader,
)
from bps.transcoder.sarproduct.biomass_stackproduct_writer import ZLIB_L1C_COMPLEVEL
from bps.transcoder.sarproduct.sta.product_content import BIOMASSStackProductStructure
from bps.transcoder.sarproduct.sta.quality_index import StackQualityIndex
from netCDF4 import Dataset

# Default print formats.
FAIL_FORMAT_STR = "\n  [ FAIL ] {:s}"
PASS_FORMAT_STR = "\n  [  OK  ] {:s}"

CURRENT_VERSION = get_version_in_logger_format(__version__)

OUTPUT_DIR = "od"  # Short name to keep the paths short enough for Windows.
WORKER_THREADS = 5
FRAMES_INT_TEST = 3
FRAMES_TOM_TEST = 7
DEFAULT_IOS_HEIGHT = 250000

# Procuct name L1a frame, used to find TDS in the testplan dataset.
TOM_REDUCED_FRAME_NAME = "BIO_S3_SCS__1S_20170213T015412_20170213T015417_T_G03_M03_C03_T000_F001_01_DNNMT7"


def test_bps_sta_001_int_with_4_polarizations(session: TestSession, env: Environment, data: DataRepository):
    """
    INT Stack Processor execution with 4 polarizations
    and "polarisationCombinationMethod" set to "None".
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//polarisationCombinationMethod/text()", "None")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_002_int_with_coreg_geometry(session: TestSession, env: Environment, data: DataRepository):
    """
    INT Stack Processor execution with
    "coregistrationMethod" set to "Geometry" aka Orbit,
    instead of "Geometry and Data" aka. COMBINED.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//coregistrationMethod/text()", "Geometry")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root, expected_ios_height=DEFAULT_IOS_HEIGHT)


def test_bps_sta_003_int_with_coreg_automatic_above_fitting_quality_threshold(
    session: TestSession, env: Environment, data: DataRepository
):
    """
    INT Stack Processor execution with
    "coregistrationMethod" set to "Automatic",
    where it will use "Geometry and Data".
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//coregistrationMethod/text()", "Automatic")
    aux_pps.set("//fittingQualityThreshold/text()", "0.0")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    fail = False
    for l1c_summary in _read_l1c(env.root):
        quality_index = StackQualityIndex.decode(l1c_summary["quality_index"])
        fail = fail or quality_index.coregistration_fitting_quality_below_threshold

    if fail:
        raise RuntimeError(
            "Found at least L1c product with wrong quality index: "
            "'coregistration_fitting_quality_below_threshold' expected False "
            "but got True"
        )


def test_bps_sta_004_int_with_coreg_automatic_below_fitting_quality_threshold(
    session: TestSession, env: Environment, data: DataRepository
):
    """
    INT Stack Processor execution with
    "coregistrationMethod" set to "Automatic",
    where it will fail to use "Geometry and Data"
    and then switch to "Geometry".
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//coregistrationMethod/text()", "Automatic")
    aux_pps.set("//fittingQualityThreshold/text()", "1.0")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    # NOTE: Coreg primary is always above threshold (no coregistration for primary).
    fail = False
    for l1c_summary in _read_l1c(env.root):
        if not l1c_summary["is_primary"]:
            quality_index = StackQualityIndex.decode(l1c_summary["quality_index"])
            fail = fail or not quality_index.coregistration_fitting_quality_below_threshold

    if fail:
        raise RuntimeError(
            "Found at least L1c product with wrong quality index: "
            "'coregistration_fitting_quality_below_threshold' expected True "
            "but got False"
        )


def test_bps_sta_005_tom_with_4_polarizations(session: TestSession, env: Environment, data: DataRepository):
    """
    TOM Stack Processor execution with 4 polarizations
    and "polarisationCombinationMethod" set to "None".
    """
    aux_pps, job_order_filename = _initialize_stack_tom_nominal_test(data)
    aux_pps.set("//polarisationCombinationMethod/text()", "None")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_006_tom_with_coreg_geometry(session: TestSession, env: Environment, data: DataRepository):
    """
    TOM Stack Processor execution with
    "coregistrationMethod" set to "Geometry" aka Orbit,
    instead of "Geometry and Data" aka. COMBINED.
    """
    aux_pps, job_order_filename = _initialize_stack_tom_nominal_test(data)
    aux_pps.set("//coregistrationMethod/text()", "Geometry")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_007_tom_with_coreg_automatic_above_fitting_quality_threshold(
    session: TestSession, env: Environment, data: DataRepository
):
    """
    TOM Stack Processor execution with
    "coregistrationMethod" set to "Automatic"
    where it will use "Geometry and Data".
    """
    aux_pps, job_order_filename = _initialize_stack_tom_nominal_test(data)
    aux_pps.set("//coregistrationMethod/text()", "Automatic")
    aux_pps.set("//fittingQualityThreshold/text()", "0.0")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_008_tom_with_coreg_automatic_below_fitting_quality_threshold(
    session: TestSession, env: Environment, data: DataRepository
):
    """
    TOM Stack Processor execution with
    "coregistrationMethod" set to "Automatic"
    where it will fail to use "Geometry and Data"
    and then switch to "Geometry".
    """
    aux_pps, job_order_filename = _initialize_stack_tom_nominal_test(data)
    aux_pps.set("//coregistrationMethod/text()", "Automatic")
    aux_pps.set("//fittingQualityThreshold/text()", "1.0")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_009_int_polarizations_used(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT but use non-default polarization for coreg."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//coregistration/polarisationUsed/text()", "VV")
    aux_pps.set("//slowIonosphereRemoval/slowIonosphereRemovalFlag/text()", "true")
    aux_pps.set("//slowIonosphereRemoval/polarisationUsed/text()", "HH")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_010_int_dem_bias_compensation(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT test but set 'false' the DEM bias compensation."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//general/flatteningPhaseBiasCompensationFlag/text()", "true")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )
    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_011_int_coreg_block_quality_threshold(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT test with non default block quality threshold."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//coregistration/blockQualityThreshold/text()", "0.95")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )
    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_012_int_skp_and_dsi_correction_flags(session: TestSession, env: Environment, data: DataRepository):
    """
    INT Stack Processor execution with
    only SKP+DSI correction activated.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//phaseCorrection", "Flattening Phase Screen")

    # Run the stack.
    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root, fit_to_l1c_raster=True)


def test_bps_sta_013_int_skp_correction_goldstein_filter(session: TestSession, env: Environment, data: DataRepository):
    """
    INT Stack Processor execution with SKP correction and post-processing via
    Goldstein filter.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseCalibration/phaseCorrection/text()", "Ground Phase Screen")
    aux_pps.set("//skpPhaseCalibration/calibrationPhasePostprocessing/text()", "Goldstein")
    aux_pps.set("//skpPhaseCalibration/goldsteinFilterWindowSize/text()", 5)

    # Run the stack.
    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_014_int_skp_correction_boxcar_filter(session: TestSession, env: Environment, data: DataRepository):
    """
    INT Stack Processor execution with SKP correction and post-processing via
    boxcar filter.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseCalibration/phaseCorrection/text()", "Ground Phase Screen")
    aux_pps.set("//skpPhaseCalibration/calibrationPhasePostprocessing/text()", "Boxcar")
    aux_pps.set("//skpPhaseCalibration/postprocessingFilterWindowSize/text()", 1000.0)

    # Run the stack.
    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root, fit_to_l1c_raster=True)


def test_bps_sta_015_int_skp_median_filter(session: TestSession, env: Environment, data: DataRepository):
    """
    INT test with data Median filter on SKP phase screens.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "true")
    aux_pps.set("//calibrationPhasePostprocessing/text()", "Median")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_016_4frames_allow_duplicates(session: TestSession, env: Environment, data: DataRepository):
    """4 Frames stack with duplicates allowed."""
    aux_pps, job_order_filename, l1a_frame_list = _initialize_stack_int_nominal_test(
        data,
        coreg_primary_index=0,
    )
    aux_pps.set("//allowDuplicateImagesFlag/text()", "true")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    # Duplicate frames.
    job_order = xml.Document(job_order_filename)
    for n, _ in enumerate(l1a_frame_list, start=1):
        job_order.set(
            f"//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[{n}]",
            l1a_frame_list[0],
        )

    # Run the stack.
    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_017_4frames_forbid_duplicates(session: TestSession, env: Environment, data: DataRepository):
    """4 Frames stack with duplicates not allowed."""
    aux_pps, job_order_filename, l1a_frame_list = _initialize_stack_int_nominal_test(
        data,
        coreg_primary_index=0,
    )
    aux_pps.set("//allowDuplicateImagesFlag/text()", "false")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    # Duplicate frames.
    job_order = xml.Document(job_order_filename)
    for n, _ in enumerate(l1a_frame_list, start=1):
        job_order.set(
            f"//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[{n}]",
            l1a_frame_list[0],
        )

    # Run the stack.
    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    session.expect_run_failure(stack_processor, fatal=True)


def test_bps_sta_018_int_l1c_lut_compression(session: TestSession, env: Environment, data: DataRepository):
    """Test that the LUTs are compressed as expected (zlib, level 4)."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)

    # Disable calibration, not needed for this test.
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    # Kick-off the stack.
    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    # Read all the LUTs.
    for l1c_product_path in env.root.glob(str(Path(OUTPUT_DIR) / "BIO_*__1S*")):
        l1c_product_structure = BIOMASSStackProductStructure(
            product_path=l1c_product_path,
            is_monitoring=False,
            exists_ok=True,
        )

        netcdf_object = Dataset(l1c_product_structure.lut_annotation2_file)
        for lut_group in netcdf_object.groups.keys():
            for lut_name in netcdf_object[lut_group].variables.keys():
                lut_filter = netcdf_object.groups[lut_group][lut_name].filters()
                if not lut_filter["zlib"] or lut_filter["complevel"] != ZLIB_L1C_COMPLEVEL:
                    raise RuntimeError(
                        FAIL_FORMAT_STR.format(
                            f"Found LUT with unexpected compression: lut={lut_name}, filter={lut_filter}"
                        )
                    )


def test_bps_sta_019_int_nondefault_l1c_config(session: TestSession, env: Environment, data: DataRepository):
    """4 Frames stack with duplicates not allowed."""
    no_data_value = -1234567.0

    aux_pps, job_order_filename, l1a_frame_list = _initialize_stack_int_nominal_test(data, coreg_primary_index=0)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")
    aux_pps.set("//noPixelValue/text()", no_data_value)
    aux_pps.set("//qlRangeDecimationFactor/text()", 4)
    aux_pps.set("//qlRangeAveragingFactor/text()", 5)
    aux_pps.set("//qlAzimuthDecimationFactor/text()", 20)
    aux_pps.set("//qlAzimuthAveragingFactor/text()", 21)
    aux_pps.set("//qlAbsoluteScalingFactor", 3)

    # Run the stack.
    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    # Read all the LUTs.
    for l1c_product_path in env.root.glob(str(Path(OUTPUT_DIR) / "BIO_*__1S*")):
        l1c_product_structure = BIOMASSStackProductStructure(
            product_path=l1c_product_path,
            is_monitoring=False,
            exists_ok=True,
        )

        netcdf_object = Dataset(l1c_product_structure.lut_annotation2_file)
        if netcdf_object.noDataValue != no_data_value:
            raise RuntimeError(
                FAIL_FORMAT_STR.format(
                    "Wrong noDataValue: ncdf={}, aux_pps={}".format(netcdf_object.noDataValue, no_data_value)
                )
            )


def test_bps_sta_020_int_use_nondefault_floating_point_config(
    session: TestSession, env: Environment, data: DataRepository
):
    """
    INT Stack Processor execution with non-default floating-point precision config.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFiltering/use32bitFlag/text()", "false")
    aux_pps.set("//skpPhaseCalibration/use32bitFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_021_tom_disable_pol_cross_covariance(session: TestSession, env: Environment, data: DataRepository):
    """
    SKP on TOM stack with only polarization auto-covariances.
    """
    aux_pps, job_order_filename = _initialize_stack_tom_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "true")
    aux_pps.set("//skpPhaseCalibration/excludeMPMBPolarizationCrossCovarianceFlag/text()", "true")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_022_int_disable_model_based_fit_average(session: TestSession, env: Environment, data: DataRepository):
    """
    INT test with data coregistration and no model based fit. All calibration modules on.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//modelBasedFitFlag/text()", "false")
    aux_pps.set("//lowPassFilterType/text()", "Average")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_023_int_disable_model_based_fit_gaussian(session: TestSession, env: Environment, data: DataRepository):
    """
    INT test with data coregistration and no model based fit. All calibration modules on.
    """
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//modelBasedFitFlag/text()", "false")
    aux_pps.set("//lowPassFilterType/text()", "Gaussian")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_024_tom_iob_only(session: TestSession, env: Environment, data: DataRepository):
    """
    IOB on a TOM.
    """
    aux_pps, job_order_filename = _initialize_stack_tom_nominal_test(data)
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "true")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")
    aux_pps.set("//multiSquintCalibrationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_025_tom_do_not_export_extra_ql(session: TestSession, env: Environment, data: DataRepository):
    """
    IOB on a TOM.
    """
    aux_pps, job_order_filename = _initialize_stack_tom_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")
    aux_pps.set("//qlExportCoherenceFlag/text()", "false")
    aux_pps.set("//qlExportInterferogramFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_026_int_msc_calibration(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT but use non-default polarization for coreg."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//multiSquintCalibrationFlag/text()", "true")
    aux_pps.set("//forceMultiSquintIonoCorrectionFlag/text()", "true")
    aux_pps.set("//disableMultiSquintIonoCorrectionFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root, fit_to_l1c_raster=True)


def test_bps_sta_027_int_msc_ms_coh_low_res(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT but use non-default polarization for coreg."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//multiSquintCalibrationFlag/text()", "true")
    aux_pps.set("//forceMultiSquintIonoCorrectionFlag/text()", "true")
    aux_pps.set("//disableMultiSquintIonoCorrectionFlag/text()", "false")
    aux_pps.set("//multiSquintCoherenceLowResThreshold/text()", "0.999")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_028_int_hsv_ql(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT but use non-default polarization for coreg."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//polarisationCombinationMethod/text()", "Average")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//multiSquintCalibrationFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "false")
    aux_pps.set("//l1cQlColourCodingMethod/text()", "HSV")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_029_int_skp_cross_pol_merging(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT but use non-default polarization for coreg."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//polarisationCombinationMethod/text()", "None")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//multiSquintCalibrationFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "true")
    aux_pps.set("//crossPolMergingFlag/text()", "true")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def test_bps_sta_030_int_skp_cross_pol_merging_only_copol(session: TestSession, env: Environment, data: DataRepository):
    """Execute a minimal INT but use non-default polarization for coreg."""
    aux_pps, job_order_filename, _ = _initialize_stack_int_nominal_test(data)
    aux_pps.set("//polarisationCombinationMethod/text()", "None")
    aux_pps.set("//azimuthSpectralFilteringFlag/text()", "false")
    aux_pps.set("//slowIonosphereRemovalFlag/text()", "false")
    aux_pps.set("//multiSquintCalibrationFlag/text()", "false")
    aux_pps.set("//skpPhaseEstimationFlag/text()", "true")
    aux_pps.set("//crossPolMergingFlag/text()", "true")
    aux_pps.set("//excludeMPMBPolarizationCrossCovarianceFlag/text()", "true")

    stack_processor = env.run(
        "bps_stack_processor",
        "--working-dir",
        tempfile.mkdtemp(dir=env.root),
        job_order_filename,
    )

    if stack_processor.returncode != 0:
        assert stack_processor.stderr is not None
        print(stack_processor.stderr.read_text())
    session.expect_run_successful(stack_processor, fatal=True)

    _read_l1c(env.root)


def _initialize_stack_tom_nominal_test(data, coreg_primary_index=1, calibration_primary_index=None):
    """Initialize TOM phase test with nominal configuration on a reduced dataset (5k x 1k)."""
    # Fetch the JobOrder.
    job_order_filename = data.pull("input/sta_tom/joborder")
    job_order = xml.Document(job_order_filename)

    # Initialize the AUX-PPS.
    aux_pps_product_dir = data.pull("configurations/sta/CONF-BPS-PPS-001")
    aux_pps_path = retrieve_aux_product_data_single_content(aux_pps_product_dir)
    initialize_aux_pps(aux_pps_path)
    aux_pps = xml.Document(aux_pps_path)

    # Get list of frames in the reduced TOM CI-TDS.
    l1a_frame_list = _get_ci_frame_list(
        data,
        tds_path="product/tom_ci_data_reduced/l1a/",
        product_name=TOM_REDUCED_FRAME_NAME,
    )

    # Fill the job order.
    _fill_ci_job_order_in_place(
        data,
        job_order,
        aux_pps_product_dir,
        coreg_primary_index=coreg_primary_index,
        calibration_primary_index=calibration_primary_index,
        Nimg=FRAMES_TOM_TEST,
        frame_list=l1a_frame_list,
        bps_version=CURRENT_VERSION,
        output_dir=OUTPUT_DIR,
        worker_threads=WORKER_THREADS,
    )

    return aux_pps, job_order_filename


def _initialize_stack_int_nominal_test(data, coreg_primary_index=1, calibration_primary_index=None):
    """Initialize INT phase test with nominal configuration on a reduced dataset (5k x 1k)."""
    # Fetch the JobOrder.
    job_order_filename = data.pull("input/sta_int/joborder")
    job_order = xml.Document(job_order_filename)

    # Initialize the AUX-PPS.
    aux_pps_product_dir = data.pull("configurations/sta/CONF-BPS-PPS-001")
    aux_pps_path = retrieve_aux_product_data_single_content(aux_pps_product_dir)
    initialize_aux_pps(aux_pps_path)
    aux_pps = xml.Document(aux_pps_path)

    # Get list of frames in the reduced TOM CI-TDS. Here we only use the first
    # three as this is an INT test.
    l1a_frame_list = _get_ci_frame_list(
        data,
        tds_path="product/tom_ci_data_reduced/l1a/",
        product_name=TOM_REDUCED_FRAME_NAME,
        only_first_n_frames=FRAMES_INT_TEST,
    )

    # Fill the job order.
    l1a_frame_list = _fill_ci_job_order_in_place(
        data,
        job_order,
        aux_pps_product_dir,
        coreg_primary_index=coreg_primary_index,
        calibration_primary_index=calibration_primary_index,
        Nimg=FRAMES_INT_TEST,
        frame_list=l1a_frame_list,
        bps_version=CURRENT_VERSION,
        output_dir=OUTPUT_DIR,
        worker_threads=WORKER_THREADS,
    )

    return aux_pps, job_order_filename, l1a_frame_list


def _read_l1c(
    root_path: Path,
    *,
    fit_to_l1c_raster: bool = False,
    l1a_frame_list: list[Path] | None = None,
    expected_ios_height: int | None = None,
    expected_coreg_primary_index: int | None = None,
    expected_calibration_primary_index: int | None = None,
) -> list[dict]:
    """Check by reading the L1c product and LUT ADS"""
    errors = []
    l1c_product_paths = list(root_path.glob(str(Path(OUTPUT_DIR).joinpath("BIO*__1S*"))))
    if len(l1c_product_paths) == 0:
        errors.append(
            FAIL_FORMAT_STR.format(
                f"No L1c products were found in the output directory at {os.path.join(root_path, OUTPUT_DIR)}"
            )
        )

    l1c_summaries = []
    for l1c_product_path in l1c_product_paths:
        try:
            stack_product_reader = BIOMASSStackProductReader(l1c_product_path)
            stack_product = stack_product_reader.read()

            l1c_summaries.append(
                {
                    "product_name": stack_product.name,
                    "stack_processing_parameters": stack_product.stack_processing_parameters,
                    "stack_in_sarparameters": stack_product.stack_in_sarparameters,
                    "stack_coregistration_parameters": stack_product.stack_coregistration_parameters,
                    "quality_index": stack_product.stack_quality.overall_product_quality_index,
                    "is_primary": (
                        stack_product.stack_coregistration_parameters.primary_image
                        == stack_product.stack_coregistration_parameters.secondary_image
                    ),
                }
            )

        except:  # noqa: E722
            errors.append(FAIL_FORMAT_STR.format(f"Invalid L1c product {l1c_product_path}"))
            traceback.print_exc()

        try:
            stack_product_reader.read_lut_annotation(fit_to_l1c_raster=fit_to_l1c_raster)
        except:  # noqa: E722
            errors.append(FAIL_FORMAT_STR.format(f"Ivalid L1c product LUTS {l1c_product_path}"))
            traceback.print_exc()

    if errors:
        raise RuntimeError("\n".join(errors))

    return l1c_summaries


def _find_frame_index(
    product_name: str,
    l1a_frame_list: list[Path],
) -> int:
    """Find index of frame in stack"""
    frame_index = None

    for i, frame in enumerate(l1a_frame_list):
        if frame.name in product_name:
            frame_index = i
    if frame_index is None:
        raise RuntimeError(f"Product name {product_name} was not found in {l1a_frame_list}")
    return frame_index


def _fill_ci_job_order_in_place(
    data: DataRepository,
    job_order: xml.Document,
    aux_pps_filename: Path,
    *,
    Nimg: int,
    frame_list: list[Path],
    bps_version: str,
    output_dir: str,
    worker_threads: int,
    max_ram_usage_MB: int = 64000,
    max_disk_usage_MB: int = 55000,
    coreg_primary_index: int | None = None,
    calibration_primary_index: int | None = None,
) -> list[Path]:
    """
    Fill a CI job order in place, using arepyextras.xml.

    Parameters
    ----------
    data: DataRepository
        CI dataset.

    job_order: xml.Document
       The job order to fill.

    aux_pps_filename: xml.Document
        AUX-PPS path to write to the job order.

    Nimg: int
        Numer of images in the test.

    frame_list: list[Path]
        The list of frames in the CI-TDS.

    bps_version: str
        The BPS SW version to write in the job order.

    output_dir: str
        The output_dir to write in the job order.

    worker_threads: int
        The number of worker threads to set in the job order.

    max_ram_usage_MB: int = 64000 [MB]
        The amount of max RAM (in Megabytes) to write in the job order.

    max_disk_usage_MB: int = 50000 [MB]
        The amout of disk usage (in Megabytes) to write in the job order.

    coreg_primary_index: Optional[int] = None
        Frame index to use for primary image.

    calibration_primary_index: Optional[int] = None
        Frame index to use for calibration image.

    Return
    ------
    list[Path]
        The paths of the L1a input stack.

    """
    if coreg_primary_index is not None and coreg_primary_index >= Nimg:
        raise ValueError("Coregistration primary index must be smaller than Nimg.")
    if calibration_primary_index is not None and calibration_primary_index >= Nimg:
        raise ValueError("Calibration reference index must be smaller than Nimg.")
    if len(frame_list) == 0:
        raise ValueError("Frame list cannot be empty")

    # Pull the L1a input stack.
    l1a_frame_paths = [_pull_wrap(data, frame_list, i) for i in range(Nimg)]

    # Set the input stack.
    for n, l1a_frame_path in enumerate(l1a_frame_paths, start=1):
        job_order.set(
            f"//Input[Input_ID/text()='Sx_SCS__1S']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name[{n}]",
            str(l1a_frame_path),
        )

    # Set the primary image.
    if coreg_primary_index is not None:
        job_order.set(
            "//Proc_Parameter[Name/text()='primary_image']/Value",
            str(l1a_frame_paths[coreg_primary_index]),
        )

    if calibration_primary_index is not None:
        job_order.set(
            "//Proc_Parameter[Name/text()='calibration_primary_image']/Value",
            str(l1a_frame_paths[calibration_primary_index]),
        )

    # Populate the AUX-PPS.
    job_order.set(
        "//Input[Input_ID/text()='AUX_PPS___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        aux_pps_filename,
    )

    # Pulling and writing remaining images to the joborder.
    job_order.set("//Output/File_Dir/text()", output_dir)
    job_order.set("//Processor_Version/text()", bps_version)
    job_order.set("//Task_Version/text()", bps_version)
    job_order.set("//Number_of_CPU_Cores/text()", worker_threads)
    job_order.set("//List_of_Tasks/Task/Amount_of_RAM/text()", max_ram_usage_MB)
    job_order.set("//List_of_Tasks/Task/Disk_Space/text()", max_disk_usage_MB)
    job_order.set("//Cfg_File_Name/text()", data.pull("input/AUX-FNF"))

    return l1a_frame_paths


def _pull_wrap(
    data: DataRepository,
    l1a_frame_list: list[Path],
    frame_index: int,
) -> Path:
    """
    Create and return a symbolic link to the frame at frame_index,
    keeping its original name.
    """
    try:
        frame_path = data.pull(str(l1a_frame_list[frame_index]), str(l1a_frame_list[frame_index].name))

    except Exception as ex:
        raise RuntimeError(f"Failed to pull frame (error: {ex})") from ex

    return frame_path


def _get_ci_frame_list(
    data: DataRepository,
    *,
    tds_path: str,
    product_name: str,
    only_first_n_frames: int | None = None,
) -> list[Path]:
    """
    Get a list of frames in the data TDS.

    Parameters
    ----------
    data: DataRepository
        CI dataset.

    tds_path: str
       Path of the tds, relative to the root of the CI dataset.

    product_name: str
        Name of a product inside the tds.

    Raises
    ------
    RuntimeError

    Return
    ------
    list[Path]
        A list of first n, or all frames found at the tds_path in the CI dataset.

    """
    try:
        l1a_frame_list = sorted(
            [p for p in data.fetch_data(str(Path(tds_path).joinpath(product_name))).parent.parent.glob("BIO_*")]
        )

    except Exception as ex:
        raise RuntimeError(f"Failed to create list of frames (error: {ex})") from ex

    if only_first_n_frames is not None:
        l1a_frame_list = l1a_frame_list[:only_first_n_frames]

    return l1a_frame_list
