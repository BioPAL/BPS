# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to fill the processing options file
---------------------------------------------
"""

from pathlib import Path

from bps.common import bps_logger
from bps.common.utils import ProductFormat
from bps.l1_core_processor.processing_options import (
    AntennaPatternCompensationLevel,
    BPSL1CoreProcessingOptions,
    BPSL1CoreProcessorProductId,
    BPSL1CoreProcessorStep,
    EarthModel,
)
from bps.l1_processor.processor_interface import aux_pp1
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1JobOrder,
    L1StripmapProducts,
)
from bps.l1_processor.settings.intermediate_names import IntermediateProductID


def retrieve_bps_l1_core_processor_steps(
    job_order: L1JobOrder, aux_pp1_conf: aux_pp1.AuxProcessingParametersL1
) -> dict[BPSL1CoreProcessorStep, bool]:
    """Retrieve processing options steps form job order and aux pp1"""
    assert aux_pp1_conf.rfi_mitigation.activation_mode in ("Enabled", "Disabled")
    polarimetric_calibration_enabled = aux_pp1_conf.polarimetric_calibration.polarimetric_correction_flag
    denoising_enabled = aux_pp1_conf.thermal_denoising.thermal_denoising_flag

    grd_required = (
        isinstance(job_order.io_products, L1StripmapProducts) and job_order.io_products.output.dgm_standard_required
    )

    return {
        BPSL1CoreProcessorStep.RFI_MITIGATION: aux_pp1_conf.rfi_mitigation.activation_mode == "Enabled"
        or aux_pp1_conf.rfi_mitigation.detection_flag,
        BPSL1CoreProcessorStep.RANGE_FOCUSER: True,
        BPSL1CoreProcessorStep.DOPPLER_CENTROID_ESTIMATOR: True,
        BPSL1CoreProcessorStep.DOPPLER_RATE_ESTIMATOR: True,
        BPSL1CoreProcessorStep.AZIMUTH_FOCUSER: True,
        BPSL1CoreProcessorStep.RANGE_COMPENSATOR: True,
        BPSL1CoreProcessorStep.POLARIMETRIC_COMPENSATOR: polarimetric_calibration_enabled,
        BPSL1CoreProcessorStep.MULTI_LOOKER: grd_required,
        BPSL1CoreProcessorStep.NESZ_MAP_GENERATOR: True,
        BPSL1CoreProcessorStep.DENOISER: grd_required and denoising_enabled,
        BPSL1CoreProcessorStep.SLANT2_GROUND: grd_required,
    }


def determine_apc_level(apc_conf: aux_pp1.AntennaPatternCorrectionConf) -> AntennaPatternCompensationLevel:
    """Determine level of APC"""
    apc1_enabled = apc_conf.antenna_pattern_correction1_flag
    apc_cross_enabled = apc_conf.antenna_cross_talk_correction_flag
    apc2_enabled = apc_conf.antenna_pattern_correction2_flag

    if not apc1_enabled:
        if apc_cross_enabled:
            bps_logger.warning("APC1 disabled, APC Cross Talk disabled")
        if apc2_enabled:
            raise RuntimeError("Not possible to activate APC2, when APC1 is disabled")

        return AntennaPatternCompensationLevel.DISABLED

    if apc_cross_enabled and apc2_enabled:
        return AntennaPatternCompensationLevel.FULL

    if apc_cross_enabled:
        assert not apc2_enabled
        return AntennaPatternCompensationLevel.APC_PRE_CROSS_ONLY

    if apc2_enabled:
        assert not apc_cross_enabled
        return AntennaPatternCompensationLevel.APC_PRE_POST_ONLY

    assert not apc_cross_enabled and not apc2_enabled
    return AntennaPatternCompensationLevel.APC_PRE_ONLY


def retrieve_processing_settings(
    aux_pp1_conf: aux_pp1.AuxProcessingParametersL1,
) -> BPSL1CoreProcessingOptions.ProcessingSettings:
    """Retrieve processing options processing settings form aux pp1"""

    apc_level = determine_apc_level(aux_pp1_conf.antenna_pattern_correction)

    aux_pp1_earth_model = aux_pp1_conf.general.height_model
    if aux_pp1_earth_model == aux_pp1.GeneralConf.EarthModel.ELLIPSOID:
        earth_model = EarthModel.WGS84
    elif aux_pp1_earth_model == aux_pp1.GeneralConf.EarthModel.COPERNICUS_DEM:
        earth_model = EarthModel.COPERNICUS
    elif aux_pp1_earth_model == aux_pp1.GeneralConf.EarthModel.SRTM:
        earth_model = EarthModel.SRTM
    else:
        raise RuntimeError(f"Earth model in aux pp1: {aux_pp1_earth_model} not supported")

    dc_est_earth_model = earth_model

    az_foc_earth_model = (
        earth_model if aux_pp1_conf.antenna_pattern_correction.antenna_pattern_correction1_flag else EarthModel.WGS84
    )

    rfi_use_chirp_product = (
        "ANNOTATION"
        if aux_pp1_conf.rfi_mitigation.freq_domain_processing_parameters.chirp_source == aux_pp1.ChirpSource.NOMINAL
        else "PRODUCT"
    )

    operation_mode = aux_pp1_conf.rfi_mitigation.activation_mode
    assert operation_mode in ("Enabled", "Disabled")
    rfi_operation_mode = "DETECTION_ONLY" if operation_mode == "Disabled" else "DETECTION_AND_MITIGATION"

    return BPSL1CoreProcessingOptions.ProcessingSettings(
        dem={
            BPSL1CoreProcessorStep.DOPPLER_CENTROID_ESTIMATOR: dc_est_earth_model,
            BPSL1CoreProcessorStep.DOPPLER_RATE_ESTIMATOR: dc_est_earth_model,
            BPSL1CoreProcessorStep.RANGE_COMPENSATOR: EarthModel.WGS84,
            BPSL1CoreProcessorStep.AZIMUTH_FOCUSER: az_foc_earth_model,
        },
        prf_change_data_post_processing=True,
        apc_level=apc_level,
        elevation_mispointing_deg=aux_pp1_conf.antenna_pattern_correction.elevation_mispointing_bias,
        ionospheric_calibration_enabled=aux_pp1_conf.is_ionospheric_calibration_enabled(),
        rfi_use_chirp_product=rfi_use_chirp_product,
        rfi_operation_mode=rfi_operation_mode,
        drop_azimuth_focuser_margin=aux_pp1_conf.azimuth_compression.azimuth_focusing_margins_removal_flag,
    )


def convert_intermediate_list_to_core_outputs(
    intermediate_files: dict[IntermediateProductID, Path],
    steps: dict[BPSL1CoreProcessorStep, bool],
    earth_model: aux_pp1.GeneralConf.EarthModel,
) -> dict[BPSL1CoreProcessorProductId, Path]:
    """Convert intermediate list to core processor outputs"""

    def get_last_slc_product_id(
        steps: dict[BPSL1CoreProcessorStep, bool],
    ) -> BPSL1CoreProcessorProductId:
        """Get the last SLC product"""

        if steps.get(BPSL1CoreProcessorStep.RANGE_COMPENSATOR, None):
            return BPSL1CoreProcessorProductId.RANGE_COMPENSATOR
        if steps.get(BPSL1CoreProcessorStep.AZIMUTH_FOCUSER, None):
            return BPSL1CoreProcessorProductId.AZIMUTH_FOCUSER

        return BPSL1CoreProcessorProductId.RANGE_COMPENSATOR

    last_slc_product_id = get_last_slc_product_id(steps=steps)

    slant_dem_id = BPSL1CoreProcessorProductId.SAR_DEM_COPERNICUS
    if earth_model == aux_pp1.GeneralConf.EarthModel.SRTM:
        slant_dem_id = BPSL1CoreProcessorProductId.SAR_DEM_SRTM

    intermediate_id_to_product_id = {
        IntermediateProductID.RAW_MITIGATED: BPSL1CoreProcessorProductId.RFI_MITIGATION,
        IntermediateProductID.RFI_TIME_MASK: BPSL1CoreProcessorProductId.RFI_MITIGATION_TIME_MASK,
        IntermediateProductID.RFI_FREQ_MASK: BPSL1CoreProcessorProductId.RFI_MITIGATION_FREQUENCY_MASK,
        IntermediateProductID.RGC_DC_FR_ESTIMATOR: BPSL1CoreProcessorProductId.DOPPLER_RATE_ESTIMATOR,
        IntermediateProductID.DOPPLER_CENTROID_ESTIMATOR_GRID: BPSL1CoreProcessorProductId.DOPPLER_CENTROID_ESTIMATOR_GRID,
        IntermediateProductID.SLANT_DEM: slant_dem_id,
        IntermediateProductID.SLC: last_slc_product_id,
        IntermediateProductID.SLC_IONO_CORRECTED: BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR,
        IntermediateProductID.IONO_CAL_REPORT: BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR_IONO_REPORT,
        IntermediateProductID.FR: BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR_FR,
        IntermediateProductID.FR_PLANE: BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR_FR_PLANE,
        IntermediateProductID.PHASE_SCREEN_BB: BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR_PHASE_SCREEN_BB,
        # IntermediateProductID.PHASE_SCREEN_AF: BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR_PHASE_SCREEN_AF,
        IntermediateProductID.SRD_MULTILOOKED: BPSL1CoreProcessorProductId.MULTI_LOOKER,
        IntermediateProductID.SRD_DENOISED: BPSL1CoreProcessorProductId.DENOISER,
        IntermediateProductID.SLC_NESZ_MAP: BPSL1CoreProcessorProductId.NESZ_MAP_GENERATOR,
        IntermediateProductID.GRD: BPSL1CoreProcessorProductId.SLANT2_GROUND,
    }

    output_products = {
        intermediate_id_to_product_id[id]: path
        for id, path in intermediate_files.items()
        if intermediate_id_to_product_id.get(id, None)
    }

    return output_products


DEM_INDEX_NAME = "demIndex.xml"
GEOID_FILE_NAME = "egm2008-2.5.tif"


def retrieve_external_resources(
    dem_path: Path | None,
    earth_model: aux_pp1.GeneralConf.EarthModel,
    resampling_filter_product: Path | None = None,
) -> BPSL1CoreProcessingOptions.ExternalResources:
    """Retrieve processing options external resources"""

    if dem_path is not None:
        if earth_model == aux_pp1.GeneralConf.EarthModel.SRTM:
            dem_info = BPSL1CoreProcessingOptions.ExternalResources.DemInfo(
                earth_model=EarthModel.SRTM,
                entry_point=dem_path,
                geoid_file=None,
            )
        else:
            dem_info = BPSL1CoreProcessingOptions.ExternalResources.DemInfo(
                earth_model=EarthModel.COPERNICUS,
                entry_point=dem_path.joinpath(DEM_INDEX_NAME),
                geoid_file=dem_path.joinpath(GEOID_FILE_NAME),
            )
    else:
        dem_info = BPSL1CoreProcessingOptions.ExternalResources.DemInfo(
            earth_model=EarthModel.WGS84, entry_point="", geoid_file=None
        )

    return BPSL1CoreProcessingOptions.ExternalResources(
        dem_info_list=[dem_info],
        prf_resampling_filter_product=resampling_filter_product,
    )


def retrieve_interface_settings(
    remove_sarfoc_intermediates: bool,
) -> BPSL1CoreProcessingOptions.InterfaceSettings:
    """Retrieve interference settings"""

    return BPSL1CoreProcessingOptions.InterfaceSettings(
        products_format=ProductFormat.BIN,
        remove_intermediate_products=remove_sarfoc_intermediates,
    )


def fill_bps_l1_core_processor_processing_options(
    dem_path: Path | None,
    aux_pp1_conf: aux_pp1.AuxProcessingParametersL1,
    steps: dict[BPSL1CoreProcessorStep, bool],
    intermediate_files: dict[IntermediateProductID, Path],
    remove_sarfoc_intermediates: bool,
    resampling_filter_product: Path | None = None,
) -> BPSL1CoreProcessingOptions:
    """Fill BPS L1 core processor processing options structure from joborder.

    Parameters
    ----------
    dem_path : Optional[Path]
        Path to the dem folder
    aux_pp1_conf : aux_pp1.AuxPP1
        Aux PP1 object
    remove_sarfoc_intermediates : bool, optional
        automatic sarfoc intermediates cleaning

    Returns
    -------
    BPSL1CoreProcessingOptions
        processing options object
    """
    # Processing settings
    processing_settings = retrieve_processing_settings(aux_pp1_conf=aux_pp1_conf)

    # Output products
    output_products = convert_intermediate_list_to_core_outputs(
        intermediate_files, steps, aux_pp1_conf.general.height_model
    )

    # External resources
    external_resources = retrieve_external_resources(
        dem_path=dem_path,
        earth_model=aux_pp1_conf.general.height_model,
        resampling_filter_product=resampling_filter_product,
    )

    # Interface settings
    interface_settings = retrieve_interface_settings(remove_sarfoc_intermediates=remove_sarfoc_intermediates)

    return BPSL1CoreProcessingOptions(
        steps=steps,
        settings=processing_settings,
        output_products={k: str(v) for k, v in output_products.items()},
        external_resources=external_resources,
        interface_settings=interface_settings,
    )
