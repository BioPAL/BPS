# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 Post Processor interface
---------------------------
"""

from pathlib import Path

import numpy as np
from arepytools.geometry.generalsarattitude import create_general_sar_attitude
from arepytools.io import read_metadata
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common, common_types
from bps.l1_processor import BPS_L1_PROCESSOR_NAME
from bps.l1_processor import __version__ as VERSION
from bps.l1_processor.processor_interface.aux_pp1 import (
    AuxProcessingParametersL1,
    GeneralConf,
    L1ProductExportConf,
)
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder
from bps.l1_processor.settings.intermediate_names import IntermediateProductID
from bps.transcoder.io import common_annotation_l1
from bps.transcoder.io import common_annotation_models_l1 as main_annotation_models
from bps.transcoder.io.iono_cal_report import IonosphericCalibrationReport
from bps.transcoder.io.preprocessor_report import L1PreProcAnnotations
from bps.transcoder.sarproduct.biomass_l0product_reader import read_l0_product
from bps.transcoder.sarproduct.biomass_l1product import (
    AcquisitionInfo,
    BIOMASSL1ProcessingParameters,
    BIOMASSL1Product,
    BIOMASSL1ProductConfiguration,
    LUTParameters,
    QualityParameters,
    QuicklookParameters,
    RFIMasksStatistics,
    SARImageParameters,
)
from bps.transcoder.sarproduct.biomass_l1product_writer import BIOMASSL1ProductWriter
from bps.transcoder.sarproduct.dem_footprint_utils import (
    compute_footprint_from_dem_lut,
    compute_gcp_from_dem_lut,
)
from bps.transcoder.sarproduct.generic_product import GenericProduct
from bps.transcoder.sarproduct.l1_annotations import DCAnnotations
from bps.transcoder.sarproduct.l1_lut_writer import ProductLUTID
from bps.transcoder.sarproduct.navigation_files_utils import (
    fill_attitude_file_template_str,
    fill_orbit_file_template_str,
)
from bps.transcoder.sarproduct.sarproduct import SARProduct
from bps.transcoder.utils import quicklook_utils

TAI_UTC = 37


def translate_earth_model(
    earth_model: GeneralConf.EarthModel,
) -> common.HeightModelBaseType:
    """Convert Earth model from AuxPP1 to height model of the transcoder"""
    if earth_model == GeneralConf.EarthModel.COPERNICUS_DEM:
        return common.HeightModelBaseType.COPERNICUS_DEM
    if earth_model == GeneralConf.EarthModel.ELLIPSOID:
        return common.HeightModelBaseType.ELLIPSOID
    if earth_model == GeneralConf.EarthModel.SRTM:
        return common.HeightModelBaseType.SRTM

    raise RuntimeError(f"Current earth model not supported in l1 output writer {earth_model}")


def fill_bps_l1_processing_parameters(
    aux_pp1: AuxProcessingParametersL1,
) -> BIOMASSL1ProcessingParameters:
    """_summary_

    Parameters
    ----------
    aux_pp1 : AuxProcessingParametersL1
        _description_

    Returns
    -------
    BIOMASSL1ProcessingParameters
        _description_
    """
    internal_calibration_estimation_flag = aux_pp1.l0_product_import.internal_calibration_estimation_flag
    internal_calibration_correction_flag = aux_pp1.internal_calibration_correction.internal_calibration_correction_flag
    range_spreading_loss_compensation_flag = aux_pp1.radiometric_calibration.range_spreading_loss_compensation_enabled
    ionospheric_phase_screen_correction_flag = aux_pp1.ionosphere_calibration.ionospheric_phase_screen_correction_flag

    assert aux_pp1.rfi_mitigation.activation_mode in ("Enabled", "Disabled")
    processing_parameters = BIOMASSL1ProcessingParameters(
        raw_data_correction_flag=aux_pp1.raw_data_correction.raw_data_correction_flag,
        rfi_detection_flag=aux_pp1.rfi_mitigation.detection_flag,
        rfi_correction_flag=aux_pp1.rfi_mitigation.activation_mode == "Enabled",
        rfi_mitigation_method=common.RfiMitigationMethodType[aux_pp1.rfi_mitigation.mitigation_method.name],
        rfi_mask=common.RfiMaskType[aux_pp1.rfi_mitigation.mask.name],
        rfi_mask_generation_method=common.RfiMaskGenerationMethodType[
            aux_pp1.rfi_mitigation.mask_generation_method.name
        ],
        rfi_fm_mitigation_method=aux_pp1.rfi_mitigation.freq_domain_processing_parameters.mitigation_method,
        rfi_fm_chirp_source=common.RangeReferenceFunctionType[
            aux_pp1.rfi_mitigation.freq_domain_processing_parameters.chirp_source.name
        ],
        internal_calibration_estimation_flag=internal_calibration_estimation_flag,
        internal_calibration_correction_flag=internal_calibration_correction_flag,
        internal_calibration_source=aux_pp1.internal_calibration_correction.internal_calibration_source,
        range_reference_function_source=common.RangeReferenceFunctionType[
            aux_pp1.range_compression.range_reference_function_source.name
        ],
        range_compression_method=common.RangeCompressionMethodType[
            aux_pp1.range_compression.range_compression_method.name
        ],
        range_window_type={
            k.value: main_annotation_models.WeightingWindowType[v.window_type.name]
            for k, v in aux_pp1.range_compression.parameters.items()
        },
        range_window_coefficient={
            k.value: v.window_coefficient for k, v in aux_pp1.range_compression.parameters.items()
        },
        extended_swath_processing=aux_pp1.range_compression.extended_swath_processing,
        dc_method=common.DcMethodType[aux_pp1.doppler_estimation.method.name],
        dc_value=aux_pp1.doppler_estimation.value,
        antenna_pattern_correction1_flag=aux_pp1.antenna_pattern_correction.antenna_pattern_correction1_flag,
        antenna_pattern_correction2_flag=aux_pp1.antenna_pattern_correction.antenna_pattern_correction2_flag,
        antenna_cross_talk_correction_flag=aux_pp1.antenna_pattern_correction.antenna_cross_talk_correction_flag,
        azimuth_compression_block_samples=aux_pp1.azimuth_compression.block_samples,
        azimuth_compression_block_lines=aux_pp1.azimuth_compression.block_lines,
        azimuth_compression_block_overlap_samples=aux_pp1.azimuth_compression.block_overlap_samples,
        azimuth_compression_block_overlap_lines=aux_pp1.azimuth_compression.block_overlap_lines,
        azimuth_window_type={
            k.value: main_annotation_models.WeightingWindowType[v.window_type.name]
            for k, v in aux_pp1.azimuth_compression.parameters.items()
        },
        azimuth_window_coefficient={
            k.value: v.window_coefficient for k, v in aux_pp1.azimuth_compression.parameters.items()
        },
        bistatic_delay_correction_flag=aux_pp1.azimuth_compression.bistatic_delay_correction,
        bistatic_delay_correction_method=common.BistaticDelayCorrectionMethodType[
            aux_pp1.azimuth_compression.bistatic_delay_correction_method.name
        ],
        azimuth_focusing_margins_removal_flag=aux_pp1.azimuth_compression.azimuth_focusing_margins_removal_flag,
        range_spreading_loss_compensation_flag=range_spreading_loss_compensation_flag,
        reference_range=aux_pp1.radiometric_calibration.reference_range,
        polarimetric_correction_flag=aux_pp1.polarimetric_calibration.polarimetric_correction_flag,
        ionosphere_height_defocusing_flag=aux_pp1.ionosphere_calibration.ionosphere_height_defocusing_flag,
        ionosphere_height_estimation_method=common.IonosphereHeightEstimationMethodType[
            aux_pp1.ionosphere_calibration.ionosphere_height_estimation_method.name
        ],
        faraday_rotation_correction_flag=aux_pp1.ionosphere_calibration.faraday_rotation_correction_flag,
        ionospheric_phase_screen_correction_flag=ionospheric_phase_screen_correction_flag,
        group_delay_correction_flag=aux_pp1.ionosphere_calibration.group_delay_correction_flag,
        autofocus_flag=aux_pp1.autofocus.autofocus_flag,
        autofocus_method=common.AutofocusMethodType[aux_pp1.autofocus.autofocus_method.name],
        range_upsampling_factor={k.value: v.upsampling_factor for k, v in aux_pp1.multilook.range_parameters.items()},
        range_downsampling_factor={
            k.value: v.downsampling_factor for k, v in aux_pp1.multilook.range_parameters.items()
        },
        azimuth_upsampling_factor={
            k.value: v.upsampling_factor for k, v in aux_pp1.multilook.azimuth_parameters.items()
        },
        azimuth_downsampling_factor={
            k.value: v.downsampling_factor for k, v in aux_pp1.multilook.azimuth_parameters.items()
        },
        detection_flag=aux_pp1.multilook.apply_detection,
        thermal_denoising_flag=aux_pp1.thermal_denoising.thermal_denoising_flag,
        noise_parameters_source=aux_pp1.thermal_denoising.noise_parameters_source,
        ground_projection_flag=aux_pp1.ground_projection.ground_projection_flag,
        requested_height_model=translate_earth_model(aux_pp1.general.requested_height_model),
        requested_height_model_version=aux_pp1.general.requested_height_model_version,
        requested_height_model_used=aux_pp1.general.requested_height_model == aux_pp1.general.height_model,
        absolute_calibration_constants={
            common.PolarisationType(pol.name): value
            for pol, value in aux_pp1.radiometric_calibration.absolute_calibration_constant.items()
        },
        polarimetric_distortion=common_annotation_l1.PolarimetricDistortionType(
            aux_pp1.polarimetric_calibration.cross_talk,
            aux_pp1.polarimetric_calibration.channel_imbalance,
        ),
    )
    return processing_parameters


def fill_acquisition_info(raw_metadata: Path) -> AcquisitionInfo:
    """Fill acquisition info"""
    metadata = read_metadata(raw_metadata)
    time_line = metadata.get_acquisition_time_line()
    _, prf_times, prf_values = time_line.prf_changes
    _, swst_times, swst_values = time_line.swst_changes
    _, swl_times, swl_values = time_line.swl_changes

    reference_time = metadata.get_raster_info().lines_start

    def update_time(times: list[float]) -> list[PreciseDateTime]:
        return [time + reference_time for time in times]

    prf = list(zip(update_time(prf_times), prf_values))
    swst = list(zip(update_time(swst_times), swst_values))
    swl = list(zip(update_time(swl_times), swl_values))
    return AcquisitionInfo(swp_list=swst, swl_list=swl, prf_list=prf)


def fill_sar_image_parameters(
    aux_pp1: AuxProcessingParametersL1,
) -> SARImageParameters:
    """_summary_

    Parameters
    ----------
    aux_pp1 : AuxProcessingParametersL1
        _description_

    Returns
    -------
    SARImageParameters
        _description_
    """
    sar_image_parameters = SARImageParameters(
        pixel_representation=aux_pp1.l1_product_export.pixel_representation,
        pixel_type=main_annotation_models.PixelTypeType.VALUE_32_BIT_FLOAT,
        abs_compression_method=common_types.CompressionMethodType[
            aux_pp1.l1_product_export.abs_compression_method.name
        ],
        abs_max_z_error=aux_pp1.l1_product_export.abs_max_zerror,
        abs_max_z_error_percentile=aux_pp1.l1_product_export.abs_max_zerror_percentile,
        phase_compression_method=common_types.CompressionMethodType[
            aux_pp1.l1_product_export.phase_compression_method.name
        ],
        phase_max_z_error=aux_pp1.l1_product_export.phase_max_zerror,
        phase_max_z_error_percentile=aux_pp1.l1_product_export.phase_max_zerror_percentile,
        no_pixel_value=aux_pp1.l1_product_export.no_pixel_value,
        block_size=aux_pp1.l1_product_export.block_size,
    )
    return sar_image_parameters


def fill_lut_parameters(
    aux_pp1: AuxProcessingParametersL1,
) -> LUTParameters:
    """_summary_

    Parameters
    ----------
    aux_pp1 : AuxProcessingParametersL1
        _description_

    Returns
    -------
    LUTParameters
        _description_
    """

    def translate_decimation_factors(
        factors: L1ProductExportConf.LutDecimationFactors,
    ) -> LUTParameters.LutDecimationFactors:
        """Translate decimation factors"""
        return LUTParameters.LutDecimationFactors(
            dem_based_quantity=factors.dem_based_quantity,
            rfi_based_quantity=factors.rfi_based_quantity,
            image_based_quantity=factors.image_based_quantity,
        )

    lut_parameters = LUTParameters(
        lut_range_decimation_factors=translate_decimation_factors(
            aux_pp1.l1_product_export.lut_range_decimation_factor
        ),
        lut_azimuth_decimation_factors=translate_decimation_factors(
            aux_pp1.l1_product_export.lut_azimuth_decimation_factor
        ),
        lut_block_size=aux_pp1.l1_product_export.lut_block_size,
        lut_layers_completeness_flag=aux_pp1.l1_product_export.lut_layers_completeness_flag,
        no_pixel_value=aux_pp1.l1_product_export.no_pixel_value,
    )
    return lut_parameters


def fill_quicklook_parameters(
    aux_pp1: AuxProcessingParametersL1,
) -> QuicklookParameters:
    """_summary_

    Parameters
    ----------
    aux_pp1 : AuxProcessingParametersL1
        _description_

    Returns
    -------
    QuicklookParameters
        _description_
    """
    quicklook_parameters = QuicklookParameters(
        ql_range_decimation_factor=aux_pp1.l1_product_export.ql_range_decimation_factor,
        ql_range_averaging_factor=aux_pp1.l1_product_export.ql_range_averaging_factor,
        ql_azimuth_decimation_factor=aux_pp1.l1_product_export.ql_azimuth_decimation_factor,
        ql_azimuth_averaging_factor=aux_pp1.l1_product_export.ql_azimuth_averaging_factor,
        ql_absolute_scaling_factor={
            k.value: v for k, v in aux_pp1.l1_product_export.ql_absolute_scaling_factor_list.items()
        },
    )
    return quicklook_parameters


def fill_quality_parameters(aux_pp1: AuxProcessingParametersL1):
    """Fill quality parameters"""
    return QualityParameters(
        max_isp_gap=aux_pp1.l0_product_import.max_isp_gap,
        raw_mean_expected=aux_pp1.l0_product_import.raw_mean_expected,
        raw_mean_threshold=aux_pp1.l0_product_import.raw_mean_threshold,
        raw_std_expected=aux_pp1.l0_product_import.raw_std_expected,
        raw_std_threshold=aux_pp1.l0_product_import.raw_std_threshold,
        max_rfi_tm_percentage=aux_pp1.rfi_mitigation.time_domain_processing_parameters.max_rfi_percentage,
        max_rfi_fm_percentage=aux_pp1.rfi_mitigation.freq_domain_processing_parameters.max_rfi_percentage,
        max_drift_amplitude_std_fraction=aux_pp1.internal_calibration_correction.max_drift_amplitude_std_fraction,
        max_drift_phase_std_fraction=aux_pp1.internal_calibration_correction.max_drift_phase_std_fraction,
        max_drift_amplitude_error=aux_pp1.internal_calibration_correction.max_drift_amplitude_error,
        max_drift_phase_error=aux_pp1.internal_calibration_correction.max_drift_phase_error,
        max_invalid_drift_fraction=aux_pp1.internal_calibration_correction.max_invalid_drift_fraction,
        dc_rmserror_threshold=aux_pp1.doppler_estimation.rms_error_threshold,
    )


def get_acquisition_raster_info(raw_metadata: Path):
    return read_metadata(raw_metadata).get_raster_info()


def fill_bps_transcoder_configuration(
    job_order: L1JobOrder,
    aux_pp1: AuxProcessingParametersL1,
    reference_raw_metadata: Path | None = None,
) -> BIOMASSL1ProductConfiguration:
    """_summary_

    Parameters
    ----------
    job_order : L1JobOrder
        _description_
    aux_pp1 : AuxProcessingParametersL1
        _description_

    Returns
    -------
    BIOMASSL1ProductConfiguration
        _description_
    """
    return BIOMASSL1ProductConfiguration(
        l1a_doi=aux_pp1.l1_product_export.l1a_product_doi,
        l1b_doi=aux_pp1.l1_product_export.l1b_product_doi,
        frame_id=job_order.processing_parameters.frame_id,
        frame_status=job_order.processing_parameters.frame_status,
        product_baseline=job_order.io_products.output.output_baseline,
        processing_parameters=fill_bps_l1_processing_parameters(aux_pp1),
        acquisition_raster_info=(
            get_acquisition_raster_info(reference_raw_metadata) if reference_raw_metadata else None
        ),
        acquisition_timeline=(fill_acquisition_info(reference_raw_metadata) if reference_raw_metadata else None),
        sar_image_parameters=fill_sar_image_parameters(aux_pp1),
        lut_parameters=fill_lut_parameters(aux_pp1),
        quicklook_parameters=fill_quicklook_parameters(aux_pp1),
        quality_parameters=fill_quality_parameters(aux_pp1),
    )


def read_intermediate_products(
    products_path_dict: dict[ProductLUTID, Path],
) -> dict[ProductLUTID, GenericProduct | None]:
    """Read intermediate products"""
    return {
        product_id: (GenericProduct.read_from_product_path(product_path) if product_path.is_dir() else None)
        for product_id, product_path in products_path_dict.items()
    }


def decimate_product_lut(
    products: dict[ProductLUTID, GenericProduct],
    range_decimation_factor_list: LUTParameters.LutDecimationFactors,
    azimuth_decimation_factor_list: LUTParameters.LutDecimationFactors,
) -> None:
    """Apply decimation factor to available LUTs"""

    def get_factor_by_product_id(product_id: ProductLUTID, factors: LUTParameters.LutDecimationFactors) -> int:
        """Return decimation factors"""
        mapping = {
            ProductLUTID.SAR_DEM: factors.dem_based_quantity,
            ProductLUTID.SLC_NESZ_MAP: factors.image_based_quantity,
            ProductLUTID.FR: factors.image_based_quantity,
            ProductLUTID.FR_PLANE: factors.image_based_quantity,
            ProductLUTID.PHASE_SCREEN_AF: factors.image_based_quantity,
            ProductLUTID.PHASE_SCREEN_BB: factors.image_based_quantity,
            ProductLUTID.RFI_TIME_MASK: factors.rfi_based_quantity,
            ProductLUTID.RFI_FREQ_MASK: factors.rfi_based_quantity,
        }

        return mapping[product_id]

    def subsample(
        data_list: list[np.ndarray],
        samples_axes: list[np.ndarray],
        azimuth_axes: list[np.ndarray],
        range_decimation_factor: int,
        azimuth_decimation_factor: int,
    ):
        for channel, (data, samples_axis, lines_axis) in enumerate(
            zip(
                data_list,
                samples_axes,
                azimuth_axes,
            )
        ):
            data_list[channel] = data[::azimuth_decimation_factor, ::range_decimation_factor]
            samples_axes[channel] = samples_axis[::range_decimation_factor]
            azimuth_axes[channel] = lines_axis[::azimuth_decimation_factor]

    for product_id, value in products.items():
        range_decimation = get_factor_by_product_id(product_id, range_decimation_factor_list)
        azimuth_decimation = get_factor_by_product_id(product_id, azimuth_decimation_factor_list)
        subsample(
            value.data_list,
            value.samples_axis_list,
            value.lines_axis_list,
            range_decimation,
            azimuth_decimation,
        )


def export_in_bps_format(
    input_product_path: Path,
    source_product_path: Path,
    source_monitoring_product_path: Path | None,
    auxiliary_files: list[Path],
    intermediate_products_dict: dict[str, Path],
    output_folder: Path,
    configuration: BIOMASSL1ProductConfiguration,
    l1_pre_proc_report: L1PreProcAnnotations,
    rfi_masks_statistics: RFIMasksStatistics,
    dc_annotations: DCAnnotations | None = None,
    dc_fallback_activated: bool = False,
    add_monitoring_product: bool = False,
    calibration_tag: str | None = None,
    gdal_num_threads: int = 1,
    iono_cal_report: IonosphericCalibrationReport | None = None,
):
    """_summary_

    Parameters
    ----------
    input_product_path : Path
        Path to input product (internal format)
    source_product_path : Path
        Path to product used to generate input one
    source_monitoring_product_path : Path, optional
        Path to monitoring product used to generate input one
    auxiliary_files_dict : Path
        Path to auxiliary files used to generate input one
    intermediate_products_path : Path
        Path to intermediate products associated to input one
    output_folder : Path
        Path to folder where to store output products (BPS format)
    configuration : BIOMASSL1ProductConfiguration
        Transcoder configuration
    add_monitoring_product : bool, optional
        Flag to export also monitoring product, by default False
    parc_scattering_response : Optional[ScatteringResponse]
        Scattering response for proper filenames of the PARC processing
        output (unspecified for normal processing)
    """
    bps_logger.info(
        f"Saving {input_product_path} in BPS format to folder {output_folder}"
        + (" (with monitoring)" if add_monitoring_product else "")
    )

    # Read input product
    input_product = SARProduct.read(product_path=input_product_path)

    # Prepare orbit template string
    reference_orbit = input_product.general_sar_orbit[0]
    orbit_xml_template_string = fill_orbit_file_template_str(
        position=reference_orbit.position_vector,
        velocity=reference_orbit.velocity_vector,
        start_time=reference_orbit.reference_time,
        time_step=reference_orbit.time_step,
        tai_utc_difference=TAI_UTC,
    )

    # Prepare attitude template string
    reference_attitude = create_general_sar_attitude(
        input_product.general_sar_orbit[0],
        input_product.attitude_info[0],
        ignore_anx_after_orbit_start=True,
    )
    attitude_xml_template_string = fill_attitude_file_template_str(
        arf=reference_attitude.get_arf(reference_attitude.time_axis_array),
        time_array=reference_attitude.time_axis_array,
    )

    # Read source product
    source_product = read_l0_product(source_product_path)

    source_monitoring_product = (
        read_l0_product(source_monitoring_product_path) if source_monitoring_product_path else None
    )

    source_auxiliary_names = [file.name for file in auxiliary_files if file.is_dir()]

    # Read intermediate products
    intermediate_id_to_lut_id: dict[str, ProductLUTID] = {
        IntermediateProductID.SLANT_DEM.name: ProductLUTID.SAR_DEM,
        IntermediateProductID.SLC_NESZ_MAP.name: ProductLUTID.SLC_NESZ_MAP,
    }
    if configuration.lut_parameters.lut_layers_completeness_flag:
        intermediate_id_to_lut_id[IntermediateProductID.RFI_TIME_MASK.name] = ProductLUTID.RFI_TIME_MASK
        intermediate_id_to_lut_id[IntermediateProductID.RFI_FREQ_MASK.name] = ProductLUTID.RFI_FREQ_MASK
        intermediate_id_to_lut_id[IntermediateProductID.FR.name] = ProductLUTID.FR
        intermediate_id_to_lut_id[IntermediateProductID.FR_PLANE.name] = ProductLUTID.FR_PLANE
        intermediate_id_to_lut_id[IntermediateProductID.PHASE_SCREEN_BB.name] = ProductLUTID.PHASE_SCREEN_BB
        intermediate_id_to_lut_id[IntermediateProductID.PHASE_SCREEN_AF.name] = ProductLUTID.PHASE_SCREEN_AF

    intermediate_lut_products_dict = {
        intermediate_id_to_lut_id[id]: path
        for id, path in intermediate_products_dict.items()
        if id in intermediate_id_to_lut_id
    }

    product_lut = read_intermediate_products(intermediate_lut_products_dict)

    dem_lut = product_lut[ProductLUTID.SAR_DEM]
    assert dem_lut is not None

    # No need to update the current footprint in the ellipsoid case
    if (
        configuration.processing_parameters.requested_height_model_used
        and configuration.processing_parameters.requested_height_model != common.HeightModelBaseType.ELLIPSOID
    ):
        input_product.footprint = compute_footprint_from_dem_lut(input_product, dem_lut)
        input_product.gcp_list = compute_gcp_from_dem_lut(input_product, dem_lut)

    products_to_subsample = {key: value for key, value in product_lut.items() if value is not None}

    # In the ellipsoid case the LUT is already sampled
    if (
        not configuration.processing_parameters.requested_height_model_used
        or configuration.processing_parameters.requested_height_model == common.HeightModelBaseType.ELLIPSOID
    ):
        del products_to_subsample[ProductLUTID.SAR_DEM]

    decimate_product_lut(
        products_to_subsample,
        configuration.lut_parameters.lut_range_decimation_factors,
        configuration.lut_parameters.lut_azimuth_decimation_factors,
    )

    # Export standard product
    output_standard = BIOMASSL1Product(
        product=input_product,
        is_monitoring=False,
        source=source_product,
        source_monitoring=source_monitoring_product,
        source_auxiliary_names=source_auxiliary_names,
        configuration=configuration,
        calibration_tag=calibration_tag,
        l1_pre_proc_report=l1_pre_proc_report,
        rfi_masks_statistics=rfi_masks_statistics,
        dc_annotations=dc_annotations,
        dc_fallback_activated=dc_fallback_activated,
        iono_cal_report=iono_cal_report,
    )
    standard_bps_product_writer = BIOMASSL1ProductWriter(
        product=output_standard,
        product_path=output_folder,
        product_lut=product_lut,
        processor_name=BPS_L1_PROCESSOR_NAME,
        processor_version=VERSION,
    )

    # Write quicklook to standard product
    standard_quicklook_file = standard_bps_product_writer.product_path.joinpath(
        standard_bps_product_writer.content.quicklook
    )
    quicklook_conf = quicklook_utils.QuickLookConf(
        range_decimation_factor=configuration.quicklook_parameters.ql_range_decimation_factor,
        range_averaging_factor=configuration.quicklook_parameters.ql_range_averaging_factor,
        azimuth_decimation_factor=configuration.quicklook_parameters.ql_azimuth_decimation_factor
        if input_product.type == "SLC"
        else configuration.quicklook_parameters.ql_range_decimation_factor,
        azimuth_averaging_factor=configuration.quicklook_parameters.ql_azimuth_averaging_factor
        if input_product.type == "SLC"
        else configuration.quicklook_parameters.ql_range_averaging_factor,
        absolute_scaling_factor=configuration.quicklook_parameters.ql_absolute_scaling_factor,
    )

    quicklook_rgb = quicklook_utils.compute_quicklook_from_pol_data(
        dict(
            zip(
                standard_bps_product_writer.product.polarization_list,
                standard_bps_product_writer.product.data_list,
            )
        ),
        quicklook_conf,
    )

    assert standard_quicklook_file is not None, "Quicklook file is None"
    quicklook_utils.write_quicklook_to_file(quicklook_rgb, Path(standard_quicklook_file))

    # Write standard product
    standard_bps_product_writer.write(orbit_xml_template_string, attitude_xml_template_string)

    # If required, export monitoring product too
    if add_monitoring_product:
        output_monitoring = BIOMASSL1Product(
            product=input_product,
            is_monitoring=True,
            source=source_product,
            source_monitoring=source_monitoring_product,
            source_auxiliary_names=source_auxiliary_names,
            configuration=configuration,
            calibration_tag=calibration_tag,
            l1_pre_proc_report=l1_pre_proc_report,
            rfi_masks_statistics=rfi_masks_statistics,
            dc_annotations=dc_annotations,
            dc_fallback_activated=dc_fallback_activated,
            iono_cal_report=iono_cal_report,
        )
        BIOMASSL1ProductWriter(
            product=output_monitoring,
            product_path=output_folder,
            product_lut=product_lut,
            processor_name=BPS_L1_PROCESSOR_NAME,
            processor_version=VERSION,
            gdal_num_threads=gdal_num_threads,
        ).write(
            orbit_xml_template_string,
            attitude_xml_template_string,
            Path(standard_quicklook_file),
        )
