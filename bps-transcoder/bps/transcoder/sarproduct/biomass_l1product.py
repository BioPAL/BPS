# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""L1ab product"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from arepytools.io import metadata
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common, common_types
from bps.transcoder.io import common_annotation_l1
from bps.transcoder.io import common_annotation_models_l1 as main_annotation_models
from bps.transcoder.io.common_annotation_l1 import PolarimetricDistortionType
from bps.transcoder.io.iono_cal_report import IonosphericCalibrationReport
from bps.transcoder.io.preprocessor_report import L1PreProcAnnotations
from bps.transcoder.sarproduct.biomass_l0product import BIOMASSL0Product
from bps.transcoder.sarproduct.l1_annotations import DCAnnotations, RFIMasksStatistics
from bps.transcoder.sarproduct.sarproduct import SARProduct
from bps.transcoder.utils.production_model_utils import encode_product_name_id_value
from bps.transcoder.utils.time_conversions import (
    pdt_to_compact_date,
    pdt_to_compact_string,
)

# Type conversions between L1 products and SAR products.
SAR_TO_L1A_PRODUCT_TYPE = {"SLC": "SCS", "GRD": "DGM"}


@dataclass
class BIOMASSL1ProcessingParameters:
    """BPS L1 processsing parameters"""

    raw_data_correction_flag: bool
    rfi_detection_flag: bool
    rfi_correction_flag: bool
    rfi_mitigation_method: common.RfiMitigationMethodType
    rfi_mask: common.RfiMaskType
    rfi_mask_generation_method: common.RfiMaskGenerationMethodType
    rfi_fm_mitigation_method: Literal["NOTCH_FILTER", "NEAREST_NEIGHBOUR_INTERPOLATION"]
    rfi_fm_chirp_source: common.RangeReferenceFunctionType
    internal_calibration_estimation_flag: bool
    internal_calibration_correction_flag: bool
    internal_calibration_source: common.InternalCalibrationSourceType
    range_reference_function_source: common.RangeReferenceFunctionType
    range_compression_method: common.RangeCompressionMethodType
    range_window_type: dict
    range_window_coefficient: dict
    extended_swath_processing: bool
    dc_method: common.DcMethodType
    dc_value: float
    antenna_pattern_correction1_flag: bool
    antenna_pattern_correction2_flag: bool
    antenna_cross_talk_correction_flag: bool
    azimuth_compression_block_samples: int
    azimuth_compression_block_lines: int
    azimuth_compression_block_overlap_samples: int
    azimuth_compression_block_overlap_lines: int
    azimuth_window_type: dict
    azimuth_window_coefficient: dict
    bistatic_delay_correction_flag: bool
    bistatic_delay_correction_method: common.BistaticDelayCorrectionMethodType
    azimuth_focusing_margins_removal_flag: bool
    range_spreading_loss_compensation_flag: bool
    reference_range: float
    polarimetric_correction_flag: bool
    ionosphere_height_defocusing_flag: bool
    ionosphere_height_estimation_method: common.IonosphereHeightEstimationMethodType
    faraday_rotation_correction_flag: bool
    ionospheric_phase_screen_correction_flag: bool
    group_delay_correction_flag: bool
    autofocus_flag: bool
    autofocus_method: common.AutofocusMethodType
    range_upsampling_factor: dict
    range_downsampling_factor: dict
    azimuth_upsampling_factor: dict
    azimuth_downsampling_factor: dict
    detection_flag: bool
    thermal_denoising_flag: bool
    noise_parameters_source: common.InternalCalibrationSourceType
    ground_projection_flag: bool
    requested_height_model: common.HeightModelBaseType
    requested_height_model_version: str
    requested_height_model_used: bool
    absolute_calibration_constants: dict[common.PolarisationType, float]
    polarimetric_distortion: PolarimetricDistortionType

    @classmethod
    def from_l1_annotation(cls, params: common_annotation_l1.ProcessingParameters) -> BIOMASSL1ProcessingParameters:
        """Populate the processing parmeters from L1 annotation."""
        default_params = _default_processing_parameters()
        return cls(
            raw_data_correction_flag=params.raw_data_correction_flag,
            rfi_detection_flag=params.rfi_detection_flag,
            rfi_correction_flag=params.rfi_correction_flag,
            rfi_mitigation_method=params.rfi_mitigation_method,
            rfi_mask_generation_method=params.rfi_mask_generation_method,
            rfi_mask=params.rfi_mask,
            rfi_fm_mitigation_method=params.rfi_fm_mitigation_method,
            rfi_fm_chirp_source=params.rfi_fm_chirp_source,
            internal_calibration_estimation_flag=params.internal_calibration_estimation_flag,
            internal_calibration_correction_flag=params.internal_calibration_correction_flag,
            internal_calibration_source=default_params.internal_calibration_source,
            range_reference_function_source=params.range_reference_function_source,
            range_compression_method=params.range_compression_method,
            range_window_type=params.range_processing_parameters.window_type,
            range_window_coefficient=params.range_processing_parameters.window_coefficient,
            extended_swath_processing=params.extended_swath_processing_flag,
            dc_method=params.dc_method,
            dc_value=params.dc_value,
            antenna_pattern_correction1_flag=params.antenna_pattern_correction1_flag,
            antenna_pattern_correction2_flag=params.antenna_pattern_correction2_flag,
            antenna_cross_talk_correction_flag=params.antenna_cross_talk_correction_flag,
            azimuth_compression_block_samples=256,
            azimuth_compression_block_lines=16700,
            azimuth_compression_block_overlap_samples=46,
            azimuth_compression_block_overlap_lines=4076,
            azimuth_window_type=params.azimuth_processing_parameters.window_type,
            azimuth_window_coefficient=params.azimuth_processing_parameters.window_coefficient,
            bistatic_delay_correction_flag=params.bistatic_delay_correction_flag,
            bistatic_delay_correction_method=params.bistatic_delay_correction_method,
            azimuth_focusing_margins_removal_flag=default_params.azimuth_focusing_margins_removal_flag,
            range_spreading_loss_compensation_flag=params.range_spreading_loss_compensation_flag,
            reference_range=params.reference_range,
            polarimetric_correction_flag=params.polarimetric_correction_flag,
            ionosphere_height_defocusing_flag=params.ionosphere_height_defocusing_flag,
            ionosphere_height_estimation_method=params.ionosphere_height_estimation_method,
            faraday_rotation_correction_flag=params.faraday_rotation_correction_flag,
            ionospheric_phase_screen_correction_flag=params.ionospheric_phase_screen_correction_flag,
            group_delay_correction_flag=params.group_delay_correction_flag,
            autofocus_flag=params.autofocus_flag,
            autofocus_method=params.autofocus_method,
            range_upsampling_factor=default_params.range_upsampling_factor,
            range_downsampling_factor=default_params.range_downsampling_factor,
            azimuth_upsampling_factor=default_params.azimuth_upsampling_factor,
            azimuth_downsampling_factor=default_params.azimuth_downsampling_factor,
            detection_flag=params.detection_flag,
            thermal_denoising_flag=params.thermal_denoising_flag,
            noise_parameters_source=default_params.noise_parameters_source,
            ground_projection_flag=params.ground_projection_flag,
            requested_height_model=default_params.requested_height_model,
            requested_height_model_version=default_params.requested_height_model_version,
            requested_height_model_used=default_params.requested_height_model_used,
            absolute_calibration_constants=default_params.absolute_calibration_constants,
            polarimetric_distortion=default_params.polarimetric_distortion,
        )


@dataclass
class AcquisitionInfo:
    """Acquition information"""

    swp_list: list[tuple[PreciseDateTime, float]]
    swl_list: list[tuple[PreciseDateTime, float]]
    prf_list: list[tuple[PreciseDateTime, float]]


@dataclass
class SARImageParameters:
    """SAR Image parameters"""

    pixel_representation: common.PixelRepresentationType
    pixel_type: main_annotation_models.PixelTypeType
    abs_compression_method: common_types.CompressionMethodType
    abs_max_z_error: float
    abs_max_z_error_percentile: float
    phase_compression_method: common_types.CompressionMethodType
    phase_max_z_error: float
    phase_max_z_error_percentile: float
    no_pixel_value: float
    block_size: int

    @classmethod
    def from_l1_annotation(cls, params) -> SARImageParameters:
        """ """
        default_params = _default_sar_image_parameters()
        return cls(
            pixel_representation=params.pixel_representation,
            pixel_type=params.pixel_type,
            abs_compression_method=default_params.abs_compression_method,
            abs_max_z_error=default_params.abs_max_z_error,
            abs_max_z_error_percentile=default_params.abs_max_z_error_percentile,
            no_pixel_value=params.no_data_value,
            phase_compression_method=default_params.phase_compression_method,
            phase_max_z_error=default_params.phase_max_z_error,
            phase_max_z_error_percentile=default_params.phase_max_z_error_percentile,
            block_size=default_params.block_size,
        )


@dataclass
class LUTParameters:
    """_summary_"""

    @dataclass
    class LutDecimationFactors:
        """LUT factors by group"""

        dem_based_quantity: int
        rfi_based_quantity: int
        image_based_quantity: int

    lut_range_decimation_factors: LutDecimationFactors
    lut_azimuth_decimation_factors: LutDecimationFactors
    lut_block_size: int
    lut_layers_completeness_flag: bool
    no_pixel_value: float


@dataclass
class QuicklookParameters:
    """Quicklook parameters"""

    ql_range_decimation_factor: int
    ql_range_averaging_factor: int
    ql_azimuth_decimation_factor: int
    ql_azimuth_averaging_factor: int
    ql_absolute_scaling_factor: dict


@dataclass
class QualityParameters:
    """Quality parameters"""

    max_isp_gap: int
    raw_mean_expected: float
    raw_mean_threshold: float
    raw_std_expected: float
    raw_std_threshold: float
    max_rfi_tm_percentage: float
    max_rfi_fm_percentage: float
    max_drift_amplitude_std_fraction: float
    max_drift_phase_std_fraction: float
    max_drift_amplitude_error: float
    max_drift_phase_error: float
    max_invalid_drift_fraction: float
    dc_rmserror_threshold: float


@dataclass
class BIOMASSL1ProductConfiguration:
    """BPS L1 product configuration"""

    l1a_doi: str
    l1b_doi: str
    frame_id: int
    frame_status: str
    product_baseline: int
    acquisition_raster_info: metadata.RasterInfo | None
    acquisition_timeline: AcquisitionInfo | None
    processing_parameters: BIOMASSL1ProcessingParameters
    sar_image_parameters: SARImageParameters
    lut_parameters: LUTParameters | None
    quicklook_parameters: QuicklookParameters | None
    quality_parameters: QualityParameters


class BIOMASSL1Product(SARProduct):
    """BPS L1ab product"""

    def __init__(
        self,
        product: SARProduct | None = None,
        is_monitoring: bool = False,
        source: BIOMASSL0Product | SARProduct | None = None,
        source_monitoring: BIOMASSL0Product | None = None,
        source_auxiliary_names: list[str] | None = None,
        configuration: BIOMASSL1ProductConfiguration | None = None,
        calibration_tag: str | None = None,
        l1_pre_proc_report: L1PreProcAnnotations | None = None,
        rfi_masks_statistics: RFIMasksStatistics | None = None,
        dc_annotations: DCAnnotations | None = None,
        dc_fallback_activated: bool = False,
        iono_cal_report: IonosphericCalibrationReport | None = None,
    ) -> None:
        """Init the BPS L1 product"""
        SARProduct.__init__(self)

        self.is_monitoring = is_monitoring

        if product is not None:
            self.__dict__.update(product.__dict__)
            self.type = SAR_TO_L1A_PRODUCT_TYPE[self.type]
            if calibration_tag is not None:
                assert len(calibration_tag) == 1
                self.calibration_tag = calibration_tag
            else:
                self.calibration_tag = "_"
        else:
            self.type = "SLC"
            self.calibration_tag = "_"
            self.start_time = PreciseDateTime.from_numeric_datetime(2025)
            self.stop_time = PreciseDateTime.from_numeric_datetime(2025, 1, 1, 0, 0, 25)
            self.swath_list = ["S1"]

        self.sensor_mode = "MEASUREMENT"
        if source is not None:
            self.source_name = source.name
            self.mission_phase_id = source.mission_phase_id
            self.instrument_configuration_id = source.instrument_configuration_id
            self.datatake_id = source.datatake_id
            self.orbit_number = source.orbit_number
            self.orbit_drift_flag = source.orbit_drift_flag
            self.global_coverage_id = source.global_coverage_id
            self.major_cycle_id = source.major_cycle_id
            self.repeat_cycle_id = source.repeat_cycle_id
            self.track_number = source.track_number
            self.anx_time = source.anx_time
            self.source_bit_rate = source.bit_rate if isinstance(source, BIOMASSL0Product) else 0.0
        else:
            self.source_name = "L0S"
            self.mission_phase_id = "INTERFEROMETRIC"
            self.instrument_configuration_id = 1
            self.datatake_id = 1
            self.orbit_number = 1
            self.orbit_drift_flag = False
            self.global_coverage_id = 1
            self.major_cycle_id = 1
            self.repeat_cycle_id = 1
            self.track_number = 1
            self.anx_time = None
            self.source_bit_rate = 0.0

        if source_monitoring is not None:
            self.source_monitoring_name = source_monitoring.name
        else:
            self.source_monitoring_name = None

        if source_auxiliary_names:
            self.source_auxiliary_names = source_auxiliary_names
        else:
            self.source_auxiliary_names = []

        if configuration is not None:
            self.doi = configuration.l1a_doi if self.type == "SCS" else configuration.l1b_doi
            self.frame_number = configuration.frame_id
            self.frame_status = configuration.frame_status
            self.baseline_id = configuration.product_baseline
            self.processing_parameters = configuration.processing_parameters
            self.acquisition_timeline = configuration.acquisition_timeline
            self.sar_image_parameters = configuration.sar_image_parameters
            self.lut_parameters = configuration.lut_parameters
            self.quicklook_parameters = configuration.quicklook_parameters
            self.quality_parameters = configuration.quality_parameters
            self.acquisition_raster_info = configuration.acquisition_raster_info
        else:
            self.doi = "DOI"
            self.frame_number = 1
            self.frame_status = "NOMINAL"
            self.baseline_id = 1
            self.processing_parameters = _default_processing_parameters()
            self.sar_image_parameters = _default_sar_image_parameters()
            self.acquisition_timeline = None
            self.lut_parameters = _default_lut_parameters()
            self.quicklook_parameters = _default_quicklook_parameters()
            self.quality_parameters = _default_quality_parameters()
            self.acquisition_raster_info = None

        self.dc_annotations = dc_annotations
        self.dc_fallback_activated = dc_fallback_activated
        self.preproc_report = l1_pre_proc_report
        self.rfi_masks_statistics = rfi_masks_statistics
        self.iono_cal_report = iono_cal_report
        self._set_product_name()

    def _set_product_name(self):
        assert self.type is not None
        assert self.mission_phase_id is not None
        assert self.start_time is not None
        assert self.stop_time is not None
        self.creation_date = PreciseDateTime.now()
        global_coverage_id_str = encode_product_name_id_value(self.global_coverage_id, npad=2)
        major_cycle_id_str = encode_product_name_id_value(self.major_cycle_id, npad=2)
        repeat_cycle_id_str = encode_product_name_id_value(self.repeat_cycle_id, npad=2)
        self.name = "_".join(
            [
                "BIO",
                self.swath_list[0],
                self.type + self.calibration_tag,
                "1S" if not self.is_monitoring else "1M",
                pdt_to_compact_string(self.start_time),
                pdt_to_compact_string(self.stop_time),
                self.mission_phase_id[0],
                "G" + ("__" if self.mission_phase_id[0] == "C" else global_coverage_id_str),
                "M" + ("__" if self.mission_phase_id[0] == "C" else major_cycle_id_str),
                "C" + ("__" if self.mission_phase_id[0] == "C" else repeat_cycle_id_str),
                "T" + ("___" if self.mission_phase_id[0] == "C" else f"{self.track_number:03d}"),
                "F" + ("___" if self.frame_number == 0 else f"{self.frame_number:03d}"),
                f"{self.baseline_id:02d}",
                pdt_to_compact_date(self.creation_date),
            ]
        )


def _default_processing_parameters() -> BIOMASSL1ProcessingParameters:
    return BIOMASSL1ProcessingParameters(
        raw_data_correction_flag=True,
        rfi_detection_flag=False,
        rfi_correction_flag=False,
        rfi_mitigation_method=common.RfiMitigationMethodType.TIME_AND_FREQUENCY,
        rfi_mask=common.RfiMaskType.SINGLE,
        rfi_mask_generation_method=common.RfiMaskGenerationMethodType.OR,
        rfi_fm_mitigation_method="NOTCH_FILTER",
        rfi_fm_chirp_source=common.RangeReferenceFunctionType.NOMINAL,
        internal_calibration_estimation_flag=True,
        internal_calibration_correction_flag=True,
        internal_calibration_source=common.InternalCalibrationSourceType.EXTRACTED,
        range_reference_function_source=common.RangeReferenceFunctionType.REPLICA,
        range_compression_method=common.RangeCompressionMethodType.INVERSE_FILTER,
        range_window_type={
            main_annotation_models.SwathType.S1.value: main_annotation_models.WeightingWindowType.HAMMING,
            main_annotation_models.SwathType.S2.value: main_annotation_models.WeightingWindowType.HAMMING,
            main_annotation_models.SwathType.S3.value: main_annotation_models.WeightingWindowType.HAMMING,
        },
        range_window_coefficient={
            main_annotation_models.SwathType.S1.value: 0.75,
            main_annotation_models.SwathType.S2.value: 0.75,
            main_annotation_models.SwathType.S3.value: 0.75,
        },
        extended_swath_processing=False,
        dc_method=common.DcMethodType.COMBINED,
        dc_value=0,
        antenna_pattern_correction1_flag=True,
        antenna_pattern_correction2_flag=True,
        antenna_cross_talk_correction_flag=True,
        azimuth_compression_block_samples=256,
        azimuth_compression_block_lines=16700,
        azimuth_compression_block_overlap_samples=46,
        azimuth_compression_block_overlap_lines=4076,
        azimuth_window_type={
            main_annotation_models.SwathType.S1.value: main_annotation_models.WeightingWindowType.HAMMING,
            main_annotation_models.SwathType.S2.value: main_annotation_models.WeightingWindowType.HAMMING,
            main_annotation_models.SwathType.S3.value: main_annotation_models.WeightingWindowType.HAMMING,
        },
        azimuth_window_coefficient={
            main_annotation_models.SwathType.S1.value: 0.75,
            main_annotation_models.SwathType.S2.value: 0.75,
            main_annotation_models.SwathType.S3.value: 0.75,
        },
        bistatic_delay_correction_flag=True,
        bistatic_delay_correction_method=common.BistaticDelayCorrectionMethodType.FULL,
        azimuth_focusing_margins_removal_flag=True,
        range_spreading_loss_compensation_flag=True,
        reference_range=800000,
        polarimetric_correction_flag=True,
        ionosphere_height_defocusing_flag=True,
        ionosphere_height_estimation_method=common.IonosphereHeightEstimationMethodType.AUTOMATIC,
        faraday_rotation_correction_flag=True,
        ionospheric_phase_screen_correction_flag=True,
        group_delay_correction_flag=True,
        autofocus_flag=False,
        autofocus_method=common.AutofocusMethodType.MAP_DRIFT,
        range_upsampling_factor={
            main_annotation_models.SwathType.S1.value: 3,
            main_annotation_models.SwathType.S2.value: 3,
            main_annotation_models.SwathType.S3.value: 3,
        },
        range_downsampling_factor={
            main_annotation_models.SwathType.S1.value: 1,
            main_annotation_models.SwathType.S2.value: 1,
            main_annotation_models.SwathType.S3.value: 1,
        },
        azimuth_upsampling_factor={
            main_annotation_models.SwathType.S1.value: 3,
            main_annotation_models.SwathType.S2.value: 3,
            main_annotation_models.SwathType.S3.value: 3,
        },
        azimuth_downsampling_factor={
            main_annotation_models.SwathType.S1.value: 10,
            main_annotation_models.SwathType.S2.value: 10,
            main_annotation_models.SwathType.S3.value: 10,
        },
        detection_flag=True,
        thermal_denoising_flag=True,
        noise_parameters_source=common.InternalCalibrationSourceType.EXTRACTED,
        ground_projection_flag=True,
        requested_height_model_used=True,
        requested_height_model=common.HeightModelBaseType.COPERNICUS_DEM,
        requested_height_model_version="COP-DEM_GLO-90-DGED-2021_1 EGM2008-2.5",
        absolute_calibration_constants={
            common.PolarisationType.HH: 1.0,
            common.PolarisationType.HV: 1.0,
            common.PolarisationType.VH: 1.0,
            common.PolarisationType.VV: 1.0,
        },
        polarimetric_distortion=PolarimetricDistortionType(
            cross_talk=common.CrossTalkList(
                hv_rx=complex(0, 0),
                hv_tx=complex(0, 0),
                vh_rx=complex(0, 0),
                vh_tx=complex(0, 0),
            ),
            channel_imbalance=common.ChannelImbalanceList(hv_rx=complex(0, 0), hv_tx=complex(0, 0)),
        ),
    )


def _default_sar_image_parameters() -> SARImageParameters:
    return SARImageParameters(
        pixel_representation=common.PixelRepresentationType.ABS_PHASE,
        pixel_type=main_annotation_models.PixelTypeType.VALUE_32_BIT_FLOAT,
        abs_compression_method=common_types.CompressionMethodType.LERC_ZSTD,
        abs_max_z_error=0.0001,
        abs_max_z_error_percentile=0,
        phase_compression_method=common_types.CompressionMethodType.LERC_ZSTD,
        phase_max_z_error=0.001,
        phase_max_z_error_percentile=0,
        no_pixel_value=-9999.0,
        block_size=512,
    )


def _default_lut_parameters() -> LUTParameters:
    return LUTParameters(
        lut_range_decimation_factors=LUTParameters.LutDecimationFactors(
            dem_based_quantity=20, rfi_based_quantity=20, image_based_quantity=20
        ),
        lut_azimuth_decimation_factors=LUTParameters.LutDecimationFactors(
            dem_based_quantity=120, rfi_based_quantity=120, image_based_quantity=120
        ),
        lut_block_size=512,
        lut_layers_completeness_flag=True,
        no_pixel_value=-9999.0,
    )


def _default_quicklook_parameters() -> QuicklookParameters:
    return QuicklookParameters(
        ql_range_decimation_factor=20,
        ql_range_averaging_factor=20,
        ql_azimuth_decimation_factor=120,
        ql_azimuth_averaging_factor=120,
        ql_absolute_scaling_factor={
            main_annotation_models.ChannelType.RED.value: 1.0,
            main_annotation_models.ChannelType.GREEN.value: 1.0,
            main_annotation_models.ChannelType.BLUE.value: 1.0,
        },
    )


def _default_quality_parameters() -> QualityParameters:
    return QualityParameters(
        dc_rmserror_threshold=100,
        max_isp_gap=0,
        raw_mean_expected=0.0,
        raw_mean_threshold=0.0,
        raw_std_expected=0.0,
        raw_std_threshold=0.0,
        max_rfi_tm_percentage=0.0,
        max_rfi_fm_percentage=0.0,
        max_drift_amplitude_std_fraction=0.0,
        max_drift_phase_std_fraction=0.0,
        max_drift_amplitude_error=0.0,
        max_drift_phase_error=0.0,
        max_invalid_drift_fraction=0.0,
    )
