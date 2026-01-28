# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 core processor configuration structures
-------------------------------------------
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from bps.common.utils import EarthModel, ProductFormat


class BPSL1CoreProcessorStep(Enum):
    """BPSL1CoreProcessor processing steps"""

    RFI_MITIGATION = "RFIMitigation"
    RANGE_FOCUSER = "RangeCompression"
    DOPPLER_CENTROID_ESTIMATOR = "DopplerCentroidEstimation"
    DOPPLER_RATE_ESTIMATOR = "DopplerRateEstimation"
    AZIMUTH_FOCUSER = "AzimuthCompression"
    RANGE_COMPENSATOR = "RadiometricCalibration"
    POLARIMETRIC_COMPENSATOR = "PolarimetricCompensation"
    MULTI_LOOKER = "Multilooking"
    NESZ_MAP_GENERATOR = "NeszMapGenerator"
    DENOISER = "Denoising"
    SLANT2_GROUND = "GroundRangeDataProjection"


class BPSL1CoreProcessorProductId(Enum):
    """BPSL1CoreProcessor output product ID"""

    RFI_MITIGATION = "RFIMitigation"
    RFI_MITIGATION_TIME_MASK = "RFIMitigation/TimeMask"
    RFI_MITIGATION_FREQUENCY_MASK = "RFIMitigation/FrequencyMask"
    RANGE_FOCUSER = "RangeCompression"
    DOPPLER_CENTROID_ESTIMATOR = "DopplerCentroidEstimation"
    DOPPLER_CENTROID_ESTIMATOR_GRID = "DopplerCentroidEstimation/DCGrid"
    DOPPLER_CENTROID_ESTIMATOR_STD_DEV_GRID = "DopplerCentroidEstimation/DCStdDevGrid"
    DOPPLER_RATE_ESTIMATOR = "DopplerRateEstimation"
    AZIMUTH_FOCUSER = "AzimuthCompression"
    RANGE_COMPENSATOR = "RadiometricCalibration"
    RANGE_COMPENSATOR_CORRECTED_FACTORS = "RadiometricCalibration/RangeCorrectedFactors"
    POLARIMETRIC_COMPENSATOR = "PolarimetricCompensation"
    POLARIMETRIC_COMPENSATOR_IONO_REPORT = "PolarimetricCompensation/IonosphericCalibrationReport"
    POLARIMETRIC_COMPENSATOR_FR = "PolarimetricCompensation/FaradayRotation"
    POLARIMETRIC_COMPENSATOR_FR_PLANE = "PolarimetricCompensation/FaradayRotationPlane"
    POLARIMETRIC_COMPENSATOR_PHASE_SCREEN_BB = "PolarimetricCompensation/PhaseScreen"
    MULTI_LOOKER = "Multilooking"
    NESZ_MAP_GENERATOR = "NeszMapGenerator"
    DENOISER = "Denoising"
    DENOISER_NESZ_MAP_RESAMPLED = "Denoising/NeszMapResampled"
    SLANT2_GROUND = "GroundRangeDataProjection"
    DEM_COPERNICUS = "Base/DEM[COPERNICUS]"
    SAR_DEM_COPERNICUS = "Base/SarDEM[COPERNICUS]"
    HEIGHT_MODEL_COPERNICUS = "Base/HeightModel[COPERNICUS]"
    DEM_SRTM = "Base/DEM[SRTM]"
    SAR_DEM_SRTM = "Base/SarDEM[SRTM]"
    HEIGHT_MODEL_SRTM = "Base/HeightModel[SRTM]"


class AntennaPatternCompensationLevel(Enum):
    """Level of antenna pattern compensation"""

    DISABLED = "DISABLED"
    APC_PRE_ONLY = "APC_PRE_ONLY"
    APC_PRE_CROSS_ONLY = "APC_PRE_CROSS_ONLY"
    APC_PRE_POST_ONLY = "APC_PRE_POST_ONLY"
    FULL = "FULL"


@dataclass
class BPSL1CoreProcessingOptions:
    """BPSL1CoreProcessor processing options"""

    @dataclass
    class ProcessingSettings:
        """BPSL1CoreProcessor processing settings"""

        dem: dict[BPSL1CoreProcessorStep, EarthModel]
        prf_change_data_post_processing: bool = False
        apc_level: AntennaPatternCompensationLevel = AntennaPatternCompensationLevel.DISABLED
        elevation_mispointing_deg: float = 0.0
        ionospheric_calibration_enabled: bool = True
        rfi_use_chirp_product: Literal["ANNOTATION", "PRODUCT", "AUTO"] = "AUTO"
        rfi_operation_mode: Literal["DETECTION_ONLY", "DETECTION_AND_MITIGATION"] = "DETECTION_AND_MITIGATION"
        drop_azimuth_focuser_margin: bool = True

    @dataclass
    class ExternalResources:
        """BPSL1CoreProcessor external resources"""

        @dataclass
        class DemInfo:
            """Dem database information"""

            earth_model: EarthModel
            entry_point: Path | str
            geoid_file: Path | None

        dem_info_list: list[DemInfo]
        prf_resampling_filter_product: Path | None = None

    @dataclass
    class InterfaceSettings:
        """BPSL1CoreProcessor interface settings"""

        products_format: ProductFormat
        enable_quick_look_generation: bool = False
        remove_intermediate_products: bool = False

    steps: dict[BPSL1CoreProcessorStep, bool]
    settings: ProcessingSettings
    output_products: dict[BPSL1CoreProcessorProductId, str]
    external_resources: ExternalResources
    interface_settings: InterfaceSettings
