# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Main annotation models l1ab
-------------------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bps.common.io.common_types import (
    AutofocusMethodType,
    AzimuthPolynomialType,
    BistaticDelayCorrectionMethodType,
    ChannelImbalanceList,
    ChannelType,
    Complex,
    ComplexArray,
    CrossTalkList,
    DataFormatModeType,
    DatumType,
    DcMethodType,
    DoubleArray,
    DoubleArrayWithUnits,
    DoubleWithUnit,
    FloatArray,
    FloatArrayWithUnits,
    FloatWithChannel,
    FloatWithPolarisation,
    FloatWithUnit,
    GeodeticReferenceFrameType,
    GroupType,
    HeightModelBaseType,
    HeightModelType,
    IntArray,
    InterferometricPairListType,
    InterferometricPairType,
    InternalCalibrationSourceType,
    IonosphereHeightEstimationMethodType,
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    MissionPhaseIdtype,
    MissionType,
    OrbitAttitudeSourceType,
    OrbitPassType,
    PixelQuantityType,
    PixelRepresentationType,
    PixelTypeType,
    PolarisationType,
    ProcessingModeType,
    ProductCompositionType,
    ProductType,
    ProjectionType,
    RangeCompressionMethodType,
    RangeReferenceFunctionType,
    RfiFmmitigationMethodType,
    RfiMaskGenerationMethodType,
    RfiMaskType,
    RfiMitigationMethodType,
    SensorModeType,
    SlantRangePolynomialType,
    StateType,
    SwathType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
    WeightingWindowType,
)
from bps.transcoder.io.common_annotation_models_l1 import (
    AcquisitionInformationType,
    CalibrationConstantListType,
    CoordinateConversionListType,
    CoordinateConversionType,
    DataFormatType,
    DcEstimateListType,
    DcEstimateType,
    DopplerParametersType,
    ErrorCountersType,
    FirstLineSensingTimeListType,
    FmRateEstimatesListType,
    GeometryType,
    InstrumentParametersType,
    InternalCalibrationParametersListType,
    InternalCalibrationSequenceListType,
    InternalCalibrationSequenceType,
    InternalCalibrationType,
    IonosphereCorrectionType,
    LastLineSensingTimeListType,
    NoiseGainListType,
    NoiseListType,
    NoiseSequenceListType,
    NoiseSequenceType,
    PolarimetricDistortionType,
    PolarisationListType,
    PrfListType,
    ProcessingGainListType,
    ProcessingParametersType,
    QualityParametersListType,
    QualityParametersType,
    QualityType,
    RadiometricCalibrationType,
    RawDataAnalysisType,
    RawDataStatisticsListType,
    RawDataStatisticsType,
    RfiIsolatedFmreportListType,
    RfiIsolatedFmreportType,
    RfiMitigationType,
    RfiPersistentFmreportListType,
    RfiPersistentFmreportType,
    RfiTmreportListType,
    RfiTmreportType,
    RxGainListType,
    SarImageType,
    SpectrumProcessingParametersType,
    SwlListType,
    SwpListType,
    TxPulseListType,
    TxPulseType,
)


@dataclass
class MainAnnotationType:
    """
    Parameters
    ----------
    acquisition_information
        Acquisition information DSR. This DSR contains information that applies to the entire data set.
    sar_image
        SAR image DSR. This DSR contains all the necessary information to exploit the measurement data set (i.e. SAR
        images).
    instrument_parameters
        Instrument parameters DSR. This DSR contains the main instrument settings at the time of imaging.
    raw_data_analysis
        RAW data analysis DSR. This DSR contains the main elements related to the RAW data consolidation and
        analysis performed by the processor.
    processing_parameters
        Processing parameters DSR. This DSR contains the exhaustive list of static SAR processing parameters and of
        corrections applied.
    internal_calibration
        Internal calibration DSR. This DSR contains the results of the internal calibration analysis performed by
        the processor.
    rfi_mitigation
        Radio Frequency Interference mitigation DSR. This DSR contains information on detected RFI and on their
        mitigation.
    doppler_parameters
        Doppler parameters DSR. This DSR contains the Doppler Centroid (DC) and Frequency Modulation rate (FM)
        parameters estimated and used during processing.
    radiometric_calibration
        Radiometric calibration DSR. This DSR contains all the necessary information to absolutely calibrate the
        data pixels.
    polarimetric_distortion
        Polarimetric distortion DSR. This DSR contains the necessary information e.g. receive and transmit
        polarisation distortion matrix, allowing users to correct for them if not applied at processing level.
    ionosphere_correction
        Ionosphere correction DSR. This DSR contains the results of the ionosphere correction estimated and applied
        by the processor.
    geometry
        Geometry DSR. This DSR contains all the necessary information to understand the Earth model/geometry used in
        the image and its geolocation.
    quality
        Quality DSR. This DSR contains in a single DSR quality flags and thresholds for basic quality assessment
        done at processing level.
    annotation_lut
        Annotation LUT DSR. This DSR contains the list of Look-Up Tables (LUTs) complementing product main
        annotations.
    """

    class Meta:
        name = "mainAnnotationType"

    acquisition_information: Optional[AcquisitionInformationType] = field(
        default=None,
        metadata={
            "name": "acquisitionInformation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sar_image: Optional[SarImageType] = field(
        default=None,
        metadata={
            "name": "sarImage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    instrument_parameters: Optional[InstrumentParametersType] = field(
        default=None,
        metadata={
            "name": "instrumentParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_data_analysis: Optional[RawDataAnalysisType] = field(
        default=None,
        metadata={
            "name": "rawDataAnalysis",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_parameters: Optional[ProcessingParametersType] = field(
        default=None,
        metadata={
            "name": "processingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_calibration: Optional[InternalCalibrationType] = field(
        default=None,
        metadata={
            "name": "internalCalibration",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_mitigation: Optional[RfiMitigationType] = field(
        default=None,
        metadata={
            "name": "rfiMitigation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    doppler_parameters: Optional[DopplerParametersType] = field(
        default=None,
        metadata={
            "name": "dopplerParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    radiometric_calibration: Optional[RadiometricCalibrationType] = field(
        default=None,
        metadata={
            "name": "radiometricCalibration",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarimetric_distortion: Optional[PolarimetricDistortionType] = field(
        default=None,
        metadata={
            "name": "polarimetricDistortion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ionosphere_correction: Optional[IonosphereCorrectionType] = field(
        default=None,
        metadata={
            "name": "ionosphereCorrection",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    geometry: Optional[GeometryType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    quality: Optional[QualityType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    annotation_lut: Optional[LayerListType] = field(
        default=None,
        metadata={
            "name": "annotationLUT",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class MainAnnotation(MainAnnotationType):
    """
    BIOMASS L1a/b product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
