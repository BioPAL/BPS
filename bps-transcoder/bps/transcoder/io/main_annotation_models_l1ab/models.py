# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Main annotation models l1ab
-------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
    InOutBandPowerRatioListType,
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
    PowerRatioListType,
    PowerRatioType,
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


@dataclass(kw_only=True)
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

    acquisition_information: AcquisitionInformationType = field(
        metadata={
            "name": "acquisitionInformation",
            "type": "Element",
            "namespace": "",
        },
    )
    sar_image: SarImageType = field(
        metadata={
            "name": "sarImage",
            "type": "Element",
            "namespace": "",
        },
    )
    instrument_parameters: InstrumentParametersType = field(
        metadata={
            "name": "instrumentParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_data_analysis: RawDataAnalysisType = field(
        metadata={
            "name": "rawDataAnalysis",
            "type": "Element",
            "namespace": "",
        },
    )
    processing_parameters: ProcessingParametersType = field(
        metadata={
            "name": "processingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_calibration: InternalCalibrationType = field(
        metadata={
            "name": "internalCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_mitigation: RfiMitigationType = field(
        metadata={
            "name": "rfiMitigation",
            "type": "Element",
            "namespace": "",
        },
    )
    doppler_parameters: DopplerParametersType = field(
        metadata={
            "name": "dopplerParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    radiometric_calibration: RadiometricCalibrationType = field(
        metadata={
            "name": "radiometricCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    polarimetric_distortion: PolarimetricDistortionType = field(
        metadata={
            "name": "polarimetricDistortion",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_correction: IonosphereCorrectionType = field(
        metadata={
            "name": "ionosphereCorrection",
            "type": "Element",
            "namespace": "",
        },
    )
    geometry: GeometryType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    quality: QualityType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    annotation_lut: LayerListType = field(
        metadata={
            "name": "annotationLUT",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class MainAnnotation(MainAnnotationType):
    """
    BIOMASS L1a/b product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
