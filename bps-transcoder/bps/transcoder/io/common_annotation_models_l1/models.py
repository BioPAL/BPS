# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD common annotation models L1
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


@dataclass
class ErrorCountersType:
    """
    Parameters
    ----------
    num_isp_header_errors
        Total number of errors detected in ISP headers.
    num_isp_missing
        Total number of missing ISP.
    """

    class Meta:
        name = "errorCountersType"

    num_isp_header_errors: Optional[int] = field(
        default=None,
        metadata={
            "name": "numIspHeaderErrors",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    num_isp_missing: Optional[int] = field(
        default=None,
        metadata={
            "name": "numIspMissing",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class NoiseSequenceType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time of the noise measurement [UTC].
    noise_power_correction_factor
        Noise power correction factor.
    number_of_noise_lines
        Number of noise lines used to calculate noise correction factor.
    """

    class Meta:
        name = "noiseSequenceType"

    azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    noise_power_correction_factor: Optional[float] = field(
        default=None,
        metadata={
            "name": "noisePowerCorrectionFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_noise_lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfNoiseLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class NoiseSequenceListType:
    """
    Parameters
    ----------
    noise_sequence
        Noise parameters derived from noise packets. List maxOccurs is set to 2 considering one entry for preamble
        and one for postamble noise sequences.
    polarisation
    count
        Number of noise sequences for the current polarisation within the list.
    """

    class Meta:
        name = "noiseSequenceListType"

    noise_sequence: list[NoiseSequenceType] = field(
        default_factory=list, metadata={"name": "noiseSequence", "type": "Element", "namespace": "", "max_occurs": 2}
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class PolarisationListType:
    """
    Parameters
    ----------
    polarisation
        Polarisation (HH, HV, VH, VV).
    count
    """

    class Meta:
        name = "polarisationListType"

    polarisation: list[PolarisationType] = field(
        default_factory=list, metadata={"type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4}
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RawDataStatisticsType:
    """
    Parameters
    ----------
    i_bias
        Calculated I bias.
    q_bias
        Calculated Q bias.
    iq_quadrature_departure
        Calculated I/Q quadrature departure.
    iq_gain_imbalance
        Calculated I/Q gain imbalance.
    polarisation
    """

    class Meta:
        name = "rawDataStatisticsType"

    i_bias: Optional[float] = field(
        default=None,
        metadata={
            "name": "iBias",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    q_bias: Optional[float] = field(
        default=None,
        metadata={
            "name": "qBias",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    iq_quadrature_departure: Optional[float] = field(
        default=None,
        metadata={
            "name": "iqQuadratureDeparture",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    iq_gain_imbalance: Optional[float] = field(
        default=None,
        metadata={
            "name": "iqGainImbalance",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RfiIsolatedFmreportType:
    """
    Parameters
    ----------
    percentage_affected_lines
        Percentage of input RAW data lines affected by isolated RFI.
    max_percentage_affected_bw
        Max percentage of bandwidth affected by isolated RFI in a single line.
    avg_percentage_affected_bw
        Average percentage of bandwidth affected by isolated RFI
    polarisation
    """

    class Meta:
        name = "rfiIsolatedFMReportType"

    percentage_affected_lines: Optional[float] = field(
        default=None,
        metadata={
            "name": "percentageAffectedLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_percentage_affected_bw: Optional[float] = field(
        default=None,
        metadata={
            "name": "maxPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    avg_percentage_affected_bw: Optional[float] = field(
        default=None,
        metadata={
            "name": "avgPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RfiPersistentFmreportType:
    """
    Parameters
    ----------
    max_percentage_affected_bw
        Max percentage of bandwidth affected by persistent RFI.
    avg_percentage_affected_bw
        Average percentage of bandwidth affected by persistent RFI.
    percentage_affected_lines
        Percentage of lines affected by persistent RFI.
    polarisation
    """

    class Meta:
        name = "rfiPersistentFMReportType"

    max_percentage_affected_bw: Optional[float] = field(
        default=None,
        metadata={
            "name": "maxPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    avg_percentage_affected_bw: Optional[float] = field(
        default=None,
        metadata={
            "name": "avgPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    percentage_affected_lines: Optional[float] = field(
        default=None,
        metadata={
            "name": "percentageAffectedLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RfiTmreportType:
    """
    Parameters
    ----------
    percentage_affected_lines
        Percentage of input RAW data lines affected by RFI.
    avg_percentage_affected_samples
        Average percentage of affected input RAW data samples in the lines containing RFI.
    max_percentage_affected_samples
        Maximum percentage of input RAW data samples affected by RFI in the same line.
    polarisation
    """

    class Meta:
        name = "rfiTMReportType"

    percentage_affected_lines: Optional[float] = field(
        default=None,
        metadata={
            "name": "percentageAffectedLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    avg_percentage_affected_samples: Optional[float] = field(
        default=None,
        metadata={
            "name": "avgPercentageAffectedSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_percentage_affected_samples: Optional[float] = field(
        default=None,
        metadata={
            "name": "maxPercentageAffectedSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class AcquisitionInformationType:
    """
    Parameters
    ----------
    mission
        Mission (BIOMASS).
    swath
        Swath (S1, S2, S3).
    product_type
        Product type (SCS, DGM).
    polarisation_list
        List of polarisations.
    start_time
        Zero Doppler start time of the image [UTC].
    stop_time
        Zero Doppler stop time of the image [UTC].
    mission_phase_id
        Mission phase identifier (COM, TOM, INT).
    drift_phase_flag
        True if in drift phase, False otherwise.
    sensor_mode
        Sensor mode (Measurement, RX Only, External Calibration).
    global_coverage_id
        Global coverage identifier.
    major_cycle_id
        Major cycle identifier.
    repeat_cycle_id
        Repeat cycle identifier.
    absolute_orbit_number
        Absolute orbit number at data set start time.
    relative_orbit_number
        Relative orbit number (track) at data set start time.
    orbit_pass
        Orbit pass (Ascending, Descending).
    platform_heading
        Platform heading relative to North [deg].
    data_take_id
        Data take identifier.
    frame
        Frame identifier. This value is considered unique per beam. In case the product is non-framed, this field is
        empty. In case the product has been obtained from the merging of two frames, this number corresponds to the
        longer one.
    product_composition
        Product composition indicator, where the valid values are: “Nominal”, to indicate a framed product of
        nominal length; “Merged”, if the product is resulting from the merging of a “short” frame with the
        contiguous one; “Partial”, if the product length is not nominal, but it is not merged to a contiguous one
        since it is “long enough”; “Incomplete”, if the product length is not nominal due to contingency (e.g., data
        loss); “Not Framed”, if the product is not framed, i.e., as long as input L0 one.
    """

    class Meta:
        name = "acquisitionInformationType"

    mission: Optional[MissionType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swath: Optional[SwathType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_type: Optional[ProductType] = field(
        default=None,
        metadata={
            "name": "productType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation_list: Optional[PolarisationListType] = field(
        default=None,
        metadata={
            "name": "polarisationList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    start_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "startTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    stop_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "stopTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    mission_phase_id: Optional[MissionPhaseIdtype] = field(
        default=None,
        metadata={
            "name": "missionPhaseID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    drift_phase_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "driftPhaseFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    sensor_mode: Optional[SensorModeType] = field(
        default=None,
        metadata={
            "name": "sensorMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    global_coverage_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "globalCoverageID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    major_cycle_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "majorCycleID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    repeat_cycle_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "repeatCycleID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    absolute_orbit_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "absoluteOrbitNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    relative_orbit_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "relativeOrbitNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    orbit_pass: Optional[OrbitPassType] = field(
        default=None,
        metadata={
            "name": "orbitPass",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    platform_heading: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "platformHeading",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    data_take_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "dataTakeID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    frame: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_composition: Optional[ProductCompositionType] = field(
        default=None,
        metadata={
            "name": "productComposition",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class CalibrationConstantListType:
    """
    Parameters
    ----------
    absolute_calibration_constant
        Absolute calibration constant applied multiplicatively to the image during processing for the current
        polarisation.
    count
    """

    class Meta:
        name = "calibrationConstantListType"

    absolute_calibration_constant: list[FloatWithPolarisation] = field(
        default_factory=list,
        metadata={
            "name": "absoluteCalibrationConstant",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
            "max_occurs": 4,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class CoordinateConversionType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time corresponding to the start of the current polynomial validity area [UTC].
    t0
        Two-way slant range time origin for coordinate conversion polynomial [s].
    sr0
        Slant range origin used for ground range calculation [m].
    slant_to_ground_coefficients
        Coefficients to convert from slant range to ground range. Ground range = g0+g1(sr-sr0)+g2(sr-sr0)^2+g3(sr-
        sr0)^3+g4(sr-sr0)^4+...+gN(sr-sr0)^N, where sr is the slant range distance to the desired pixel and N is the
        number of coefficients in the array minus one.
    gr0
        Ground range origin used for slant range calculation [m].
    ground_to_slant_coefficients
        Coefficients to convert from ground range to slant range coefficients. Slant range = s0+s1(gr-gr0)+s2(gr-
        gr0)^2+s3(gr-gr0)^3+s4(gr-gr0)^4+...+sN(gr-gr0)^N, where gr is the ground range distance to the desired
        pixel and N is the number of coefficients in the array minus one.
    """

    class Meta:
        name = "coordinateConversionType"

    azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    t0: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sr0: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slant_to_ground_coefficients: Optional[DoubleArray] = field(
        default=None,
        metadata={
            "name": "slantToGroundCoefficients",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    gr0: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ground_to_slant_coefficients: Optional[DoubleArray] = field(
        default=None,
        metadata={
            "name": "groundToSlantCoefficients",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class DataFormatType:
    """
    Parameters
    ----------
    echo_format
        Data format of echo packets.
    calibration_format
        Data format of calibration packets.
    noise_format
        Data format of noise packets.
    mean_bit_rate
        The calculated mean bit rate for the segment provide as input to the processor [Mbps].
    """

    class Meta:
        name = "dataFormatType"

    echo_format: Optional[DataFormatModeType] = field(
        default=None,
        metadata={
            "name": "echoFormat",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    calibration_format: Optional[DataFormatModeType] = field(
        default=None,
        metadata={
            "name": "calibrationFormat",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    noise_format: Optional[DataFormatModeType] = field(
        default=None,
        metadata={
            "name": "noiseFormat",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    mean_bit_rate: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "meanBitRate",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class DcEstimateType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time corresponding to the start of the current polynomial validity area [UTC].
    t0
        Two-way slant range time origin for polynomial [s].
    geometry_dcpolynomial
        Doppler Centroid estimated from orbit and attitude, expressed as the following polynomial (assuming 5
        coefficients): DC=d0+d1(tSR-t0)+d2(tSR-t0)^2+d3(tSR-t0)^3+d4(tSR-t0)^4, where tSR is the two-way slant range
        time.
    combined_dcpolynomial
        Doppler Centroid estimated from both orbit and attitude and data, expressed as the following polynomial
        (assuming 5 coefficients): DC=d0+d1(tSR-t0)+d2(tSR-t0)^2+d3(tSR-t0)^3+d4(tSR-t0)^4, where tSR is the two-way
        slant range time.
    combined_dcvalues
        Combined Doppler Centroid estimates [Hz]. Each estimate represents the Doppler Centroid frequency at the
        slant range time specified by combinedDCSlantRangeTimes, estimated within the current block. The values are
        averaged on all the polarisations used for estimation.
    combined_dcslant_range_times
        Two-way slant range times combinedDCValues are referred to [s].
    combined_dcrmserror
        The RMS error of the Doppler Centroid estimate [Hz]. It is calculated as the average of the individual RMS
        residual errors between input fine Doppler Centroid estimates and the fitted polynomial. If the Doppler
        Centroid was not estimated with combined approach, this is set to 0.
    combined_dcrmserror_above_threshold
        False if the RMS error is below the acceptable threshold for the Doppler Centroid estimated with combined
        approach. True if the RMS error is greater than or equal to the acceptable threshold.
    """

    class Meta:
        name = "dcEstimateType"

    azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    t0: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    geometry_dcpolynomial: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "geometryDCPolynomial",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    combined_dcpolynomial: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "combinedDCPolynomial",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    combined_dcvalues: Optional[DoubleArrayWithUnits] = field(
        default=None,
        metadata={
            "name": "combinedDCValues",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    combined_dcslant_range_times: Optional[DoubleArrayWithUnits] = field(
        default=None,
        metadata={
            "name": "combinedDCSlantRangeTimes",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    combined_dcrmserror: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "combinedDCRMSError",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    combined_dcrmserror_above_threshold: Optional[str] = field(
        default=None,
        metadata={
            "name": "combinedDCRMSErrorAboveThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class FirstLineSensingTimeListType:
    """
    Parameters
    ----------
    first_line_sensing_time
        Sensing time of first line of the input RAW data for the current polarisation [UTC].
    count
    """

    class Meta:
        name = "firstLineSensingTimeListType"

    first_line_sensing_time: list[TimeTypeWithPolarisation] = field(
        default_factory=list,
        metadata={"name": "firstLineSensingTime", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class GeometryType:
    """
    Parameters
    ----------
    height_model
        Digital Elevation Model (DEM) commanded for usage during processing (with version as attribute).
    height_model_used_flag
        True if the DEM effectively used during processing is the commanded one reported in heightModel, False
        otherwise (i.e. ellipsoid has been used for contingency handling).
    roll_bias
        Bias added to roll estimated from attitude to offset it [deg].
    """

    class Meta:
        name = "geometryType"

    height_model: Optional[HeightModelType] = field(
        default=None,
        metadata={
            "name": "heightModel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    height_model_used_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "heightModelUsedFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    roll_bias: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "rollBias",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class InternalCalibrationSequenceType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time in azimuth at which internal calibration parameters applies [UTC].
    drift_amplitude
        Amplitude drift for the current power gain product (PGP).
    drift_phase
        Phase drift for the current power gain product (PGP) [rad].
    model_drift_amplitude
        Amplitude drift derived from model.
    model_drift_phase
        Phase drift derived from model [rad].
    relative_drift_valid_flag
        Indicates if the current amplitude and phase drifts are valid (within the configured threshold) when
        compared to the mean and standard deviation of all the amplitude and phase drifts.
    absolute_drift_valid_flag
        Indicates if the current amplitude and phase drifts are valid (within the configured threshold) when
        compared to the value of the model.
    cross_correlation_bandwidth
        3-dB pulse width of chirp replica cross-correlation function between the reconstructed replica and the
        nominal replica [Hz].
    cross_correlation_pslr
        Peak Side Lobe Ratio (PSLR) of replica cross-correlation function between the reconstructed replica and the
        nominal replica [dB].
    cross_correlation_islr
        Integrated Side Lobe Ratio (ISLR) of cross-correlation function between the reconstructed replica and the
        nominal replica [dB].
    cross_correlation_peak_location
        Peak location of cross-correlation function between the reconstructed replica and the nominal replica
        [samples].
    reconstructed_replica_valid_flag
        Indicates if the cross-correlation between the nominal replica and this extracted replica resulted in a
        valid peak location.
    internal_time_delay
        Internal time delay [s] representing the calculated deviation of the location of this replica from the
        location of the transmitted pulse, i.e. the nominal replica.
    internal_tx_channel_imbalance_amplitude
        Amplitude of instrument internal transmit channel imbalance.
    internal_tx_channel_imbalance_phase
        Phase of instrument internal transmit channel imbalance [rad].
    internal_rx_channel_imbalance_amplitude
        Amplitude of instrument internal receive channel imbalance.
    internal_rx_channel_imbalance_phase
        Phase of instrument internal receive channel imbalance [rad].
    transmit_power_tracking_d1_amplitude
        Amplitude of transmit power tracking for doublet D1.
    transmit_power_tracking_d1_phase
        Phase of transmit power tracking for doublet D1 [rad].
    receive_power_tracking_d1_amplitude
        Amplitude of receive power tracking for doublet D1.
    receive_power_tracking_d1_phase
        Phase of receive power tracking for doublet D1 [rad].
    transmit_power_tracking_d2_amplitude
        Amplitude of transmit power tracking for doublet D2.
    transmit_power_tracking_d2_phase
        Phase of transmit power tracking for doublet D2 [rad].
    receive_power_tracking_d2_amplitude
        Amplitude of receive power tracking for doublet D2.
    receive_power_tracking_d2_phase
        Phase of receive power tracking for doublet D2 [rad].
    """

    class Meta:
        name = "internalCalibrationSequenceType"

    azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    drift_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "driftAmplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    drift_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "driftPhase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    model_drift_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "modelDriftAmplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    model_drift_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "modelDriftPhase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    relative_drift_valid_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "relativeDriftValidFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    absolute_drift_valid_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "absoluteDriftValidFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    cross_correlation_bandwidth: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "crossCorrelationBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    cross_correlation_pslr: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "crossCorrelationPslr",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    cross_correlation_islr: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "crossCorrelationIslr",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    cross_correlation_peak_location: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "crossCorrelationPeakLocation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reconstructed_replica_valid_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "reconstructedReplicaValidFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    internal_time_delay: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "internalTimeDelay",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_tx_channel_imbalance_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "internalTxChannelImbalanceAmplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_tx_channel_imbalance_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "internalTxChannelImbalancePhase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_rx_channel_imbalance_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "internalRxChannelImbalanceAmplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_rx_channel_imbalance_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "internalRxChannelImbalancePhase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    transmit_power_tracking_d1_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "transmitPowerTrackingD1Amplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    transmit_power_tracking_d1_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "transmitPowerTrackingD1Phase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    receive_power_tracking_d1_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "receivePowerTrackingD1Amplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    receive_power_tracking_d1_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "receivePowerTrackingD1Phase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    transmit_power_tracking_d2_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "transmitPowerTrackingD2Amplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    transmit_power_tracking_d2_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "transmitPowerTrackingD2Phase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    receive_power_tracking_d2_amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "receivePowerTrackingD2Amplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    receive_power_tracking_d2_phase: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "receivePowerTrackingD2Phase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class IonosphereCorrectionType:
    """
    Parameters
    ----------
    ionosphere_height_used
        Ionosphere height used for processing [m] (-1 if not available).
    ionosphere_height_estimated
        Ionosphere height estimated using Feature Tracking or Squint Sensitivity method, if selected via
        configuration [m] (-1 if not available).
    ionosphere_height_estimation_method_selected
        Ionosphere height estimation method selected for ionospheric processing, between Feature Tracking and Squint
        Sensitivity, when ionosphereHeightEstimationMethod is set to Automatic (NA if not available).
    ionosphere_height_estimation_latitude_value
        Latitude value used to select ionosphere height estimation method between Feature Tracking and Squint
        Sensitivity, when ionosphereHeightEstimationMethod is set to Automatic [deg].
    ionosphere_height_estimation_flag
        True if ionosphere height estimation is completed successfully, False otherwise.
    ionosphere_height_estimation_method_used
        Ionosphere height estimation method effectively used for ionospheric processing. Can be different from
        ionosphereHeightEstimationMethod and ionosphereHeightEstimationMethodSelected. (NA if not available)
    gaussian_filter_computation_flag
        True if computed filter dimensions are below maximum allowed values, False otherwise.
    faraday_rotation_correction_applied
        Flag indicating if Faraday Rotation correction has been effectively applied or not. Can be different from
        faradayRotationCorrectionFlag.
    autofocus_shifts_applied
        Flag indicating if correction of azimuth shift estimated through autofocus has been effectively applied or
        not. Can be different from autofocusFlag.
    """

    class Meta:
        name = "ionosphereCorrectionType"

    ionosphere_height_used: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ionosphere_height_estimated: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightEstimated",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ionosphere_height_estimation_method_selected: Optional[IonosphereHeightEstimationMethodType] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightEstimationMethodSelected",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ionosphere_height_estimation_latitude_value: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightEstimationLatitudeValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ionosphere_height_estimation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightEstimationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    ionosphere_height_estimation_method_used: Optional[IonosphereHeightEstimationMethodType] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightEstimationMethodUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    gaussian_filter_computation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "gaussianFilterComputationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    faraday_rotation_correction_applied: Optional[str] = field(
        default=None,
        metadata={
            "name": "faradayRotationCorrectionApplied",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    autofocus_shifts_applied: Optional[str] = field(
        default=None,
        metadata={
            "name": "autofocusShiftsApplied",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class LastLineSensingTimeListType:
    """
    Parameters
    ----------
    last_line_sensing_time
        Sensing time of last line of the input RAW data for the current polarisation [UTC].
    count
    """

    class Meta:
        name = "lastLineSensingTimeListType"

    last_line_sensing_time: list[TimeTypeWithPolarisation] = field(
        default_factory=list,
        metadata={"name": "lastLineSensingTime", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class NoiseGainListType:
    """
    Parameters
    ----------
    noise_gain
        Processing gain computed and applied multiplicatively to the thermal noise level before using it during
        denoising step for the current polarisation.
    count
    """

    class Meta:
        name = "noiseGainListType"

    noise_gain: list[FloatWithPolarisation] = field(
        default_factory=list,
        metadata={"name": "noiseGain", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class NoiseListType:
    """
    Parameters
    ----------
    noise_sequence_list
        Noise parameters derived from the noise packets for the current polarisation.
    count
    """

    class Meta:
        name = "noiseListType"

    noise_sequence_list: list[NoiseSequenceListType] = field(
        default_factory=list,
        metadata={"name": "noiseSequenceList", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class PolarimetricDistortionType:
    """
    Parameters
    ----------
    cross_talk_list
        System cross-talk values
    channel_imbalance_list
        System channel imbalance values
    """

    class Meta:
        name = "polarimetricDistortionType"

    cross_talk_list: Optional[CrossTalkList] = field(
        default=None,
        metadata={
            "name": "crossTalkList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    channel_imbalance_list: Optional[ChannelImbalanceList] = field(
        default=None,
        metadata={
            "name": "channelImbalanceList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class ProcessingGainListType:
    """
    Parameters
    ----------
    processing_gain
        Processing gain computed and applied multiplicatively to the image during processing for the current
        polarisation.
    count
    """

    class Meta:
        name = "processingGainListType"

    processing_gain: list[FloatWithPolarisation] = field(
        default_factory=list,
        metadata={"name": "processingGain", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class QualityParametersType:
    """
    Parameters
    ----------
    missing_ispfraction
        Number of missing ISP, expressed as a normalized fraction of the total number of ISP.
    max_ispgap
        Maximum number of gaps between science data ISPs.
    max_ispgap_threshold
        Threshold on maxISPGap. If maxISPGap exceeds this parameter, the processing will not continue.
    invalid_raw_data_samples
        Number of invalid raw data samples (according to their mean and standard deviation), expressed as a
        normalized fraction of the total number of samples.
    raw_mean_expected
        This parameter specifies the expected mean of the samples in the extracted raw data.
    raw_mean_threshold
        This is the value X such that the measured mean must fall between the rawMeanExpected-X and
        rawMeanExpected+X.
    raw_std_expected
        This parameter specifies the expected standard deviation of the samples in the extracted raw data.
    raw_std_threshold
        This is the value X such that the measured standard deviation must fall between the rawStdExpected-X and
        rawStdExpected+X.
    rfi_tmfraction
        Normalized fraction of image pixels impacted by time-domain RFI.
    max_rfitmpercentage
        Maximum normalized percentage of image pixels impacted by time-domain RFI. If the percentage exceeds this
        parameter, the processing will not continue.
    rfi_fmfraction
        Normalized fraction of image spectrum impacted by frequency-domain RFI.
    max_rfifmpercentage
        Maximum normalized percentage of image spectrum impacted by frequency-domain RFI. If the percentage exceeds
        this parameter, the processing will not continue.
    invalid_drift_fraction
        Number of invalid drift values, expressed as a normalized fraction of the total number of drifts.
    max_invalid_drift_fraction
        Maximum number of invalid drift values allowed, expressed as a normalized fraction of the total number of
        drifts. If the percentage of the invalid drifts does not exceed this value, then the invalid drifts will be
        discarded and only the valid ones will be further used in the processing. Otherwise, all the calculated
        drift values will be discarded and replaced with the corresponding model values.
    invalid_replica_fraction
        Number of invalid replicas, expressed as a normalized fraction of the total number of replicas extracted.
    invalid_dcestimates_fraction
        Number of invalid DC estimated, expressed as a normalized fraction of the total number of DC estimates.
    dc_rmserror_threshold
        Doppler Centroid estimation root mean squared (RMS) error threshold [Hz]. If the RMS error of the Doppler
        Centroid combined estimates is above this threshold they are not used during processing; instead, the
        Doppler Centroid calculated from geometry is used.
    residual_ionospheric_phase_screen_std
        Quality measure of the ionospheric phase estimated by Faraday Rotation, expressed as standard deviation of
        the residual ionospheric phase screen [rad].
    invalid_blocks_percentage
        Normalized fraction of image blocks that need to have estimated an azimuth shift higher than the maximum
        allowed one, used to consider an autofocus estimation valid or not.
    invalid_blocks_percentage_threshold
        Minimum normalized percentage of blocks within the image that need to have estimated an azimuth shift
        smaller than the maxValidShift threshold. If the minimum percentage is not met, no correction based on the
        azimuth shift estimates will be applied.
    polarisation
    """

    class Meta:
        name = "qualityParametersType"

    missing_ispfraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "missingISPFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_ispgap: Optional[int] = field(
        default=None,
        metadata={
            "name": "maxISPGap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_ispgap_threshold: Optional[int] = field(
        default=None,
        metadata={
            "name": "maxISPGapThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_raw_data_samples: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidRawDataSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_mean_expected: Optional[float] = field(
        default=None,
        metadata={
            "name": "rawMeanExpected",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_mean_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "rawMeanThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_std_expected: Optional[float] = field(
        default=None,
        metadata={
            "name": "rawStdExpected",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_std_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "rawStdThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_tmfraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "rfiTMFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_rfitmpercentage: Optional[float] = field(
        default=None,
        metadata={
            "name": "maxRFITMPercentage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_fmfraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "rfiFMFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_rfifmpercentage: Optional[float] = field(
        default=None,
        metadata={
            "name": "maxRFIFMPercentage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_drift_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidDriftFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_invalid_drift_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "maxInvalidDriftFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_replica_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidReplicaFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_dcestimates_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidDCEstimatesFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    dc_rmserror_threshold: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "dcRMSErrorThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    residual_ionospheric_phase_screen_std: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "residualIonosphericPhaseScreenStd",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_blocks_percentage: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidBlocksPercentage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_blocks_percentage_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidBlocksPercentageThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RawDataStatisticsListType:
    """
    Parameters
    ----------
    raw_data_statistics
        Extracted RAW data I and Q channels statistics for the current polarisation.
    count
    """

    class Meta:
        name = "rawDataStatisticsListType"

    raw_data_statistics: list[RawDataStatisticsType] = field(
        default_factory=list,
        metadata={"name": "rawDataStatistics", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RfiIsolatedFmreportListType:
    """
    Parameters
    ----------
    rfi_isolated_fmreport
        Frequency-domain isolated RFI mitigation report.
    count
    """

    class Meta:
        name = "rfiIsolatedFMReportListType"

    rfi_isolated_fmreport: list[RfiIsolatedFmreportType] = field(
        default_factory=list,
        metadata={"name": "rfiIsolatedFMReport", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RfiPersistentFmreportListType:
    """
    Parameters
    ----------
    rfi_persistent_fmreport
        Frequency-domain persistent RFI mitigation report.
    count
    """

    class Meta:
        name = "rfiPersistentFMReportListType"

    rfi_persistent_fmreport: list[RfiPersistentFmreportType] = field(
        default_factory=list,
        metadata={
            "name": "rfiPersistentFMReport",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
            "max_occurs": 4,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RfiTmreportListType:
    """
    Parameters
    ----------
    rfi_tmreport
        Time-domain RFI mitigation report.
    count
    """

    class Meta:
        name = "rfiTMReportListType"

    rfi_tmreport: list[RfiTmreportType] = field(
        default_factory=list,
        metadata={"name": "rfiTMReport", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RxGainListType:
    """
    Parameters
    ----------
    rx_gain
        RX gain, extracted from RAS control message.
    count
        Number of RX gains within the list.
    """

    class Meta:
        name = "rxGainListType"

    rx_gain: list[FloatWithPolarisation] = field(
        default_factory=list,
        metadata={"name": "rxGain", "type": "Element", "namespace": "", "min_occurs": 2, "max_occurs": 2},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class SpectrumProcessingParametersType:
    """
    Parameters
    ----------
    window_type
        Name of the weighting window type used during processing.
    window_coefficient
        Value of the weighting window coefficient used during processing.
    total_bandwidth
        Total available bandwidth [Hz].
    processing_bandwidth
        Bandwidth used during processing [Hz].
    look_bandwidth
        Bandwidth for each look used during processing [Hz].
    number_of_looks
        Number of looks.
    look_overlap
        Overlap between looks [Hz].
    """

    class Meta:
        name = "spectrumProcessingParametersType"

    window_type: Optional[WeightingWindowType] = field(
        default=None,
        metadata={
            "name": "windowType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    window_coefficient: Optional[float] = field(
        default=None,
        metadata={
            "name": "windowCoefficient",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    total_bandwidth: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "totalBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_bandwidth: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "processingBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    look_bandwidth: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "lookBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_looks: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfLooks",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    look_overlap: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "lookOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class TxPulseType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time of transmitted pulse change [UTC].
    tx_pulse_length
        Transmit pulse length [s].
    tx_pulse_start_frequency
        Starting frequency of the transmit pulse [Hz].
    tx_pulse_start_phase
        Starting phase of the transmit pulse [rad].
    tx_pulse_ramp_rate
        The linear rate at which the frequency changes over the pulse duration [Hz/s].
    """

    class Meta:
        name = "txPulseType"

    azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    tx_pulse_length: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "txPulseLength",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tx_pulse_start_frequency: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "txPulseStartFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tx_pulse_start_phase: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "txPulseStartPhase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tx_pulse_ramp_rate: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "txPulseRampRate",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class CoordinateConversionListType:
    """
    Parameters
    ----------
    coordinate_conversion
        Polynomial used to convert image pixels between slant range and ground range. The coefficients used on range
        lines between updates are found by linear interpolation between the updated and previous values. Considering
        the worst case (entire slice processed non-framed, one entry each 10 seconds), list maxOccurs is set to 15.
    count
        Number of coordinate conversion records within the list.
    """

    class Meta:
        name = "coordinateConversionListType"

    coordinate_conversion: list[CoordinateConversionType] = field(
        default_factory=list,
        metadata={"name": "coordinateConversion", "type": "Element", "namespace": "", "max_occurs": 15},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class DcEstimateListType:
    """
    Parameters
    ----------
    dc_estimate
        Doppler Centroid estimate record which contains the Doppler Centroid calculated from geometry and estimated
        from the data, associated signal-to-noise ratio values and indicates which DCE method was used by the IPF
        during image processing. Considering the worst case (entire slice processed non-framed, one entry each 5
        seconds), list maxOccurs is set to 30.
    count
        Number of dcEstimate records within the list.
    """

    class Meta:
        name = "dcEstimateListType"

    dc_estimate: list[DcEstimateType] = field(
        default_factory=list,
        metadata={"name": "dcEstimate", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 150},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FmRateEstimatesListType:
    """
    Parameters
    ----------
    fm_rate_estimate
        Frequency Modulation rate estimate record. Considering the worst case (entire slice processed non-framed,
        one entry each 5 seconds), list maxOccurs is set to 30.
    count
        Number of fmRateEstimate records within the list.
    """

    class Meta:
        name = "fmRateEstimatesListType"

    fm_rate_estimate: list[SlantRangePolynomialType] = field(
        default_factory=list,
        metadata={"name": "fmRateEstimate", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 150},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class InternalCalibrationSequenceListType:
    """
    Parameters
    ----------
    internal_calibration_sequence
        Internal calibration parameters derived from the calibration pulses. Considering the worst case (30 minutes
        datatake, one entry each 10 seconds), list maxOccurs is set to 180.
    polarisation
    count
        Number of internal calibration sequences for the current polarisation within the list.
    """

    class Meta:
        name = "internalCalibrationSequenceListType"

    internal_calibration_sequence: list[InternalCalibrationSequenceType] = field(
        default_factory=list,
        metadata={"name": "internalCalibrationSequence", "type": "Element", "namespace": "", "max_occurs": 180},
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class PrfListType:
    """
    Parameters
    ----------
    prf
        PRF record [Hz]. This record holds the PRF for the given Zero Doppler azimuth time. List maxOccurs is set to
        5, considered as worst case.
    count
        Number of PRF records within the list.
    """

    class Meta:
        name = "prfListType"

    prf: list[StateType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
            "max_occurs": 5,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ProcessingParametersType:
    """
    Parameters
    ----------
    processor_version
        Version of the processor used to generate the product.
    product_generation_time
        Product generation time [UTC].
    processing_mode
        Processing mode (Nominal, PARC).
    orbit_source
        Source of the orbit data used during processing. Set to “Extracted” if the orbit information extracted from
        source packets is used during processing. Set to “Auxiliary” if the orbit information from an input
        auxiliary file is used during processing.
    attitude_source
        Source of the attitude data used during processing. Set to “Extracted” if the attitude information extracted
        from source packets is used during processing. Set to “Auxiliary” if the attitude information from an input
        auxiliary file is used during processing.
    raw_data_correction_flag
        True if raw data correction was performed, False otherwise.
    rfi_detection_flag
        True if RFI detection was performed, False otherwise.
    rfi_correction_flag
        True if RFI correction was performed, False otherwise. Implies activation of rfiDetectionFlag.
    rfi_mitigation_method
        Domain where the RFI mitigation step was performed (Time, Frequency, Time and Frequency, Frequency and
        Time).
    rfi_mask
        RFI mask used for mitigation. Valid values are: "Single", to use the same mask for all the polarizations;
        "Multiple", to use a dedicated mask for each polarization.
    rfi_mask_generation_method
        Polarization-dependent RFI masks combination method (AND, OR). Used only in case rfiMask is set to Single.
    rfi_fmchirp_source
        Chirp source for Frequency Domain detection
    rfi_fmmitigation_method
        Frequency Domain mitigation method
    internal_calibration_estimation_flag
        True if internal calibration estimation was performed, False otherwise.
    internal_calibration_correction_flag
        True if internal calibration correction was performed, False otherwise.
    range_reference_function_source
        Chirp source to be used for range compression (Nominal, Replica or Internal).
    range_compression_method
        Range compression method used during processing (Matched Filter or Inverse Filter).
    extended_swath_processing_flag
        True if processing was extended in the range direction including samples not having the full phase history,
        False otherwise.
    dc_method
        Doppler Centroid estimation method used during processing (Geometry, Combined, Fixed).
    dc_value
        Doppler centroid value used during processing [Hz]. Used only in case dcMethod is set to Fixed.
    antenna_pattern_correction1_flag
        True if antenna pattern correction (first step) was applied, False otherwise.
    antenna_pattern_correction2_flag
        True if antenna pattern correction (second step) was applied, False otherwise.
    antenna_cross_talk_correction_flag
        True if antenna cross-talk correction was applied, False otherwise.
    range_processing_parameters
        Parameters used during range processing.
    azimuth_processing_parameters
        Parameters used during azimuth processing.
    bistatic_delay_correction_flag
        True if bistatic delay correction was applied, False otherwise.
    bistatic_delay_correction_method
        Method used for bistatic delay correction (Bulk or Full).
    range_spreading_loss_compensation_flag
        True if range spreading loss compensation was performed, False otherwise.
    reference_range
        Range spreading loss reference slant range [m]. The range spreading loss is compensated by amplitude scaling
        each range sample by 1/Grsl(R) where: Grsl(R) = cuberoot(rRef/R); and, R = slant range of sample.
    processing_gain_list
        Processing gain computed and applied multiplicatively to the image during processing for all the
        polarisations.
    polarimetric_correction_flag
        True if polarimetric correction has been performed, False otherwise.
    ionosphere_height_defocusing_flag
        True if defocusing at ionosphere height has been performed, False otherwise.
    ionosphere_height_estimation_method
        Ionosphere height estimation method (Feature Tracking, Squint Sensitivity, Model).
    faraday_rotation_correction_flag
        True if Faraday Rotation correction has been performed, False otherwise.
    ionospheric_phase_screen_correction_flag
        True if ionospheric phase screen correction has been performed, False otherwise.
    group_delay_correction_flag
        True if group delay correction has been performed, False otherwise.
    autofocus_flag
        True if autofocus has been performed, False otherwise.
    autofocus_method
        Autofocus method used during processing (Map Drift).
    detection_flag
        True if detection has been performed, False otherwise.
    thermal_denoising_flag
        True if thermal denoising has been performed, False otherwise.
    noise_gain_list
        Processing gain computed and applied multiplicatively to the thermal noise level before using it during
        denoising step for all the polarisations.
    ground_projection_flag
        True if slant range to ground range conversion has been performed, False otherwise.
    """

    class Meta:
        name = "processingParametersType"

    processor_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_generation_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    processing_mode: Optional[ProcessingModeType] = field(
        default=None,
        metadata={
            "name": "processingMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    orbit_source: Optional[OrbitAttitudeSourceType] = field(
        default=None,
        metadata={
            "name": "orbitSource",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    attitude_source: Optional[OrbitAttitudeSourceType] = field(
        default=None,
        metadata={
            "name": "attitudeSource",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_data_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "rawDataCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    rfi_detection_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "rfiDetectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    rfi_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "rfiCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    rfi_mitigation_method: Optional[RfiMitigationMethodType] = field(
        default=None,
        metadata={
            "name": "rfiMitigationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_mask: Optional[RfiMaskType] = field(
        default=None,
        metadata={
            "name": "rfiMask",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_mask_generation_method: Optional[RfiMaskGenerationMethodType] = field(
        default=None,
        metadata={
            "name": "rfiMaskGenerationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_fmchirp_source: Optional[RangeReferenceFunctionType] = field(
        default=None,
        metadata={
            "name": "rfiFMChirpSource",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_fmmitigation_method: Optional[RfiFmmitigationMethodType] = field(
        default=None,
        metadata={
            "name": "rfiFMMitigationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_calibration_estimation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "internalCalibrationEstimationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    internal_calibration_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "internalCalibrationCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    range_reference_function_source: Optional[RangeReferenceFunctionType] = field(
        default=None,
        metadata={
            "name": "rangeReferenceFunctionSource",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_compression_method: Optional[RangeCompressionMethodType] = field(
        default=None,
        metadata={
            "name": "rangeCompressionMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    extended_swath_processing_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "extendedSwathProcessingFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    dc_method: Optional[DcMethodType] = field(
        default=None,
        metadata={
            "name": "dcMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    dc_value: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "dcValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    antenna_pattern_correction1_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "antennaPatternCorrection1Flag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    antenna_pattern_correction2_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "antennaPatternCorrection2Flag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    antenna_cross_talk_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "antennaCrossTalkCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    range_processing_parameters: Optional[SpectrumProcessingParametersType] = field(
        default=None,
        metadata={
            "name": "rangeProcessingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_processing_parameters: Optional[SpectrumProcessingParametersType] = field(
        default=None,
        metadata={
            "name": "azimuthProcessingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    bistatic_delay_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "bistaticDelayCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    bistatic_delay_correction_method: Optional[BistaticDelayCorrectionMethodType] = field(
        default=None,
        metadata={
            "name": "bistaticDelayCorrectionMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_spreading_loss_compensation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "rangeSpreadingLossCompensationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    reference_range: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "referenceRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_gain_list: Optional[ProcessingGainListType] = field(
        default=None,
        metadata={
            "name": "processingGainList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarimetric_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "polarimetricCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    ionosphere_height_defocusing_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightDefocusingFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    ionosphere_height_estimation_method: Optional[IonosphereHeightEstimationMethodType] = field(
        default=None,
        metadata={
            "name": "ionosphereHeightEstimationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    faraday_rotation_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "faradayRotationCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    ionospheric_phase_screen_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "ionosphericPhaseScreenCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    group_delay_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "groupDelayCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    autofocus_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "autofocusFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    autofocus_method: Optional[AutofocusMethodType] = field(
        default=None,
        metadata={
            "name": "autofocusMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    detection_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "detectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    thermal_denoising_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "thermalDenoisingFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    noise_gain_list: Optional[NoiseGainListType] = field(
        default=None,
        metadata={
            "name": "noiseGainList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ground_projection_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "groundProjectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class QualityParametersListType:
    """
    Parameters
    ----------
    quality_parameters
        Quality parameters for the current polarisation.
    count
    """

    class Meta:
        name = "qualityParametersListType"

    quality_parameters: list[QualityParametersType] = field(
        default_factory=list,
        metadata={"name": "qualityParameters", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RadiometricCalibrationType:
    """
    Parameters
    ----------
    absolute_calibration_constant_list
        Absolute calibration constant. Already applied to the image during processing for all the polarisations.
    """

    class Meta:
        name = "radiometricCalibrationType"

    absolute_calibration_constant_list: Optional[CalibrationConstantListType] = field(
        default=None,
        metadata={
            "name": "absoluteCalibrationConstantList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class RawDataAnalysisType:
    """
    Parameters
    ----------
    error_counters
        Error counters computed starting from input Instrument Source Packets stream.
    raw_data_statistics_list
        Extracted RAW data I and Q channels statistics for all the polarisations.
    """

    class Meta:
        name = "rawDataAnalysisType"

    error_counters: Optional[ErrorCountersType] = field(
        default=None,
        metadata={
            "name": "errorCounters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_data_statistics_list: Optional[RawDataStatisticsListType] = field(
        default=None,
        metadata={
            "name": "rawDataStatisticsList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class RfiMitigationType:
    """
    Parameters
    ----------
    rfi_tmreport_list
        Time-domain RFI mitigation report list. This element is present only if time-domain RFI mitigation is
        performed.
    rfi_isolated_fmreport_list
        Frequency-domain isolated RFI mitigation report list. This element is present only if frequency-domain RFI
        mitigation is performed.
    rfi_persistent_fmreport_list
        Frequency-domain persistent RFI mitigation report list. This element is present only if frequency-domain RFI
        mitigation is performed.
    """

    class Meta:
        name = "rfiMitigationType"

    rfi_tmreport_list: Optional[RfiTmreportListType] = field(
        default=None,
        metadata={
            "name": "rfiTMReportList",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_isolated_fmreport_list: Optional[RfiIsolatedFmreportListType] = field(
        default=None,
        metadata={
            "name": "rfiIsolatedFMReportList",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_persistent_fmreport_list: Optional[RfiPersistentFmreportListType] = field(
        default=None,
        metadata={
            "name": "rfiPersistentFMReportList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class SwlListType:
    """
    Parameters
    ----------
    swl
        SWL record [s]. This record holds the SWL for the given Zero Doppler azimuth time. Considering the worst
        case (entire slice processed non-framed, one change each 10 seconds), list maxOccurs is set to 15.
    count
        Number of SWL records within the list.
    """

    class Meta:
        name = "swlListType"

    swl: list[StateType] = field(
        default_factory=list, metadata={"type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 15}
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class SwpListType:
    """
    Parameters
    ----------
    swp
        SWP record [s]. This record holds the SWP for the given Zero Doppler azimuth time. Considering the worst
        case (entire slice processed non-framed, one change each 10 seconds), list maxOccurs is set to 15.
    count
        Number of SWP records within the list.
    """

    class Meta:
        name = "swpListType"

    swp: list[StateType] = field(
        default_factory=list, metadata={"type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 15}
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class TxPulseListType:
    """
    Parameters
    ----------
    tx_pulse
        Transmitted pulse record. This record holds the nominal transmitted pulse information for the given Zero
        Doppler azimuth time. List maxOccurs is set to 5, considered as worst case.
    count
        Number of transmitted pulse records within the list.
    """

    class Meta:
        name = "txPulseListType"

    tx_pulse: list[TxPulseType] = field(
        default_factory=list,
        metadata={
            "name": "txPulse",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
            "max_occurs": 5,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class DopplerParametersType:
    """
    Parameters
    ----------
    dc_estimate_list
        List of Doppler Centroid estimates that have been calculated during image processing. The list contains an
        entry for each Doppler Centroid estimate made along azimuth, i.e., each N seconds, where N is configurable.
    fm_rate_estimate_list
        List of Frequency Modulation rate estimates that have been calculated during image processing. The list
        contains an entry for each Frequency Modulation rate estimate made along azimuth, i.e., each N seconds,
        where N is configurable.
    """

    class Meta:
        name = "dopplerParametersType"

    dc_estimate_list: Optional[DcEstimateListType] = field(
        default=None,
        metadata={
            "name": "dcEstimateList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    fm_rate_estimate_list: Optional[FmRateEstimatesListType] = field(
        default=None,
        metadata={
            "name": "fmRateEstimateList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class InstrumentParametersType:
    """
    Parameters
    ----------
    first_line_sensing_time_list
        Sensing time of first line of the input RAW data for all the polarisations [UTC].
    last_line_sensing_time_list
        Sensing time of last line of the input RAW data for all the polarisations [UTC].
    number_of_input_samples
        Number of samples of the input RAW data.
    number_of_input_lines
        Number of lines of the input RAW data.
    swp_list
        List of Sampling Window Position changes (SWP) [s]. The list contains an entry for each on-board SWP update.
    swl_list
        List of Sampling Window Length changes (SWL) [s]. The list contains an entry for each on-board SWL update.
    prf_list
        List of Pulse Repetition Frequency changes (PRF) [Hz]. The list contains an entry for each on-board PRF
        update.
    rank
        Number of PRIs between transmitted pulse and received echo.
    tx_pulse_list
        List of transmitted pulses. The list contains an entry for each on-board transmitted pulse update.
    instrument_configuration_id
        Instrument configuration identifier.
    radar_carrier_frequency
        Radar carrier frequency [Hz].
    rx_gain_list
        RX gain list.
    preamble_flag
        True if input RAW data contain preamble sequence, False otherwise.
    postamble_flag
        True if input RAW data contain postamble sequence, False otherwise.
    interleaved_calibration_flag
        True if input RAW data contain interleaved calibration sequences, False otherwise.
    data_format
        Data format for instrument samples. There is one element corresponding to the data format for each packet
        type in the segment.
    """

    class Meta:
        name = "instrumentParametersType"

    first_line_sensing_time_list: Optional[FirstLineSensingTimeListType] = field(
        default=None,
        metadata={
            "name": "firstLineSensingTimeList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    last_line_sensing_time_list: Optional[LastLineSensingTimeListType] = field(
        default=None,
        metadata={
            "name": "lastLineSensingTimeList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_input_samples: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfInputSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_input_lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfInputLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swp_list: Optional[SwpListType] = field(
        default=None,
        metadata={
            "name": "swpList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swl_list: Optional[SwlListType] = field(
        default=None,
        metadata={
            "name": "swlList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    prf_list: Optional[PrfListType] = field(
        default=None,
        metadata={
            "name": "prfList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rank: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tx_pulse_list: Optional[TxPulseListType] = field(
        default=None,
        metadata={
            "name": "txPulseList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    instrument_configuration_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "instrumentConfigurationID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    radar_carrier_frequency: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "radarCarrierFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rx_gain_list: Optional[RxGainListType] = field(
        default=None,
        metadata={
            "name": "rxGainList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    preamble_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "preambleFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    postamble_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "postambleFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    interleaved_calibration_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "interleavedCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    data_format: Optional[DataFormatType] = field(
        default=None,
        metadata={
            "name": "dataFormat",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class InternalCalibrationParametersListType:
    """
    Parameters
    ----------
    internal_calibration_sequence_list
        Internal calibration parameters derived from the calibration pulses for the current polarisation.
    count
    """

    class Meta:
        name = "internalCalibrationParametersListType"

    internal_calibration_sequence_list: list[InternalCalibrationSequenceListType] = field(
        default_factory=list,
        metadata={
            "name": "internalCalibrationSequenceList",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
            "max_occurs": 4,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class QualityType:
    """
    Parameters
    ----------
    overall_product_quality_index
        Overall product quality index. This annotation is calculated based on specific quality parameters and gives
        an overall quality value to the product. Equal to 0 for valid products and to 1 for invalid ones.
    quality_parameters_list
        Quality parameters list. The list contains an entry for each polarisation.
    """

    class Meta:
        name = "qualityType"

    overall_product_quality_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "overallProductQualityIndex",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    quality_parameters_list: Optional[QualityParametersListType] = field(
        default=None,
        metadata={
            "name": "qualityParametersList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class SarImageType:
    """
    Parameters
    ----------
    first_sample_slant_range_time
        Two-way slant range time to first sample of the image [s].
    last_sample_slant_range_time
        Two-way slant range time to last sample of the image [s].
    first_line_azimuth_time
        Zero Doppler azimuth time of the first line of the image [UTC].
    last_line_azimuth_time
        Zero Doppler azimuth time of the last line of the image [UTC].
    range_time_interval
        Time spacing between range samples of the image [s]. For L1a products, this value is the inverse of the
        range sampling frequency. For L1b products, this value is obtained dividing rangePixelSpacing for half the
        speed of light.
    azimuth_time_interval
        Time spacing between azimuth lines of the image [s].
    range_pixel_spacing
        Pixel spacing between range samples [m].
    azimuth_pixel_spacing
        Nominal pixel spacing between azimuth lines [m].
    number_of_samples
        Total number of samples in the image (image width).
    number_of_lines
        Total number of lines in the image (image length).
    projection
        Projection of the image, either Slant Range or Ground Range.
    range_coordinate_conversion
        List of coordinateConversion records that describe conversion between the slant range and ground range
        coordinate systems. The list contains an entry for each 10 seconds. This list applies to and is filled in
        only for DGM products and therefore has a length of zero for SCS products.
    datum
        Datum used during processing.
    footprint
        Image footprint, expressed as a list of latitude and longitude values separated by spaces [deg]. Values are
        specified for each of the 4 corner coordinates of the scene quicklook, starting with the first-right in
        flight direction and proceeding CCW (same convention used in MPH and overlay ADS).
    pixel_representation
        Representation of the image pixels within the image MDS (I Q, Abs Phase, Abs, Pow Phase, Pow).
    pixel_type
        Data type of output pixels within the image MDS.
    pixel_quantity
        Physical quantity stored in output data (Beta-Nought, Sigma-Nought or Gamma-Nought).
    no_data_value
        Pixel value in case of invalid data.
    """

    class Meta:
        name = "sarImageType"

    first_sample_slant_range_time: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "firstSampleSlantRangeTime",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    last_sample_slant_range_time: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "lastSampleSlantRangeTime",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    first_line_azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "firstLineAzimuthTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    last_line_azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "lastLineAzimuthTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    range_time_interval: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "rangeTimeInterval",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_time_interval: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "azimuthTimeInterval",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_pixel_spacing: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "rangePixelSpacing",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_pixel_spacing: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "azimuthPixelSpacing",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_samples: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    projection: Optional[ProjectionType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_coordinate_conversion: Optional[CoordinateConversionListType] = field(
        default=None,
        metadata={
            "name": "rangeCoordinateConversion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    datum: Optional[DatumType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    footprint: Optional[FloatArrayWithUnits] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixel_representation: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "pixelRepresentation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixel_type: Optional[PixelTypeType] = field(
        default=None,
        metadata={
            "name": "pixelType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixel_quantity: Optional[PixelQuantityType] = field(
        default=None,
        metadata={
            "name": "pixelQuantity",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    no_data_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "noDataValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class InternalCalibrationType:
    """
    Parameters
    ----------
    internal_calibration_parameters_used
        Internal calibration parameters effectively used during processing (Model or Extracted).
    range_reference_function_used
        Chirp source effectively used for range compression (Nominal, Replica or Internal). Can be different from
        rangeReferenceFunctionSource.
    noise_parameters_used
        Noise parameters used during processing (Model or Extracted).
    internal_calibration_parameters_list
        Internal calibration parameters list. This element contains lists of internal calibration parameters
        calculated from the calibration pulses extracted from the downlink. The list contains an entry for each
        calibration sequence contained in the input data and it is organised per polarisation. If the list is empty,
        the nominal parameters values in the instrument auxiliary data file will be used instead.
    noise_list
        Noise list. This element contains lists of noise parameters derived from the noise ISPs. The list contains
        an entry for each noise update made along azimuth and it is organised per polarisation. If the list is
        empty, the nominal noise value in the instrument auxiliary data file will be used instead.
    """

    class Meta:
        name = "internalCalibrationType"

    internal_calibration_parameters_used: Optional[InternalCalibrationSourceType] = field(
        default=None,
        metadata={
            "name": "internalCalibrationParametersUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_reference_function_used: Optional[RangeReferenceFunctionType] = field(
        default=None,
        metadata={
            "name": "rangeReferenceFunctionUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    noise_parameters_used: Optional[InternalCalibrationSourceType] = field(
        default=None,
        metadata={
            "name": "noiseParametersUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_calibration_parameters_list: Optional[InternalCalibrationParametersListType] = field(
        default=None,
        metadata={
            "name": "internalCalibrationParametersList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    noise_list: Optional[NoiseListType] = field(
        default=None,
        metadata={
            "name": "noiseList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
