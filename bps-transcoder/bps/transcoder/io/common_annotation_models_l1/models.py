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


@dataclass(kw_only=True)
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

    num_isp_header_errors: int = field(
        metadata={
            "name": "numIspHeaderErrors",
            "type": "Element",
            "namespace": "",
        },
    )
    num_isp_missing: int = field(
        metadata={
            "name": "numIspMissing",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    azimuth_time: str = field(
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    noise_power_correction_factor: float = field(
        metadata={
            "name": "noisePowerCorrectionFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_noise_lines: int = field(
        metadata={
            "name": "numberOfNoiseLines",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class PowerRatioType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time of the in/out-of-band power ratio measurement [UTC].
    value
        In/out-of-band power ratio measurement.
    """

    class Meta:
        name = "powerRatioType"

    azimuth_time: str = field(
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    value: float = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class PowerRatioListType:
    """
    Parameters
    ----------
    power_ratio
        In/out-of-band power ratio measurements. Considering the worst case (entire slice processed non-framed, one
        entry each 5 seconds), list maxOccurs is set to 30.
    polarisation
    count
        Number of in/out-of-band power ratio measurements for the current polarisation within the list.
    """

    class Meta:
        name = "powerRatioListType"

    power_ratio: list[PowerRatioType] = field(
        default_factory=list,
        metadata={"name": "powerRatio", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 30},
    )
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    i_bias: float = field(
        metadata={
            "name": "iBias",
            "type": "Element",
            "namespace": "",
        },
    )
    q_bias: float = field(
        metadata={
            "name": "qBias",
            "type": "Element",
            "namespace": "",
        },
    )
    iq_quadrature_departure: float = field(
        metadata={
            "name": "iqQuadratureDeparture",
            "type": "Element",
            "namespace": "",
        },
    )
    iq_gain_imbalance: float = field(
        metadata={
            "name": "iqGainImbalance",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    percentage_affected_lines: float = field(
        metadata={
            "name": "percentageAffectedLines",
            "type": "Element",
            "namespace": "",
        },
    )
    max_percentage_affected_bw: float = field(
        metadata={
            "name": "maxPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
        },
    )
    avg_percentage_affected_bw: float = field(
        metadata={
            "name": "avgPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    max_percentage_affected_bw: float = field(
        metadata={
            "name": "maxPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
        },
    )
    avg_percentage_affected_bw: float = field(
        metadata={
            "name": "avgPercentageAffectedBW",
            "type": "Element",
            "namespace": "",
        },
    )
    percentage_affected_lines: float = field(
        metadata={
            "name": "percentageAffectedLines",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    percentage_affected_lines: float = field(
        metadata={
            "name": "percentageAffectedLines",
            "type": "Element",
            "namespace": "",
        },
    )
    avg_percentage_affected_samples: float = field(
        metadata={
            "name": "avgPercentageAffectedSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    max_percentage_affected_samples: float = field(
        metadata={
            "name": "maxPercentageAffectedSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    mission: MissionType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    swath: SwathType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    product_type: ProductType = field(
        metadata={
            "name": "productType",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation_list: PolarisationListType = field(
        metadata={
            "name": "polarisationList",
            "type": "Element",
            "namespace": "",
        },
    )
    start_time: str = field(
        metadata={
            "name": "startTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    stop_time: str = field(
        metadata={
            "name": "stopTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    mission_phase_id: MissionPhaseIdtype = field(
        metadata={
            "name": "missionPhaseID",
            "type": "Element",
            "namespace": "",
        },
    )
    drift_phase_flag: str = field(
        metadata={"name": "driftPhaseFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    sensor_mode: SensorModeType = field(
        metadata={
            "name": "sensorMode",
            "type": "Element",
            "namespace": "",
        },
    )
    global_coverage_id: int = field(
        metadata={
            "name": "globalCoverageID",
            "type": "Element",
            "namespace": "",
        },
    )
    major_cycle_id: int = field(
        metadata={
            "name": "majorCycleID",
            "type": "Element",
            "namespace": "",
        },
    )
    repeat_cycle_id: int = field(
        metadata={
            "name": "repeatCycleID",
            "type": "Element",
            "namespace": "",
        },
    )
    absolute_orbit_number: int = field(
        metadata={
            "name": "absoluteOrbitNumber",
            "type": "Element",
            "namespace": "",
        },
    )
    relative_orbit_number: int = field(
        metadata={
            "name": "relativeOrbitNumber",
            "type": "Element",
            "namespace": "",
        },
    )
    orbit_pass: OrbitPassType = field(
        metadata={
            "name": "orbitPass",
            "type": "Element",
            "namespace": "",
        },
    )
    platform_heading: DoubleWithUnit = field(
        metadata={
            "name": "platformHeading",
            "type": "Element",
            "namespace": "",
        },
    )
    data_take_id: int = field(
        metadata={
            "name": "dataTakeID",
            "type": "Element",
            "namespace": "",
        },
    )
    frame: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    product_composition: ProductCompositionType = field(
        metadata={
            "name": "productComposition",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    azimuth_time: str = field(
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    t0: DoubleWithUnit = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    sr0: DoubleWithUnit = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    slant_to_ground_coefficients: DoubleArray = field(
        metadata={
            "name": "slantToGroundCoefficients",
            "type": "Element",
            "namespace": "",
        },
    )
    gr0: DoubleWithUnit = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    ground_to_slant_coefficients: DoubleArray = field(
        metadata={
            "name": "groundToSlantCoefficients",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    echo_format: DataFormatModeType = field(
        metadata={
            "name": "echoFormat",
            "type": "Element",
            "namespace": "",
        },
    )
    calibration_format: DataFormatModeType = field(
        metadata={
            "name": "calibrationFormat",
            "type": "Element",
            "namespace": "",
        },
    )
    noise_format: DataFormatModeType = field(
        metadata={
            "name": "noiseFormat",
            "type": "Element",
            "namespace": "",
        },
    )
    mean_bit_rate: DoubleWithUnit = field(
        metadata={
            "name": "meanBitRate",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    azimuth_time: str = field(
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    t0: DoubleWithUnit = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    geometry_dcpolynomial: FloatArray = field(
        metadata={
            "name": "geometryDCPolynomial",
            "type": "Element",
            "namespace": "",
        },
    )
    combined_dcpolynomial: FloatArray = field(
        metadata={
            "name": "combinedDCPolynomial",
            "type": "Element",
            "namespace": "",
        },
    )
    combined_dcvalues: DoubleArrayWithUnits = field(
        metadata={
            "name": "combinedDCValues",
            "type": "Element",
            "namespace": "",
        },
    )
    combined_dcslant_range_times: DoubleArrayWithUnits = field(
        metadata={
            "name": "combinedDCSlantRangeTimes",
            "type": "Element",
            "namespace": "",
        },
    )
    combined_dcrmserror: DoubleWithUnit = field(
        metadata={
            "name": "combinedDCRMSError",
            "type": "Element",
            "namespace": "",
        },
    )
    combined_dcrmserror_above_threshold: str = field(
        metadata={
            "name": "combinedDCRMSErrorAboveThreshold",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    height_model: HeightModelType = field(
        metadata={
            "name": "heightModel",
            "type": "Element",
            "namespace": "",
        },
    )
    height_model_used_flag: str = field(
        metadata={"name": "heightModelUsedFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    roll_bias: FloatWithUnit = field(
        metadata={
            "name": "rollBias",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class InOutBandPowerRatioListType:
    """
    Parameters
    ----------
    power_ratio_list
        In/out-of-band power ratio measurements for the current polarisation.
    count
    """

    class Meta:
        name = "inOutBandPowerRatioListType"

    power_ratio_list: list[PowerRatioListType] = field(
        default_factory=list,
        metadata={"name": "powerRatioList", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    azimuth_time: str = field(
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    drift_amplitude: float = field(
        metadata={
            "name": "driftAmplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    drift_phase: FloatWithUnit = field(
        metadata={
            "name": "driftPhase",
            "type": "Element",
            "namespace": "",
        },
    )
    model_drift_amplitude: float = field(
        metadata={
            "name": "modelDriftAmplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    model_drift_phase: FloatWithUnit = field(
        metadata={
            "name": "modelDriftPhase",
            "type": "Element",
            "namespace": "",
        },
    )
    relative_drift_valid_flag: str = field(
        metadata={"name": "relativeDriftValidFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    absolute_drift_valid_flag: str = field(
        metadata={"name": "absoluteDriftValidFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    cross_correlation_bandwidth: FloatWithUnit = field(
        metadata={
            "name": "crossCorrelationBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    cross_correlation_pslr: FloatWithUnit = field(
        metadata={
            "name": "crossCorrelationPslr",
            "type": "Element",
            "namespace": "",
        },
    )
    cross_correlation_islr: FloatWithUnit = field(
        metadata={
            "name": "crossCorrelationIslr",
            "type": "Element",
            "namespace": "",
        },
    )
    cross_correlation_peak_location: FloatWithUnit = field(
        metadata={
            "name": "crossCorrelationPeakLocation",
            "type": "Element",
            "namespace": "",
        },
    )
    reconstructed_replica_valid_flag: str = field(
        metadata={
            "name": "reconstructedReplicaValidFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    internal_time_delay: FloatWithUnit = field(
        metadata={
            "name": "internalTimeDelay",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_tx_channel_imbalance_amplitude: float = field(
        metadata={
            "name": "internalTxChannelImbalanceAmplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_tx_channel_imbalance_phase: FloatWithUnit = field(
        metadata={
            "name": "internalTxChannelImbalancePhase",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_rx_channel_imbalance_amplitude: float = field(
        metadata={
            "name": "internalRxChannelImbalanceAmplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_rx_channel_imbalance_phase: FloatWithUnit = field(
        metadata={
            "name": "internalRxChannelImbalancePhase",
            "type": "Element",
            "namespace": "",
        },
    )
    transmit_power_tracking_d1_amplitude: float = field(
        metadata={
            "name": "transmitPowerTrackingD1Amplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    transmit_power_tracking_d1_phase: FloatWithUnit = field(
        metadata={
            "name": "transmitPowerTrackingD1Phase",
            "type": "Element",
            "namespace": "",
        },
    )
    receive_power_tracking_d1_amplitude: float = field(
        metadata={
            "name": "receivePowerTrackingD1Amplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    receive_power_tracking_d1_phase: FloatWithUnit = field(
        metadata={
            "name": "receivePowerTrackingD1Phase",
            "type": "Element",
            "namespace": "",
        },
    )
    transmit_power_tracking_d2_amplitude: float = field(
        metadata={
            "name": "transmitPowerTrackingD2Amplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    transmit_power_tracking_d2_phase: FloatWithUnit = field(
        metadata={
            "name": "transmitPowerTrackingD2Phase",
            "type": "Element",
            "namespace": "",
        },
    )
    receive_power_tracking_d2_amplitude: float = field(
        metadata={
            "name": "receivePowerTrackingD2Amplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    receive_power_tracking_d2_phase: FloatWithUnit = field(
        metadata={
            "name": "receivePowerTrackingD2Phase",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    ionosphere_height_used: FloatWithUnit = field(
        metadata={
            "name": "ionosphereHeightUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_estimated: FloatWithUnit = field(
        metadata={
            "name": "ionosphereHeightEstimated",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_estimation_method_selected: IonosphereHeightEstimationMethodType = field(
        metadata={
            "name": "ionosphereHeightEstimationMethodSelected",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_estimation_latitude_value: FloatWithUnit = field(
        metadata={
            "name": "ionosphereHeightEstimationLatitudeValue",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_estimation_flag: str = field(
        metadata={
            "name": "ionosphereHeightEstimationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    ionosphere_height_estimation_method_used: IonosphereHeightEstimationMethodType = field(
        metadata={
            "name": "ionosphereHeightEstimationMethodUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    gaussian_filter_computation_flag: str = field(
        metadata={
            "name": "gaussianFilterComputationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    faraday_rotation_correction_applied: str = field(
        metadata={
            "name": "faradayRotationCorrectionApplied",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    autofocus_shifts_applied: str = field(
        metadata={"name": "autofocusShiftsApplied", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    cross_talk_list: CrossTalkList = field(
        metadata={
            "name": "crossTalkList",
            "type": "Element",
            "namespace": "",
        },
    )
    channel_imbalance_list: ChannelImbalanceList = field(
        metadata={
            "name": "channelImbalanceList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    missing_ispfraction: float = field(
        metadata={
            "name": "missingISPFraction",
            "type": "Element",
            "namespace": "",
        },
    )
    max_ispgap: int = field(
        metadata={
            "name": "maxISPGap",
            "type": "Element",
            "namespace": "",
        },
    )
    max_ispgap_threshold: int = field(
        metadata={
            "name": "maxISPGapThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_raw_data_samples: float = field(
        metadata={
            "name": "invalidRawDataSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_mean_expected: float = field(
        metadata={
            "name": "rawMeanExpected",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_mean_threshold: float = field(
        metadata={
            "name": "rawMeanThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_std_expected: float = field(
        metadata={
            "name": "rawStdExpected",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_std_threshold: float = field(
        metadata={
            "name": "rawStdThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_tmfraction: float = field(
        metadata={
            "name": "rfiTMFraction",
            "type": "Element",
            "namespace": "",
        },
    )
    max_rfitmpercentage: float = field(
        metadata={
            "name": "maxRFITMPercentage",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_fmfraction: float = field(
        metadata={
            "name": "rfiFMFraction",
            "type": "Element",
            "namespace": "",
        },
    )
    max_rfifmpercentage: float = field(
        metadata={
            "name": "maxRFIFMPercentage",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_drift_fraction: float = field(
        metadata={
            "name": "invalidDriftFraction",
            "type": "Element",
            "namespace": "",
        },
    )
    max_invalid_drift_fraction: float = field(
        metadata={
            "name": "maxInvalidDriftFraction",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_replica_fraction: float = field(
        metadata={
            "name": "invalidReplicaFraction",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_dcestimates_fraction: float = field(
        metadata={
            "name": "invalidDCEstimatesFraction",
            "type": "Element",
            "namespace": "",
        },
    )
    dc_rmserror_threshold: FloatWithUnit = field(
        metadata={
            "name": "dcRMSErrorThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    residual_ionospheric_phase_screen_std: FloatWithUnit = field(
        metadata={
            "name": "residualIonosphericPhaseScreenStd",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_blocks_percentage: float = field(
        metadata={
            "name": "invalidBlocksPercentage",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_blocks_percentage_threshold: float = field(
        metadata={
            "name": "invalidBlocksPercentageThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    window_type: WeightingWindowType = field(
        metadata={
            "name": "windowType",
            "type": "Element",
            "namespace": "",
        },
    )
    window_coefficient: float = field(
        metadata={
            "name": "windowCoefficient",
            "type": "Element",
            "namespace": "",
        },
    )
    total_bandwidth: DoubleWithUnit = field(
        metadata={
            "name": "totalBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    processing_bandwidth: DoubleWithUnit = field(
        metadata={
            "name": "processingBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    look_bandwidth: DoubleWithUnit = field(
        metadata={
            "name": "lookBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_looks: int = field(
        metadata={
            "name": "numberOfLooks",
            "type": "Element",
            "namespace": "",
        },
    )
    look_overlap: DoubleWithUnit = field(
        metadata={
            "name": "lookOverlap",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    azimuth_time: str = field(
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    tx_pulse_length: DoubleWithUnit = field(
        metadata={
            "name": "txPulseLength",
            "type": "Element",
            "namespace": "",
        },
    )
    tx_pulse_start_frequency: DoubleWithUnit = field(
        metadata={
            "name": "txPulseStartFrequency",
            "type": "Element",
            "namespace": "",
        },
    )
    tx_pulse_start_phase: DoubleWithUnit = field(
        metadata={
            "name": "txPulseStartPhase",
            "type": "Element",
            "namespace": "",
        },
    )
    tx_pulse_ramp_rate: DoubleWithUnit = field(
        metadata={
            "name": "txPulseRampRate",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class DcEstimateListType:
    """
    Parameters
    ----------
    dc_estimate
        Doppler Centroid estimate record which contains the Doppler Centroid calculated from geometry and estimated
        from the data, associated signal-to-noise ratio values and indicates which DCE method was used by the IPF
        during image processing. Considering the worst case (entire slice processed non-framed, one entry each 1
        second), list maxOccurs is set to 150.
    count
        Number of dcEstimate records within the list.
    """

    class Meta:
        name = "dcEstimateListType"

    dc_estimate: list[DcEstimateType] = field(
        default_factory=list,
        metadata={"name": "dcEstimate", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 150},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class FmRateEstimatesListType:
    """
    Parameters
    ----------
    fm_rate_estimate
        Frequency Modulation rate estimate record. Considering the worst case (entire slice processed non-framed,
        one entry each 1 second), list maxOccurs is set to 150.
    count
        Number of fmRateEstimate records within the list.
    """

    class Meta:
        name = "fmRateEstimatesListType"

    fm_rate_estimate: list[SlantRangePolynomialType] = field(
        default_factory=list,
        metadata={"name": "fmRateEstimate", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 150},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    processor_version: str = field(
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
        },
    )
    product_generation_time: str = field(
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    processing_mode: ProcessingModeType = field(
        metadata={
            "name": "processingMode",
            "type": "Element",
            "namespace": "",
        },
    )
    orbit_source: OrbitAttitudeSourceType = field(
        metadata={
            "name": "orbitSource",
            "type": "Element",
            "namespace": "",
        },
    )
    attitude_source: OrbitAttitudeSourceType = field(
        metadata={
            "name": "attitudeSource",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_data_correction_flag: str = field(
        metadata={"name": "rawDataCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    rfi_detection_flag: str = field(
        metadata={"name": "rfiDetectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    rfi_correction_flag: str = field(
        metadata={"name": "rfiCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    rfi_mitigation_method: RfiMitigationMethodType = field(
        metadata={
            "name": "rfiMitigationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_mask: RfiMaskType = field(
        metadata={
            "name": "rfiMask",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_mask_generation_method: RfiMaskGenerationMethodType = field(
        metadata={
            "name": "rfiMaskGenerationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_fmchirp_source: RangeReferenceFunctionType = field(
        metadata={
            "name": "rfiFMChirpSource",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_fmmitigation_method: RfiFmmitigationMethodType = field(
        metadata={
            "name": "rfiFMMitigationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_calibration_estimation_flag: str = field(
        metadata={
            "name": "internalCalibrationEstimationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    internal_calibration_correction_flag: str = field(
        metadata={
            "name": "internalCalibrationCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    range_reference_function_source: RangeReferenceFunctionType = field(
        metadata={
            "name": "rangeReferenceFunctionSource",
            "type": "Element",
            "namespace": "",
        },
    )
    range_compression_method: RangeCompressionMethodType = field(
        metadata={
            "name": "rangeCompressionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    extended_swath_processing_flag: str = field(
        metadata={
            "name": "extendedSwathProcessingFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    dc_method: DcMethodType = field(
        metadata={
            "name": "dcMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    dc_value: DoubleWithUnit = field(
        metadata={
            "name": "dcValue",
            "type": "Element",
            "namespace": "",
        },
    )
    antenna_pattern_correction1_flag: str = field(
        metadata={
            "name": "antennaPatternCorrection1Flag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    antenna_pattern_correction2_flag: str = field(
        metadata={
            "name": "antennaPatternCorrection2Flag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    antenna_cross_talk_correction_flag: str = field(
        metadata={
            "name": "antennaCrossTalkCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    range_processing_parameters: SpectrumProcessingParametersType = field(
        metadata={
            "name": "rangeProcessingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_processing_parameters: SpectrumProcessingParametersType = field(
        metadata={
            "name": "azimuthProcessingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    bistatic_delay_correction_flag: str = field(
        metadata={
            "name": "bistaticDelayCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    bistatic_delay_correction_method: BistaticDelayCorrectionMethodType = field(
        metadata={
            "name": "bistaticDelayCorrectionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    range_spreading_loss_compensation_flag: str = field(
        metadata={
            "name": "rangeSpreadingLossCompensationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    reference_range: DoubleWithUnit = field(
        metadata={
            "name": "referenceRange",
            "type": "Element",
            "namespace": "",
        },
    )
    processing_gain_list: ProcessingGainListType = field(
        metadata={
            "name": "processingGainList",
            "type": "Element",
            "namespace": "",
        },
    )
    polarimetric_correction_flag: str = field(
        metadata={
            "name": "polarimetricCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    ionosphere_height_defocusing_flag: str = field(
        metadata={
            "name": "ionosphereHeightDefocusingFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    ionosphere_height_estimation_method: IonosphereHeightEstimationMethodType = field(
        metadata={
            "name": "ionosphereHeightEstimationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    faraday_rotation_correction_flag: str = field(
        metadata={
            "name": "faradayRotationCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    ionospheric_phase_screen_correction_flag: str = field(
        metadata={
            "name": "ionosphericPhaseScreenCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    group_delay_correction_flag: str = field(
        metadata={"name": "groupDelayCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    autofocus_flag: str = field(
        metadata={"name": "autofocusFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    autofocus_method: AutofocusMethodType = field(
        metadata={
            "name": "autofocusMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    detection_flag: str = field(
        metadata={"name": "detectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    thermal_denoising_flag: str = field(
        metadata={"name": "thermalDenoisingFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    noise_gain_list: NoiseGainListType = field(
        metadata={
            "name": "noiseGainList",
            "type": "Element",
            "namespace": "",
        },
    )
    ground_projection_flag: str = field(
        metadata={"name": "groundProjectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RadiometricCalibrationType:
    """
    Parameters
    ----------
    absolute_calibration_constant_list
        Absolute calibration constant. Already applied to the image during processing for all the polarisations.
    """

    class Meta:
        name = "radiometricCalibrationType"

    absolute_calibration_constant_list: CalibrationConstantListType = field(
        metadata={
            "name": "absoluteCalibrationConstantList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class RawDataAnalysisType:
    """
    Parameters
    ----------
    error_counters
        Error counters computed starting from input Instrument Source Packets stream.
    raw_data_statistics_list
        Extracted RAW data I and Q channels statistics for all the polarisations.
    in_out_band_power_ratio_list
        In/out-of-band power ratio list. This element contains lists of in/out-of-band power ratio measurements. The
        list contains an entry for each measurement made along azimuth and it is organised per polarisation.
    """

    class Meta:
        name = "rawDataAnalysisType"

    error_counters: ErrorCountersType = field(
        metadata={
            "name": "errorCounters",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_data_statistics_list: RawDataStatisticsListType = field(
        metadata={
            "name": "rawDataStatisticsList",
            "type": "Element",
            "namespace": "",
        },
    )
    in_out_band_power_ratio_list: InOutBandPowerRatioListType = field(
        metadata={
            "name": "inOutBandPowerRatioList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    rfi_tmreport_list: None | RfiTmreportListType = field(
        default=None,
        metadata={
            "name": "rfiTMReportList",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_isolated_fmreport_list: None | RfiIsolatedFmreportListType = field(
        default=None,
        metadata={
            "name": "rfiIsolatedFMReportList",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_persistent_fmreport_list: None | RfiPersistentFmreportListType = field(
        default=None,
        metadata={
            "name": "rfiPersistentFMReportList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    dc_estimate_list: DcEstimateListType = field(
        metadata={
            "name": "dcEstimateList",
            "type": "Element",
            "namespace": "",
        },
    )
    fm_rate_estimate_list: FmRateEstimatesListType = field(
        metadata={
            "name": "fmRateEstimateList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    first_line_sensing_time_list: FirstLineSensingTimeListType = field(
        metadata={
            "name": "firstLineSensingTimeList",
            "type": "Element",
            "namespace": "",
        },
    )
    last_line_sensing_time_list: LastLineSensingTimeListType = field(
        metadata={
            "name": "lastLineSensingTimeList",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_input_samples: int = field(
        metadata={
            "name": "numberOfInputSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_input_lines: int = field(
        metadata={
            "name": "numberOfInputLines",
            "type": "Element",
            "namespace": "",
        },
    )
    swp_list: SwpListType = field(
        metadata={
            "name": "swpList",
            "type": "Element",
            "namespace": "",
        },
    )
    swl_list: SwlListType = field(
        metadata={
            "name": "swlList",
            "type": "Element",
            "namespace": "",
        },
    )
    prf_list: PrfListType = field(
        metadata={
            "name": "prfList",
            "type": "Element",
            "namespace": "",
        },
    )
    rank: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    tx_pulse_list: TxPulseListType = field(
        metadata={
            "name": "txPulseList",
            "type": "Element",
            "namespace": "",
        },
    )
    instrument_configuration_id: int = field(
        metadata={
            "name": "instrumentConfigurationID",
            "type": "Element",
            "namespace": "",
        },
    )
    radar_carrier_frequency: DoubleWithUnit = field(
        metadata={
            "name": "radarCarrierFrequency",
            "type": "Element",
            "namespace": "",
        },
    )
    rx_gain_list: RxGainListType = field(
        metadata={
            "name": "rxGainList",
            "type": "Element",
            "namespace": "",
        },
    )
    preamble_flag: str = field(
        metadata={"name": "preambleFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    postamble_flag: str = field(
        metadata={"name": "postambleFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    interleaved_calibration_flag: str = field(
        metadata={
            "name": "interleavedCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    data_format: DataFormatType = field(
        metadata={
            "name": "dataFormat",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    overall_product_quality_index: int = field(
        metadata={
            "name": "overallProductQualityIndex",
            "type": "Element",
            "namespace": "",
        },
    )
    quality_parameters_list: QualityParametersListType = field(
        metadata={
            "name": "qualityParametersList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    first_sample_slant_range_time: DoubleWithUnit = field(
        metadata={
            "name": "firstSampleSlantRangeTime",
            "type": "Element",
            "namespace": "",
        },
    )
    last_sample_slant_range_time: DoubleWithUnit = field(
        metadata={
            "name": "lastSampleSlantRangeTime",
            "type": "Element",
            "namespace": "",
        },
    )
    first_line_azimuth_time: str = field(
        metadata={
            "name": "firstLineAzimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    last_line_azimuth_time: str = field(
        metadata={
            "name": "lastLineAzimuthTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    range_time_interval: DoubleWithUnit = field(
        metadata={
            "name": "rangeTimeInterval",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_time_interval: DoubleWithUnit = field(
        metadata={
            "name": "azimuthTimeInterval",
            "type": "Element",
            "namespace": "",
        },
    )
    range_pixel_spacing: FloatWithUnit = field(
        metadata={
            "name": "rangePixelSpacing",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_pixel_spacing: FloatWithUnit = field(
        metadata={
            "name": "azimuthPixelSpacing",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_samples: int = field(
        metadata={
            "name": "numberOfSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_lines: int = field(
        metadata={
            "name": "numberOfLines",
            "type": "Element",
            "namespace": "",
        },
    )
    projection: ProjectionType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    range_coordinate_conversion: CoordinateConversionListType = field(
        metadata={
            "name": "rangeCoordinateConversion",
            "type": "Element",
            "namespace": "",
        },
    )
    datum: DatumType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    footprint: FloatArrayWithUnits = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    pixel_representation: PixelRepresentationType = field(
        metadata={
            "name": "pixelRepresentation",
            "type": "Element",
            "namespace": "",
        },
    )
    pixel_type: PixelTypeType = field(
        metadata={
            "name": "pixelType",
            "type": "Element",
            "namespace": "",
        },
    )
    pixel_quantity: PixelQuantityType = field(
        metadata={
            "name": "pixelQuantity",
            "type": "Element",
            "namespace": "",
        },
    )
    no_data_value: float = field(
        metadata={
            "name": "noDataValue",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    internal_calibration_parameters_used: InternalCalibrationSourceType = field(
        metadata={
            "name": "internalCalibrationParametersUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    range_reference_function_used: RangeReferenceFunctionType = field(
        metadata={
            "name": "rangeReferenceFunctionUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    noise_parameters_used: InternalCalibrationSourceType = field(
        metadata={
            "name": "noiseParametersUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_calibration_parameters_list: InternalCalibrationParametersListType = field(
        metadata={
            "name": "internalCalibrationParametersList",
            "type": "Element",
            "namespace": "",
        },
    )
    noise_list: NoiseListType = field(
        metadata={
            "name": "noiseList",
            "type": "Element",
            "namespace": "",
        },
    )
