# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD INS models
--------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bps.common.io.common_types import (
    AcquisitionModeIdtype,
    AzimuthPolynomialType,
    ChannelImbalanceList,
    ChannelType,
    Complex,
    ComplexArray,
    CrossTalkList,
    DataFormatModeType,
    DatumType,
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
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    PolarisationType,
    SignalType,
    SlantRangePolynomialType,
    StateType,
    SwathType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
)


@dataclass
class AttenuatorSettingType:
    """
    Parameters
    ----------
    digital
        List of settings for the 3 attenuators (A1, A2, A3)
    """

    class Meta:
        name = "Attenuator_Setting_Type"

    digital: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "Digital",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class IdType:
    """
    Parameters
    ----------
    param_id
        ParamID list
    weight
        Weights list
    count
    """

    class Meta:
        name = "ID_Type"

    param_id: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "ParamID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    weight: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "Weight",
            "type": "Element",
            "namespace": "",
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
class PolyType(FloatArray):
    class Meta:
        name = "Poly_Type"

    value_attribute: Optional[str] = field(
        default=None,
        metadata={
            "name": "value",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class SttSettingType:
    """
    Parameters
    ----------
    cdn_att1
        List of settings for attenuator A1 (one for each signal type)
    cdn_att2
        List of settings for attenuator A2 (one for each signal type)
    cdn_att3
        List of settings for attenuator A3 (one for each signal type)
    id
    """

    class Meta:
        name = "STT_Setting_Type"

    cdn_att1: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "CDN_Att1",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    cdn_att2: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "CDN_Att2",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    cdn_att3: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "CDN_Att3",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ID",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class IspType:
    """
    Parameters
    ----------
    signal_type
        Signal type.
    pri_number
        Number of repetition of the current entry.
    repetition_number
        Number of repetition of the block framed by the current entry and the entry identified together with the
        Return Address Offset (RAO) parameter. Set to -1 in case the number is not fixed (e.g., for initial and
        final slots).
    return_address_offset
        Offset identifying the entry where a repeated block starts.
    """

    class Meta:
        name = "ispType"

    signal_type: Optional[SignalType] = field(
        default=None,
        metadata={
            "name": "signalType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pri_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "priNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    repetition_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "repetitionNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": -1,
        },
    )
    return_address_offset: Optional[int] = field(
        default=None,
        metadata={
            "name": "returnAddressOffset",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class UnBaqluttype:
    """
    Parameters
    ----------
    baq_code
        BAQ code (BAQ 4 Bit, BAQ 5 Bit or BAQ 6 Bit).
    magnitude_code
        Magnitude codes array.
    normalized_iq
        Normalized I and Q value for each magnitude code in magnitudeCode array.
    """

    class Meta:
        name = "unBAQLUTType"

    baq_code: Optional[DataFormatModeType] = field(
        default=None,
        metadata={
            "name": "baqCode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    magnitude_code: Optional[IntArray] = field(
        default=None,
        metadata={
            "name": "magnitudeCode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    normalized_iq: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "normalizedIQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class UnBaqsmallAsmluttype:
    """
    Parameters
    ----------
    baq_code
        BAQ code (BAQ 4 Bit, BAQ 5 Bit or BAQ 6 Bit).
    asm_value
        Average Signal Magnitude (ASM) values array.
    uncompressed_iq
        Uncompressed I and Q value for each ASM value in asmValue array.
    """

    class Meta:
        name = "unBAQSmallASMLUTType"

    baq_code: Optional[DataFormatModeType] = field(
        default=None,
        metadata={
            "name": "baqCode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    asm_value: Optional[IntArray] = field(
        default=None,
        metadata={
            "name": "asmValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    uncompressed_iq: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "uncompressedIQ",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class FreqVariation2Type:
    """
    Parameters
    ----------
    ref_temperature
        Reference temperature [C]
    date
        Reference date [UTC]
    reference_1
        Reference document #1
    reference_2
        Reference document #2
    id1
        ID1 weigths
    id2
        ID2 weigths
    id3
        ID3 weigths
    id4
        ID4 weigths
    id5
        ID5 weigths
    id6
        ID6 weigths
    id7
        ID7 weigths
    id8
        ID8 weigths
    id9
        ID9 weigths
    id10
        ID10 weigths
    id11
        ID11 weigths
    id12
        ID12 weigths
    id13
        ID13 weigths
    id14
        ID14 weigths
    """

    class Meta:
        name = "Freq_Variation_2_Type"

    ref_temperature: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "Ref_Temperature",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    date: Optional[str] = field(
        default=None,
        metadata={
            "name": "Date",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_1: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_1",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_2: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_2",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    id1: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID1",
            "type": "Element",
            "namespace": "",
        },
    )
    id2: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID2",
            "type": "Element",
            "namespace": "",
        },
    )
    id3: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID3",
            "type": "Element",
            "namespace": "",
        },
    )
    id4: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID4",
            "type": "Element",
            "namespace": "",
        },
    )
    id5: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID5",
            "type": "Element",
            "namespace": "",
        },
    )
    id6: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID6",
            "type": "Element",
            "namespace": "",
        },
    )
    id7: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID7",
            "type": "Element",
            "namespace": "",
        },
    )
    id8: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID8",
            "type": "Element",
            "namespace": "",
        },
    )
    id9: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID9",
            "type": "Element",
            "namespace": "",
        },
    )
    id10: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID10",
            "type": "Element",
            "namespace": "",
        },
    )
    id11: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID11",
            "type": "Element",
            "namespace": "",
        },
    )
    id12: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID12",
            "type": "Element",
            "namespace": "",
        },
    )
    id13: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID13",
            "type": "Element",
            "namespace": "",
        },
    )
    id14: Optional[IdType] = field(
        default=None,
        metadata={
            "name": "ID14",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class PolyListType:
    """
    Parameters
    ----------
    polarisation
        Polynomial coefficients for current polarization
    count
    """

    class Meta:
        name = "Poly_List_Type"

    polarisation: list[PolyType] = field(
        default_factory=list,
        metadata={"name": "Polarisation", "type": "Element", "namespace": "", "min_occurs": 2, "max_occurs": 2},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class AttenuatorsSettingsType:
    """
    Parameters
    ----------
    stt_setting
        List of attenuators settings (one for each ID=XY, with X=[A..I] and Y=[A..I])
    count
    """

    class Meta:
        name = "attenuatorsSettingsType"

    stt_setting: list[SttSettingType] = field(
        default_factory=list,
        metadata={"name": "STT_Setting", "type": "Element", "namespace": "", "min_occurs": 18, "max_occurs": 18},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class IspListType:
    """
    Parameters
    ----------
    isp
        Element describing one unique or a series of unique transmission packets. The packets are identified by the
        packet signal type, the number of PRIs, the number of repetitions and the Return Address Offset (RAO).
    count
    """

    class Meta:
        name = "ispListType"

    isp: list[IspType] = field(
        default_factory=list, metadata={"type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 100}
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ModelDriftType:
    """
    Parameters
    ----------
    model_interval
        Interval between adjacent model values in the list [s].
    model_values
        Array of modelled complex values. The array contains "count" complex floating point values separated by
        spaces. The first value in the array corresponds to the time at the ascending node of the current orbit.
        Model values for times that fall between the points in the model are obtained by linear interpolation
        between the two nearest points.
    """

    class Meta:
        name = "modelDriftType"

    model_interval: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "modelInterval",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    model_values: Optional[ComplexArray] = field(
        default=None,
        metadata={
            "name": "modelValues",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class TemperatureLuttype:
    """
    Parameters
    ----------
    temperature_code
        Temperature codes array.
    temperature_value
        Temperature value [C] for each temperature code in temperatureCode array.
    """

    class Meta:
        name = "temperatureLUTType"

    temperature_code: Optional[IntArray] = field(
        default=None,
        metadata={
            "name": "temperatureCode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    temperature_value: Optional[FloatArrayWithUnits] = field(
        default=None,
        metadata={
            "name": "temperatureValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class UnBaqlutlistType:
    """
    Parameters
    ----------
    un_baqlut
        BAQ uncompressing look-up table for given BAQ code.
    count
    """

    class Meta:
        name = "unBAQLUTListType"

    un_baqlut: list[UnBaqluttype] = field(
        default_factory=list,
        metadata={"name": "unBAQLUT", "type": "Element", "namespace": "", "min_occurs": 3, "max_occurs": 3},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class UnBaqsmallAsmlutlistType:
    """
    Parameters
    ----------
    un_baqsmall_asmlut
        BAQ uncompressing look-up table for small ASM values and maximum magnitude for given BAQ code.
    count
    """

    class Meta:
        name = "unBAQSmallASMLUTListType"

    un_baqsmall_asmlut: list[UnBaqsmallAsmluttype] = field(
        default_factory=list,
        metadata={"name": "unBAQSmallASMLUT", "type": "Element", "namespace": "", "min_occurs": 3, "max_occurs": 3},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FreqVariationType:
    """
    Parameters
    ----------
    ref_temperature
        Reference temperature [C]
    date
        Reference date [UTC]
    reference_1
        Reference document #1
    reference_2
        Reference document #2
    amplitude
        Amplitude polynomials
    phase
        Phase polynomials
    """

    class Meta:
        name = "Freq_Variation_Type"

    ref_temperature: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "Ref_Temperature",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    date: Optional[str] = field(
        default=None,
        metadata={
            "name": "Date",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_1: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_1",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_2: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_2",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    amplitude: Optional[PolyListType] = field(
        default=None,
        metadata={
            "name": "Amplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    phase: Optional[PolyListType] = field(
        default=None,
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class TempFreqVariation2Type:
    """
    Parameters
    ----------
    freq_variation
        Weights describing variation with frequency
    """

    class Meta:
        name = "Temp_Freq_Variation_2_Type"

    freq_variation: Optional[FreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Freq_Variation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class TempVariationType:
    """
    Parameters
    ----------
    ref_frequency
        Reference frequency [MHz]
    date
        Reference date [UTC]
    reference_1
        Reference document #1
    reference_2
        Reference document #2
    amplitude
        Amplitude polynomials
    phase
        Phase polynomials
    """

    class Meta:
        name = "Temp_Variation_Type"

    ref_frequency: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "Ref_Frequency",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    date: Optional[str] = field(
        default=None,
        metadata={
            "name": "Date",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_1: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_1",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_2: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_2",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    amplitude: Optional[PolyListType] = field(
        default=None,
        metadata={
            "name": "Amplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    phase: Optional[PolyListType] = field(
        default=None,
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class IntCalParametersType:
    """
    Parameters
    ----------
    polarisation
        Polarisation to which this set of internal calibration parameters applies.
    model_drift
        Model drift. The model is relative to the ascending node of the current orbit. Already includes transmit and
        receive channel imbalances.
    reference_drift
        Reference value for the normalization of the drifts.
    internal_delay
        Internal time delay [s] to be applied to range compressed data. The L1 Processor uses this parameter only in
        case ideal chirp is used and internal delay cannot be derived from internal calibration pulses.
    tx_channel_imbalance
        Transmit channel imbalance to be applied to range compressed data. The L1 Processor uses this parameter only
        in case ideal chirp is used and channel imbalance cannot be derived from internal calibration pulses.
    rx_channel_imbalance
        Receive channel imbalance to be applied to range compressed data. The L1 Processor uses this parameter only
        in case ideal chirp is used and channel imbalance cannot be derived from internal calibration pulses.
    tx_power_tracking
        Transmit power tracking to be used for the computation of the combined secondary transmit patterns from the
        individual doublet patterns. The L1 Processor uses this parameter only in case it cannot be derived from
        internal calibration pulses.
    noise_power
        Nominal noise power value used in processing in case it cannot be derived from noise pulses [dB].
    """

    class Meta:
        name = "intCalParametersType"

    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    model_drift: Optional[ModelDriftType] = field(
        default=None,
        metadata={
            "name": "modelDrift",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_drift: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "referenceDrift",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_delay: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "internalDelay",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tx_channel_imbalance: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "txChannelImbalance",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rx_channel_imbalance: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "rxChannelImbalance",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tx_power_tracking: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "txPowerTracking",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    noise_power: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "noisePower",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class RawDataDecodingParametersType:
    """
    Parameters
    ----------
    un_baqlutlist
        BAQ uncompressing look-up tables. These LUT are taken from SAR Data ICD document.
    un_baqsmall_asmlutlist
        BAQ uncompressing look-up tables for small ASM values and maximum magnitude. These LUT are taken from SAR
        Data ICD document.
    temperature_lut
        Temperature calibration curve. This LUT is taken from SAR Data ICD document.
    """

    class Meta:
        name = "rawDataDecodingParametersType"

    un_baqlutlist: Optional[UnBaqlutlistType] = field(
        default=None,
        metadata={
            "name": "unBAQLUTList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    un_baqsmall_asmlutlist: Optional[UnBaqsmallAsmlutlistType] = field(
        default=None,
        metadata={
            "name": "unBAQSmallASMLUTList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    temperature_lut: Optional[TemperatureLuttype] = field(
        default=None,
        metadata={
            "name": "temperatureLUT",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class SectionType:
    """
    Parameters
    ----------
    name
        Section name.
    repeat_flag
        Section repeat flag. True for central slot, False otherwise.
    isp_list
        List of expected ISP within this section in the order they should be received.
    """

    class Meta:
        name = "sectionType"

    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    repeat_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "repeatFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    isp_list: Optional[IspListType] = field(
        default=None,
        metadata={
            "name": "ispList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class TempFreqVariationType:
    """
    Parameters
    ----------
    temp_variation
        Polynomials describing variation with temperature
    freq_variation
        Polynomials describing variation with frequency
    """

    class Meta:
        name = "Temp_Freq_Variation_Type"

    temp_variation: Optional[TempVariationType] = field(
        default=None,
        metadata={
            "name": "Temp_Variation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    freq_variation: Optional[FreqVariationType] = field(
        default=None,
        metadata={
            "name": "Freq_Variation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class IntCalParametersListType:
    """
    Parameters
    ----------
    int_cal_parameters
        Instrument parameters for all the polarizations of a given swath.
    count
    """

    class Meta:
        name = "intCalParametersListType"

    int_cal_parameters: list[IntCalParametersType] = field(
        default_factory=list,
        metadata={"name": "intCalParameters", "type": "Element", "namespace": "", "min_occurs": 4, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class TimelineParametersType:
    """
    Parameters
    ----------
    section
        Timeline section. The maximum number of timeline sections is set to 5: preamble, initial slot, central slot
        (including internal calibration sequence followed by science data sequence), final slot, postamble. Initial
        and final slots are typically partial, while central slot is typically repeated.
    count
    """

    class Meta:
        name = "timelineParametersType"

    section: list[SectionType] = field(
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
class CdnRxcalType:
    """
    Parameters
    ----------
    lcdn_rxcal_nom
        LCDN_RXCal_Nom polynomials
    lcdn_rxcal_red
        LCDN_RXCal_Red polynomials
    thermistor_cdn_a
        Thermistor DCU A weights
    thermistor_cdn_b
        Thermistor DCU B weights
    attenuator_setting
        Attenuators setting
    a1
    a2
    a3
    """

    class Meta:
        name = "CDN_RXCal_Type"

    lcdn_rxcal_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LCDN_RXCal_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lcdn_rxcal_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LCDN_RXCal_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_cdn_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_CDN_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_cdn_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_CDN_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    attenuator_setting: Optional[AttenuatorSettingType] = field(
        default=None,
        metadata={
            "name": "Attenuator_Setting",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    a1: Optional[int] = field(
        default=None,
        metadata={
            "name": "A1",
            "type": "Attribute",
            "required": True,
        },
    )
    a2: Optional[int] = field(
        default=None,
        metadata={
            "name": "A2",
            "type": "Attribute",
            "required": True,
        },
    )
    a3: Optional[int] = field(
        default=None,
        metadata={
            "name": "A3",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class CdnShCalType:
    """
    Parameters
    ----------
    lcdn_sh_cal_nom
        LCDN_ShCal_Nom polynomials
    lcdn_sh_cal_red
        LCDN_ShCal_Red polynomials
    thermistor_cdn_a
        Thermistor DCU A weights
    thermistor_cdn_b
        Thermistor DCU B weights
    attenuator_setting
        Attenuators setting
    a1
    a2
    a3
    """

    class Meta:
        name = "CDN_ShCal_Type"

    lcdn_sh_cal_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LCDN_ShCal_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lcdn_sh_cal_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LCDN_ShCal_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_cdn_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_CDN_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_cdn_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_CDN_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    attenuator_setting: Optional[AttenuatorSettingType] = field(
        default=None,
        metadata={
            "name": "Attenuator_Setting",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    a1: Optional[int] = field(
        default=None,
        metadata={
            "name": "A1",
            "type": "Attribute",
            "required": True,
        },
    )
    a2: Optional[int] = field(
        default=None,
        metadata={
            "name": "A2",
            "type": "Attribute",
            "required": True,
        },
    )
    a3: Optional[int] = field(
        default=None,
        metadata={
            "name": "A3",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class CdnTxcalType:
    """
    Parameters
    ----------
    lcdn_txcal_nom
        LCDN_TXCal_Nom polynomials
    lcdn_txcal_red
        LCDN_TXCal_Red polynomials
    thermistor_cdn_a
        Thermistor DCU A weights
    thermistor_cdn_b
        Thermistor DCU B weights
    attenuator_setting
        Attenuators setting
    a1
    a2
    a3
    """

    class Meta:
        name = "CDN_TXCal_Type"

    lcdn_txcal_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LCDN_TXCal_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lcdn_txcal_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LCDN_TXCal_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_cdn_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_CDN_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_cdn_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_CDN_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    attenuator_setting: Optional[AttenuatorSettingType] = field(
        default=None,
        metadata={
            "name": "Attenuator_Setting",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    a1: Optional[int] = field(
        default=None,
        metadata={
            "name": "A1",
            "type": "Attribute",
            "required": True,
        },
    )
    a2: Optional[int] = field(
        default=None,
        metadata={
            "name": "A2",
            "type": "Attribute",
            "required": True,
        },
    )
    a3: Optional[int] = field(
        default=None,
        metadata={
            "name": "A3",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class PasCdnCalibrationCableType:
    """
    Parameters
    ----------
    lpas_cdn_nom
        LPAS_CDN_Nom polynomials
    lpas_cdn_red
        LPAS_CDN_Red polynomials
    thermistor_dcu_a
        Thermistor DCU A weights
    thermistor_dcu_b
        Thermistor DCU B weights
    """

    class Meta:
        name = "PAS-CDN_Calibration_Cable_Type"

    lpas_cdn_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_CDN_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_cdn_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_CDN_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PasFaCablesType:
    """
    Parameters
    ----------
    lpas_fa1
        LPAS_FA1 polynomials
    lpas_fa2
        LPAS_FA2 polynomials
    thermistor_dcu_a
        Thermistor DCU A weights
    thermistor_dcu_b
        Thermistor DCU B weights
    """

    class Meta:
        name = "PAS-FA_Cables_Type"

    lpas_fa1: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_FA1",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_fa2: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_FA2",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PasRxCalPathType:
    """
    Parameters
    ----------
    lpas_rxcal_path_nom
        LPAS_RXCalPath_Nom polynomials
    lpas_rxcal_path_red
        LPAS_RXCalPath_Red polynomials
    thermistor_dcu_a
        Thermistor DCU A weights
    thermistor_dcu_b
        Thermistor DCU B weights
    """

    class Meta:
        name = "PAS_RX_Cal_Path_Type"

    lpas_rxcal_path_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_RXCalPath_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_rxcal_path_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_RXCalPath_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PasRxCalType:
    """
    Parameters
    ----------
    lpas_rxcal_nom
        LPAS_RXCal_Nom polynomials
    lpas_rxcal_red
        LPAS_RXCal_Red polynomials
    thermistor_dcu_a
        Thermistor DCU A weights
    thermistor_dcu_b
        Thermistor DCU B weights
    """

    class Meta:
        name = "PAS_RX_Cal_Type"

    lpas_rxcal_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_RXCal_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_rxcal_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_RXCal_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PasRxPathType:
    """
    Parameters
    ----------
    lpas_rx1
        LPAS_RX1 polynomials
    lpas_rx2
        LPAS_RX2 polynomials
    thermistor_dcu_a
        Thermistor DCU A weights
    thermistor_dcu_b
        Thermistor DCU B weights
    """

    class Meta:
        name = "PAS_RX_Path_Type"

    lpas_rx1: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_RX1",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_rx2: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_RX2",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PasTxCalType:
    """
    Parameters
    ----------
    lpas_txcal1_nom
        LPAS_TXCal1_Nom polynomials
    lpas_txcal2_nom
        LPAS_TXCal2_Nom polynomials
    lpas_txcal_nom
        LPAS_TXCal_Nom polynomials
    lpas_txcal1_red
        LPAS_TXCal1_Red polynomials
    lpas_txcal2_red
        LPAS_TXCal2_Red polynomials
    lpas_txcal_red
        LPAS_TXCal_Red polynomials
    thermistor_dcu_a
        Thermistor DCU A weights
    thermistor_dcu_b
        Thermistor DCU B weights
    """

    class Meta:
        name = "PAS_TX_Cal_Type"

    lpas_txcal1_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_TXCal1_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_txcal2_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_TXCal2_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_txcal_nom: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_TXCal_Nom",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_txcal1_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_TXCal1_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_txcal2_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_TXCal2_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lpas_txcal_red: Optional[TempFreqVariationType] = field(
        default=None,
        metadata={
            "name": "LPAS_TXCal_Red",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_a: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    thermistor_dcu_b: Optional[TempFreqVariation2Type] = field(
        default=None,
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AcquisitionModeType:
    """
    Parameters
    ----------
    acquisition_mode
        Acquisition mode (i.e., S1 INT, S1 TOM, S2 INT, ...).
    gstl_index
        Generic SAR Mode Timeline (GSTL) index.
    swath
        Swath (S1, S2, S3).
    int_cal_parameters_list
        Swath- and polarization-dependent instrument parameters.
    timeline_parameters_odd_rank
        Expected packet transmission sequence in case of odd rank.
    timeline_parameters_even_rank
        Expected packet transmission sequence in case of even rank.
    """

    class Meta:
        name = "acquisitionModeType"

    acquisition_mode: Optional[AcquisitionModeIdtype] = field(
        default=None,
        metadata={
            "name": "acquisitionMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    gstl_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "gstlIndex",
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
    int_cal_parameters_list: Optional[IntCalParametersListType] = field(
        default=None,
        metadata={
            "name": "intCalParametersList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    timeline_parameters_odd_rank: Optional[TimelineParametersType] = field(
        default=None,
        metadata={
            "name": "timelineParametersOddRank",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    timeline_parameters_even_rank: Optional[TimelineParametersType] = field(
        default=None,
        metadata={
            "name": "timelineParametersEvenRank",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AcquisitionModeListType:
    """
    Parameters
    ----------
    acquisition_mode
        Instrument parameters for a given acquisition mode. The maximum number of entries in the list is set to 48,
        i.e. the currently foreseen number of GSTL indexes.
    count
    """

    class Meta:
        name = "acquisitionModeListType"

    acquisition_mode: list[AcquisitionModeType] = field(
        default_factory=list,
        metadata={"name": "acquisitionMode", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 48},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class CharacterisationDataType:
    """
    Parameters
    ----------
    pas_tx_cal
        PAS characterization data for PAS TX Cal (PAS-4452)
    pas_rx_cal
        Derived PAS RX Cal polynomial based on PAS characterization data
    pas_rx_path
        PAS characterization data for PAS RX Path (PAS-4502)
    pas_rx_cal_path
        PAS characterization data for PAS RX Cal Path (PAS-4454)
    cdn_txcal
        CDN characterization data TXCal (CDN-390) for each combination of attenuators settings
    cdn_rxcal
        CDN characterization data RXCal (CDN-390) for each combination of attenuators settings
    cdn_sh_cal
        CDN characterization data ShCal (CDN-390) for each combination of attenuators settings
    pas_cdn_calibration_cable
        PAS-CDN Calibration cable characterization data (HAR-5524)
    pas_fa_cables
        PAS-FA cable characterization data (HAR-5525)
    """

    class Meta:
        name = "characterisationDataType"

    pas_tx_cal: Optional[PasTxCalType] = field(
        default=None,
        metadata={
            "name": "PAS_TX_Cal",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pas_rx_cal: Optional[PasRxCalType] = field(
        default=None,
        metadata={
            "name": "PAS_RX_Cal",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pas_rx_path: Optional[PasRxPathType] = field(
        default=None,
        metadata={
            "name": "PAS_RX_Path",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pas_rx_cal_path: Optional[PasRxCalPathType] = field(
        default=None,
        metadata={
            "name": "PAS_RX_Cal_Path",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    cdn_txcal: list[CdnTxcalType] = field(
        default_factory=list,
        metadata={
            "name": "CDN_TXCal",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    cdn_rxcal: list[CdnRxcalType] = field(
        default_factory=list,
        metadata={
            "name": "CDN_RXCal",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    cdn_sh_cal: list[CdnShCalType] = field(
        default_factory=list,
        metadata={
            "name": "CDN_ShCal",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    pas_cdn_calibration_cable: Optional[PasCdnCalibrationCableType] = field(
        default=None,
        metadata={
            "name": "PAS-CDN_Calibration_Cable",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pas_fa_cables: Optional[PasFaCablesType] = field(
        default=None,
        metadata={
            "name": "PAS-FA_Cables",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryInstrumentParametersType:
    """
    Parameters
    ----------
    radar_frequency
        Radar frequency [Hz].
    roll_bias
        Bias to be added to roll estimated from attitude to offset it (by default set to 0) [deg].
    tx_start_time
        TX start time (T9) [s].
    calibration_signals_swp
        Calibration signals Sampling Window Position (SWP) [s].
    acquisition_mode_list
        List of instrument parameters for each foreseen acquisition mode (i.e., S1 INT, S1 TOM, S2 INT, ...). Each
        acquisition mode is one-to-one associated with a GSTL index.
    raw_data_decoding_parameters
        Raw data decoding parameters.
    characterisation_data
        Characterization data.
    attenuators_settings
        Attenuators settings.
    """

    class Meta:
        name = "auxiliaryInstrumentParametersType"

    radar_frequency: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "radarFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
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
    tx_start_time: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "txStartTime",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    calibration_signals_swp: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "calibrationSignalsSWP",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    acquisition_mode_list: Optional[AcquisitionModeListType] = field(
        default=None,
        metadata={
            "name": "acquisitionModeList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_data_decoding_parameters: Optional[RawDataDecodingParametersType] = field(
        default=None,
        metadata={
            "name": "rawDataDecodingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    characterisation_data: Optional[CharacterisationDataType] = field(
        default=None,
        metadata={
            "name": "characterisationData",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    attenuators_settings: Optional[AttenuatorsSettingsType] = field(
        default=None,
        metadata={
            "name": "attenuatorsSettings",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryInstrumentParameters(AuxiliaryInstrumentParametersType):
    """
    BIOMASS auxiliary instrument parameters element.
    """

    class Meta:
        name = "auxiliaryInstrumentParameters"
