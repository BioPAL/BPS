# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD INS models
--------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


@dataclass(kw_only=True)
class AttenuatorSettingType:
    """
    Parameters
    ----------
    digital
        List of settings for the 3 attenuators (A1, A2, A3)
    """

    class Meta:
        name = "Attenuator_Setting_Type"

    digital: FloatArray = field(
        metadata={
            "name": "Digital",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    param_id: FloatArray = field(
        metadata={
            "name": "ParamID",
            "type": "Element",
            "namespace": "",
        },
    )
    weight: FloatArray = field(
        metadata={
            "name": "Weight",
            "type": "Element",
            "namespace": "",
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class PolyType(FloatArray):
    class Meta:
        name = "Poly_Type"

    value_attribute: str = field(
        metadata={
            "name": "value",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    cdn_att1: FloatArray = field(
        metadata={
            "name": "CDN_Att1",
            "type": "Element",
            "namespace": "",
        },
    )
    cdn_att2: FloatArray = field(
        metadata={
            "name": "CDN_Att2",
            "type": "Element",
            "namespace": "",
        },
    )
    cdn_att3: FloatArray = field(
        metadata={
            "name": "CDN_Att3",
            "type": "Element",
            "namespace": "",
        },
    )
    id: str = field(
        metadata={
            "name": "ID",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    signal_type: SignalType = field(
        metadata={
            "name": "signalType",
            "type": "Element",
            "namespace": "",
        },
    )
    pri_number: int = field(
        metadata={
            "name": "priNumber",
            "type": "Element",
            "namespace": "",
        },
    )
    repetition_number: int = field(
        metadata={"name": "repetitionNumber", "type": "Element", "namespace": "", "min_inclusive": -1}
    )
    return_address_offset: int = field(
        metadata={
            "name": "returnAddressOffset",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    baq_code: DataFormatModeType = field(
        metadata={
            "name": "baqCode",
            "type": "Element",
            "namespace": "",
        },
    )
    magnitude_code: IntArray = field(
        metadata={
            "name": "magnitudeCode",
            "type": "Element",
            "namespace": "",
        },
    )
    normalized_iq: FloatArray = field(
        metadata={
            "name": "normalizedIQ",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    baq_code: DataFormatModeType = field(
        metadata={
            "name": "baqCode",
            "type": "Element",
            "namespace": "",
        },
    )
    asm_value: IntArray = field(
        metadata={
            "name": "asmValue",
            "type": "Element",
            "namespace": "",
        },
    )
    uncompressed_iq: FloatArray = field(
        metadata={
            "name": "uncompressedIQ",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    ref_temperature: FloatWithUnit = field(
        metadata={
            "name": "Ref_Temperature",
            "type": "Element",
            "namespace": "",
        },
    )
    date: str = field(
        metadata={
            "name": "Date",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_1: str = field(
        metadata={
            "name": "Reference_1",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_2: str = field(
        metadata={
            "name": "Reference_2",
            "type": "Element",
            "namespace": "",
        },
    )
    id1: None | IdType = field(
        default=None,
        metadata={
            "name": "ID1",
            "type": "Element",
            "namespace": "",
        },
    )
    id2: None | IdType = field(
        default=None,
        metadata={
            "name": "ID2",
            "type": "Element",
            "namespace": "",
        },
    )
    id3: None | IdType = field(
        default=None,
        metadata={
            "name": "ID3",
            "type": "Element",
            "namespace": "",
        },
    )
    id4: None | IdType = field(
        default=None,
        metadata={
            "name": "ID4",
            "type": "Element",
            "namespace": "",
        },
    )
    id5: None | IdType = field(
        default=None,
        metadata={
            "name": "ID5",
            "type": "Element",
            "namespace": "",
        },
    )
    id6: None | IdType = field(
        default=None,
        metadata={
            "name": "ID6",
            "type": "Element",
            "namespace": "",
        },
    )
    id7: None | IdType = field(
        default=None,
        metadata={
            "name": "ID7",
            "type": "Element",
            "namespace": "",
        },
    )
    id8: None | IdType = field(
        default=None,
        metadata={
            "name": "ID8",
            "type": "Element",
            "namespace": "",
        },
    )
    id9: None | IdType = field(
        default=None,
        metadata={
            "name": "ID9",
            "type": "Element",
            "namespace": "",
        },
    )
    id10: None | IdType = field(
        default=None,
        metadata={
            "name": "ID10",
            "type": "Element",
            "namespace": "",
        },
    )
    id11: None | IdType = field(
        default=None,
        metadata={
            "name": "ID11",
            "type": "Element",
            "namespace": "",
        },
    )
    id12: None | IdType = field(
        default=None,
        metadata={
            "name": "ID12",
            "type": "Element",
            "namespace": "",
        },
    )
    id13: None | IdType = field(
        default=None,
        metadata={
            "name": "ID13",
            "type": "Element",
            "namespace": "",
        },
    )
    id14: None | IdType = field(
        default=None,
        metadata={
            "name": "ID14",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    model_interval: FloatWithUnit = field(
        metadata={
            "name": "modelInterval",
            "type": "Element",
            "namespace": "",
        },
    )
    model_values: ComplexArray = field(
        metadata={
            "name": "modelValues",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    temperature_code: IntArray = field(
        metadata={
            "name": "temperatureCode",
            "type": "Element",
            "namespace": "",
        },
    )
    temperature_value: FloatArrayWithUnits = field(
        metadata={
            "name": "temperatureValue",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    ref_temperature: FloatWithUnit = field(
        metadata={
            "name": "Ref_Temperature",
            "type": "Element",
            "namespace": "",
        },
    )
    date: str = field(
        metadata={
            "name": "Date",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_1: str = field(
        metadata={
            "name": "Reference_1",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_2: str = field(
        metadata={
            "name": "Reference_2",
            "type": "Element",
            "namespace": "",
        },
    )
    amplitude: PolyListType = field(
        metadata={
            "name": "Amplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    phase: PolyListType = field(
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class TempFreqVariation2Type:
    """
    Parameters
    ----------
    freq_variation
        Weights describing variation with frequency
    """

    class Meta:
        name = "Temp_Freq_Variation_2_Type"

    freq_variation: FreqVariation2Type = field(
        metadata={
            "name": "Freq_Variation",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    ref_frequency: FloatWithUnit = field(
        metadata={
            "name": "Ref_Frequency",
            "type": "Element",
            "namespace": "",
        },
    )
    date: str = field(
        metadata={
            "name": "Date",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_1: str = field(
        metadata={
            "name": "Reference_1",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_2: str = field(
        metadata={
            "name": "Reference_2",
            "type": "Element",
            "namespace": "",
        },
    )
    amplitude: PolyListType = field(
        metadata={
            "name": "Amplitude",
            "type": "Element",
            "namespace": "",
        },
    )
    phase: PolyListType = field(
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    polarisation: PolarisationType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    model_drift: ModelDriftType = field(
        metadata={
            "name": "modelDrift",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_drift: Complex = field(
        metadata={
            "name": "referenceDrift",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_delay: FloatWithUnit = field(
        metadata={
            "name": "internalDelay",
            "type": "Element",
            "namespace": "",
        },
    )
    tx_channel_imbalance: Complex = field(
        metadata={
            "name": "txChannelImbalance",
            "type": "Element",
            "namespace": "",
        },
    )
    rx_channel_imbalance: Complex = field(
        metadata={
            "name": "rxChannelImbalance",
            "type": "Element",
            "namespace": "",
        },
    )
    tx_power_tracking: Complex = field(
        metadata={
            "name": "txPowerTracking",
            "type": "Element",
            "namespace": "",
        },
    )
    noise_power: FloatWithUnit = field(
        metadata={
            "name": "noisePower",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    un_baqlutlist: UnBaqlutlistType = field(
        metadata={
            "name": "unBAQLUTList",
            "type": "Element",
            "namespace": "",
        },
    )
    un_baqsmall_asmlutlist: UnBaqsmallAsmlutlistType = field(
        metadata={
            "name": "unBAQSmallASMLUTList",
            "type": "Element",
            "namespace": "",
        },
    )
    temperature_lut: TemperatureLuttype = field(
        metadata={
            "name": "temperatureLUT",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    name: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    repeat_flag: str = field(
        metadata={"name": "repeatFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    isp_list: IspListType = field(
        metadata={
            "name": "ispList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    temp_variation: TempVariationType = field(
        metadata={
            "name": "Temp_Variation",
            "type": "Element",
            "namespace": "",
        },
    )
    freq_variation: FreqVariationType = field(
        metadata={
            "name": "Freq_Variation",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    lcdn_rxcal_nom: TempFreqVariationType = field(
        metadata={
            "name": "LCDN_RXCal_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lcdn_rxcal_red: TempFreqVariationType = field(
        metadata={
            "name": "LCDN_RXCal_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_cdn_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_CDN_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_cdn_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_CDN_B",
            "type": "Element",
            "namespace": "",
        },
    )
    attenuator_setting: AttenuatorSettingType = field(
        metadata={
            "name": "Attenuator_Setting",
            "type": "Element",
            "namespace": "",
        },
    )
    a1: int = field(
        metadata={
            "name": "A1",
            "type": "Attribute",
        },
    )
    a2: int = field(
        metadata={
            "name": "A2",
            "type": "Attribute",
        },
    )
    a3: int = field(
        metadata={
            "name": "A3",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    lcdn_sh_cal_nom: TempFreqVariationType = field(
        metadata={
            "name": "LCDN_ShCal_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lcdn_sh_cal_red: TempFreqVariationType = field(
        metadata={
            "name": "LCDN_ShCal_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_cdn_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_CDN_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_cdn_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_CDN_B",
            "type": "Element",
            "namespace": "",
        },
    )
    attenuator_setting: AttenuatorSettingType = field(
        metadata={
            "name": "Attenuator_Setting",
            "type": "Element",
            "namespace": "",
        },
    )
    a1: int = field(
        metadata={
            "name": "A1",
            "type": "Attribute",
        },
    )
    a2: int = field(
        metadata={
            "name": "A2",
            "type": "Attribute",
        },
    )
    a3: int = field(
        metadata={
            "name": "A3",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    lcdn_txcal_nom: TempFreqVariationType = field(
        metadata={
            "name": "LCDN_TXCal_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lcdn_txcal_red: TempFreqVariationType = field(
        metadata={
            "name": "LCDN_TXCal_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_cdn_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_CDN_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_cdn_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_CDN_B",
            "type": "Element",
            "namespace": "",
        },
    )
    attenuator_setting: AttenuatorSettingType = field(
        metadata={
            "name": "Attenuator_Setting",
            "type": "Element",
            "namespace": "",
        },
    )
    a1: int = field(
        metadata={
            "name": "A1",
            "type": "Attribute",
        },
    )
    a2: int = field(
        metadata={
            "name": "A2",
            "type": "Attribute",
        },
    )
    a3: int = field(
        metadata={
            "name": "A3",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    lpas_cdn_nom: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_CDN_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_cdn_red: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_CDN_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    lpas_fa1: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_FA1",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_fa2: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_FA2",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    lpas_rxcal_path_nom: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_RXCalPath_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_rxcal_path_red: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_RXCalPath_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    lpas_rxcal_nom: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_RXCal_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_rxcal_red: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_RXCal_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    lpas_rx1: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_RX1",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_rx2: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_RX2",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    lpas_txcal1_nom: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_TXCal1_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_txcal2_nom: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_TXCal2_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_txcal_nom: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_TXCal_Nom",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_txcal1_red: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_TXCal1_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_txcal2_red: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_TXCal2_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    lpas_txcal_red: TempFreqVariationType = field(
        metadata={
            "name": "LPAS_TXCal_Red",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_a: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_A",
            "type": "Element",
            "namespace": "",
        },
    )
    thermistor_dcu_b: TempFreqVariation2Type = field(
        metadata={
            "name": "Thermistor_DCU_B",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    acquisition_mode: AcquisitionModeIdtype = field(
        metadata={
            "name": "acquisitionMode",
            "type": "Element",
            "namespace": "",
        },
    )
    gstl_index: int = field(
        metadata={
            "name": "gstlIndex",
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
    int_cal_parameters_list: IntCalParametersListType = field(
        metadata={
            "name": "intCalParametersList",
            "type": "Element",
            "namespace": "",
        },
    )
    timeline_parameters_odd_rank: TimelineParametersType = field(
        metadata={
            "name": "timelineParametersOddRank",
            "type": "Element",
            "namespace": "",
        },
    )
    timeline_parameters_even_rank: TimelineParametersType = field(
        metadata={
            "name": "timelineParametersEvenRank",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    pas_tx_cal: PasTxCalType = field(
        metadata={
            "name": "PAS_TX_Cal",
            "type": "Element",
            "namespace": "",
        },
    )
    pas_rx_cal: PasRxCalType = field(
        metadata={
            "name": "PAS_RX_Cal",
            "type": "Element",
            "namespace": "",
        },
    )
    pas_rx_path: PasRxPathType = field(
        metadata={
            "name": "PAS_RX_Path",
            "type": "Element",
            "namespace": "",
        },
    )
    pas_rx_cal_path: PasRxCalPathType = field(
        metadata={
            "name": "PAS_RX_Cal_Path",
            "type": "Element",
            "namespace": "",
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
    pas_cdn_calibration_cable: PasCdnCalibrationCableType = field(
        metadata={
            "name": "PAS-CDN_Calibration_Cable",
            "type": "Element",
            "namespace": "",
        },
    )
    pas_fa_cables: PasFaCablesType = field(
        metadata={
            "name": "PAS-FA_Cables",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    radar_frequency: FloatWithUnit = field(
        metadata={
            "name": "radarFrequency",
            "type": "Element",
            "namespace": "",
        },
    )
    roll_bias: FloatWithUnit = field(
        metadata={
            "name": "rollBias",
            "type": "Element",
            "namespace": "",
        },
    )
    tx_start_time: FloatWithUnit = field(
        metadata={
            "name": "txStartTime",
            "type": "Element",
            "namespace": "",
        },
    )
    calibration_signals_swp: FloatWithUnit = field(
        metadata={
            "name": "calibrationSignalsSWP",
            "type": "Element",
            "namespace": "",
        },
    )
    acquisition_mode_list: AcquisitionModeListType = field(
        metadata={
            "name": "acquisitionModeList",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_data_decoding_parameters: RawDataDecodingParametersType = field(
        metadata={
            "name": "rawDataDecodingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    characterisation_data: CharacterisationDataType = field(
        metadata={
            "name": "characterisationData",
            "type": "Element",
            "namespace": "",
        },
    )
    attenuators_settings: AttenuatorsSettingsType = field(
        metadata={
            "name": "attenuatorsSettings",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryInstrumentParameters(AuxiliaryInstrumentParametersType):
    """
    BIOMASS auxiliary instrument parameters element.
    """

    class Meta:
        name = "auxiliaryInstrumentParameters"
