# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common translate
----------------
"""

from __future__ import annotations

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common
from bps.transcoder.io.v445 import common as v445
from bps.transcoder.io.v450 import common as v450
from xsdata.models.datatype import XmlDateTime

MinMaxType = v450.MinMaxType | v445.MinMaxType
MinMaxTypeWithUnit = v450.MinMaxTypeWithUnit | v445.MinMaxTypeWithUnit
ProcessingModeType = v450.ProcessingModeType | v445.ProcessingModeType
OrbitAttitudeSourceType = v450.OrbitAttitudeSourceType | v445.OrbitAttitudeSourceType
RfiMaskType = v450.RfiMaskType | v445.RfiMaskType
RfiMitigationMethodType = v450.RfiMitigationMethodType | v445.RfiMitigationMethodType
ChannelType = v450.ChannelType | v445.ChannelType
ChannelImbalanceList = v450.ChannelImbalanceList | v445.ChannelImbalanceList
CrossTalkList = v450.CrossTalkList | v445.CrossTalkList
CoregistrationMethodType = v450.CoregistrationMethodType | v445.CoregistrationMethodType
SkpPhaseCalibrationPostprocessingType = (
    v450.SkpPhaseCalibrationPostprocessingType | v445.SkpPhaseCalibrationPostprocessingType
)
PrimaryImageSelectionMethodType = v450.PrimaryImageSelectionMethodType | v445.PrimaryImageSelectionMethodType
PrimaryImageSelectionInformationType = (
    v450.PrimaryImageSelectionInformationType | v445.PrimaryImageSelectionInformationType
)
PolarisationCombinationMethodType = v450.PolarisationCombinationMethodType | v445.PolarisationCombinationMethodType
LayerListType = v450.LayerListType | v445.LayerListType
LayerType = v450.LayerType | v445.LayerType
HeightModelType = v450.HeightModelType | v445.HeightModelType
HeightModelBaseType = v450.HeightModelBaseType | v445.HeightModelBaseType
InternalCalibrationSourceType = v450.InternalCalibrationSourceType | v445.InternalCalibrationSourceType
SlantRangePolynomialType = v450.SlantRangePolynomialType | v445.SlantRangePolynomialType
TimeTypeWithPolarisation = v450.TimeTypeWithPolarisation | v445.TimeTypeWithPolarisation
DataFormatModeType = v450.DataFormatModeType | v445.DataFormatModeType
StateType = v450.StateType | v445.StateType
ProductCompositionType = v450.ProductCompositionType | v445.ProductCompositionType
OrbitPassType = v450.OrbitPassType | v445.OrbitPassType
SensorModeType = v450.SensorModeType | v445.SensorModeType
MissionPhaseIdtype = v450.MissionPhaseIdtype | v445.MissionPhaseIdtype
ProductType = v450.ProductType | v445.ProductType
SwathType = v450.SwathType | v445.SwathType
MissionType = v450.MissionType | v445.MissionType
PixelQuantityType = v450.PixelQuantityType | v445.PixelQuantityType
PixelTypeType = v450.PixelTypeType | v445.PixelTypeType
FloatWithUnit = v450.FloatWithUnit | v445.FloatWithUnit
UomType = v450.UomType | v445.UomType
FloatArrayWithUnits = v450.FloatArrayWithUnits | v445.FloatArrayWithUnits
FloatArray = v450.FloatArray | v445.FloatArray
DoubleArray = v450.DoubleArray | v445.DoubleArray
IntArray = v450.IntArray | v445.IntArray
DoubleArrayWithUnits = v450.DoubleArrayWithUnits | v445.DoubleArrayWithUnits
Complex = v450.Complex | v445.Complex
InterferometricPairType = v450.InterferometricPairType | v445.InterferometricPairType
InterferometricPairListType = v450.InterferometricPairListType | v445.InterferometricPairListType
ProjectionType = v450.ProjectionType | v445.ProjectionType
GeodeticReferenceFrameType = v450.GeodeticReferenceFrameType | v445.GeodeticReferenceFrameType
DatumType = v450.DatumType | v445.DatumType
PixelRepresentationType = v450.PixelRepresentationType | v445.PixelRepresentationType
ChannelType = v450.ChannelType | v445.ChannelType
ChannelImbalanceList = v450.ChannelImbalanceList | v445.ChannelImbalanceList
CrossTalkList = v450.CrossTalkList | v445.CrossTalkList
CoregistrationMethodType = v450.CoregistrationMethodType | v445.CoregistrationMethodType
SkpPhaseCalibrationPostprocessingType = (
    v450.SkpPhaseCalibrationPostprocessingType | v445.SkpPhaseCalibrationPostprocessingType
)
PrimaryImageSelectionMethodType = v450.PrimaryImageSelectionMethodType | v445.PrimaryImageSelectionMethodType
PrimaryImageSelectionInformationType = (
    v450.PrimaryImageSelectionInformationType | v445.PrimaryImageSelectionInformationType
)
PolarisationCombinationMethodType = v450.PolarisationCombinationMethodType | v445.PolarisationCombinationMethodType
LayerListType = v450.LayerListType | v445.LayerListType
LayerType = v450.LayerType | v445.LayerType
HeightModelType = v450.HeightModelType | v445.HeightModelType
HeightModelBaseType = v450.HeightModelBaseType | v445.HeightModelBaseType
InternalCalibrationSourceType = v450.InternalCalibrationSourceType | v445.InternalCalibrationSourceType
SlantRangePolynomialType = v450.SlantRangePolynomialType | v445.SlantRangePolynomialType
TimeTypeWithPolarisation = v450.TimeTypeWithPolarisation | v445.TimeTypeWithPolarisation
DataFormatModeType = v450.DataFormatModeType | v445.DataFormatModeType
StateType = v450.StateType | v445.StateType
ProductCompositionType = v450.ProductCompositionType | v445.ProductCompositionType
OrbitPassType = v450.OrbitPassType | v445.OrbitPassType
SensorModeType = v450.SensorModeType | v445.SensorModeType
MissionPhaseIdtype = v450.MissionPhaseIdtype | v445.MissionPhaseIdtype
ProductType = v450.ProductType | v445.ProductType
SwathType = v450.SwathType | v445.SwathType
MissionType = v450.MissionType | v445.MissionType
PixelQuantityType = v450.PixelQuantityType | v445.PixelQuantityType
PixelTypeType = v450.PixelTypeType | v445.PixelTypeType
DoubleWithUnit = v450.DoubleWithUnit | v445.DoubleWithUnit
RfiMaskGenerationMethodType = v450.RfiMaskGenerationMethodType | v445.RfiMaskGenerationMethodType
RfiFmmitigationMethodType = v450.RfiFmmitigationMethodType | v445.RfiFmmitigationMethodType
RangeCompressionMethodType = v450.RangeCompressionMethodType | v445.RangeCompressionMethodType
RangeReferenceFunctionType = v450.RangeReferenceFunctionType | v445.RangeReferenceFunctionType
DcMethodType = v450.DcMethodType | v445.DcMethodType
WeightingWindowType = v450.WeightingWindowType | v445.WeightingWindowType
BistaticDelayCorrectionMethodType = v450.BistaticDelayCorrectionMethodType | v445.BistaticDelayCorrectionMethodType
PolarisationType = v450.PolarisationType | v445.PolarisationType
FloatWithPolarisation = v450.FloatWithPolarisation | v445.FloatWithPolarisation
IonosphereHeightEstimationMethodType = (
    v450.IonosphereHeightEstimationMethodType | v445.IonosphereHeightEstimationMethodType
)
AutofocusMethodType = v450.AutofocusMethodType | v445.AutofocusMethodType

LATEST_VERSION = v450


def translate_min_max_type(values: MinMaxType) -> common.MinMaxType:
    """Translate minimum and maximum values from the annotation model."""
    return common.MinMaxType(min=float(values.min), max=float(values.max))


def translate_min_max_type_to_model(values: common.MinMaxType) -> LATEST_VERSION.MinMaxType:
    """Translate minimum and maximum values to the annotation model."""
    return LATEST_VERSION.MinMaxType(min=float(values.min), max=float(values.max))


def translate_min_max_type_with_unit(values: MinMaxTypeWithUnit) -> common.MinMaxTypeWithUnit:
    """Translate minimum and maximum values while retaining their shared unit."""
    if values.min.units != values.max.units:
        raise ValueError("MinMaxTypeWithUnit endpoints must use the same unit")
    return common.MinMaxTypeWithUnit(
        min=translate_float_with_unit(values.min),
        max=translate_float_with_unit(values.max),
        units=translate_uom_type(values.min.units),
    )


def translate_min_max_type_with_unit_to_model(
    values: common.MinMaxTypeWithUnit,
) -> LATEST_VERSION.MinMaxTypeWithUnit:
    """Translate minimum and maximum values with their unit to the annotation model."""
    return LATEST_VERSION.MinMaxTypeWithUnit(
        min=translate_float_with_unit_to_model(values.min, values.units),
        max=translate_float_with_unit_to_model(values.max, values.units),
    )


def translate_processing_mode(
    mode: ProcessingModeType,
) -> common.ProcessingModeType:
    """Translate nominal/parc enum"""
    return common.ProcessingModeType[mode.name]


def translate_processing_mode_to_model(
    mode: common.ProcessingModeType,
) -> LATEST_VERSION.ProcessingModeType:
    """Translate nominal/parc enum"""
    return LATEST_VERSION.ProcessingModeType[mode.name]


def translate_orbit_attitude_source(
    source: OrbitAttitudeSourceType,
) -> common.OrbitAttitudeSourceType:
    """Translate downling/auxiliary enum"""
    return common.OrbitAttitudeSourceType[source.name]


def translate_orbit_attitude_source_to_model(
    source: common.OrbitAttitudeSourceType,
) -> LATEST_VERSION.OrbitAttitudeSourceType:
    """Translate downling/auxiliary enum"""
    return LATEST_VERSION.OrbitAttitudeSourceType[source.name]


def translate_rfi_mask_type(mask_type: RfiMaskType) -> common.RfiMaskType:
    """Translate single/mutliple rfi mask enum"""
    return common.RfiMaskType[mask_type.name]


def translate_rfi_mask_type_to_model(
    mask_type: common.RfiMaskType,
) -> LATEST_VERSION.RfiMaskType:
    """Translate single/mutliple rfi mask enum"""
    return LATEST_VERSION.RfiMaskType[mask_type.name]


def translate_rfi_mitigation_method_type(
    method: RfiMitigationMethodType,
) -> common.RfiMitigationMethodType:
    """Translate time/frequency/... enum"""
    return common.RfiMitigationMethodType[method.name]


def translate_rfi_mitigation_method_type_to_model(
    method: common.RfiMitigationMethodType,
) -> LATEST_VERSION.RfiMitigationMethodType:
    """Translate time/frequency/... enum"""
    return LATEST_VERSION.RfiMitigationMethodType[method.name]


def translate_rfi_mask_generation_method_type(
    method: RfiMaskGenerationMethodType,
) -> common.RfiMaskGenerationMethodType:
    """Translate and/or enum"""
    return common.RfiMaskGenerationMethodType[method.name]


def translate_rfi_mask_generation_method_type_to_model(
    method: common.RfiMaskGenerationMethodType,
) -> LATEST_VERSION.RfiMaskGenerationMethodType:
    """Translate and/or enum"""
    return LATEST_VERSION.RfiMaskGenerationMethodType[method.name]


def translate_rfi_fmmitigation_method_type(method: RfiFmmitigationMethodType) -> common.RfiFmmitigationMethodType:
    """Translate frequency-domain RFI mitigation method."""
    return common.RfiFmmitigationMethodType[method.name]


def translate_rfi_fmmitigation_method_type_to_model(
    method: common.RfiFmmitigationMethodType,
) -> LATEST_VERSION.RfiFmmitigationMethodType:
    """Translate frequency-domain RFI mitigation method to the annotation model."""
    return LATEST_VERSION.RfiFmmitigationMethodType[method.name]


def translate_range_compression_method_type(
    method: RangeCompressionMethodType,
) -> common.RangeCompressionMethodType:
    """Translate matched filters/inverse filters enum"""
    return common.RangeCompressionMethodType[method.name]


def translate_range_compression_method_type_to_model(
    method: common.RangeCompressionMethodType,
) -> LATEST_VERSION.RangeCompressionMethodType:
    """Translate matched filters/inverse filters enum"""
    return LATEST_VERSION.RangeCompressionMethodType[method.name]


def translate_range_reference_function_type(
    source_type: RangeReferenceFunctionType,
) -> common.RangeReferenceFunctionType:
    """Translate range reference source nominal/internal/replica enum"""
    return common.RangeReferenceFunctionType[source_type.name]


def translate_range_reference_function_type_to_model(
    source_type: common.RangeReferenceFunctionType,
) -> LATEST_VERSION.RangeReferenceFunctionType:
    """Translate range reference source nominal/internal/replica enum"""
    return LATEST_VERSION.RangeReferenceFunctionType[source_type.name]


def translate_dc_method_type(method: DcMethodType) -> common.DcMethodType:
    """Translate dc method"""
    return common.DcMethodType[method.name]


def translate_dc_method_type_to_model(method: common.DcMethodType) -> LATEST_VERSION.DcMethodType:
    """Translate dc method"""
    return LATEST_VERSION.DcMethodType[method.name]


def translate_weighting_window_type(window_type: WeightingWindowType) -> common.WeightingWindowType:
    """Translate weighting window"""
    return common.WeightingWindowType[window_type.name]


def translate_weighting_window_type_to_model(
    window_type: common.WeightingWindowType,
) -> LATEST_VERSION.WeightingWindowType:
    """Translate weighting window"""
    return LATEST_VERSION.WeightingWindowType[window_type.name]


def translate_bistatic_delay_correction_method_type(
    method: BistaticDelayCorrectionMethodType,
) -> common.BistaticDelayCorrectionMethodType:
    """Translate bistatic delay correction"""
    return common.BistaticDelayCorrectionMethodType[method.name]


def translate_bistatic_delay_correction_method_type_to_model(
    method: common.BistaticDelayCorrectionMethodType,
) -> LATEST_VERSION.BistaticDelayCorrectionMethodType:
    """Translate bistatic delay correction"""
    return LATEST_VERSION.BistaticDelayCorrectionMethodType[method.name]


def translate_polarisation_type(polarisation: PolarisationType) -> common.PolarisationType:
    """Translate polarisation type"""
    return common.PolarisationType[polarisation.name]


def translate_polarisation_type_to_model(
    polarisation: common.PolarisationType,
) -> LATEST_VERSION.PolarisationType:
    """Translate polarisation type"""
    return LATEST_VERSION.PolarisationType[polarisation.name]


def translate_float_with_polarisation(float_with_polarisation: FloatWithPolarisation) -> common.FloatWithPolarisation:
    """Translate float with polarisation"""
    return float_with_polarisation.value, translate_polarisation_type(float_with_polarisation.polarisation)


def translate_float_with_polarisation_to_model(
    float_with_polarisation: common.FloatWithPolarisation,
) -> LATEST_VERSION.FloatWithPolarisation:
    """Translate float with polarisation"""
    value, pol = float_with_polarisation
    return LATEST_VERSION.FloatWithPolarisation(
        value=float(value), polarisation=translate_polarisation_type_to_model(pol)
    )


def translate_ionosphere_height_estimation_method_type(
    method: IonosphereHeightEstimationMethodType,
) -> common.IonosphereHeightEstimationMethodType:
    """Translate height estimation method"""
    return common.IonosphereHeightEstimationMethodType[method.name]


def translate_ionosphere_height_estimation_method_type_to_model(
    method: common.IonosphereHeightEstimationMethodType,
) -> LATEST_VERSION.IonosphereHeightEstimationMethodType:
    """Translate height estimation method"""
    return LATEST_VERSION.IonosphereHeightEstimationMethodType[method.name]


def translate_autofocus_method_type(method: AutofocusMethodType) -> common.AutofocusMethodType:
    """Translate autofocus method"""
    return common.AutofocusMethodType[method.name]


def translate_autofocus_method_type_to_model(
    method: common.AutofocusMethodType,
) -> LATEST_VERSION.AutofocusMethodType:
    """Translate autofocus method"""
    return LATEST_VERSION.AutofocusMethodType[method.name]


def translate_datetime(date: XmlDateTime | str) -> PreciseDateTime:
    """Translate date"""
    return PreciseDateTime.fromisoformat(str(date))


def translate_datetime_to_model(date: PreciseDateTime) -> str:
    """Translate date"""
    return date.isoformat(timespec="microseconds")[:-1]


def translate_bool(flag: str) -> bool:
    """Translate bool"""
    if flag == "true":
        return True
    elif flag == "false":
        return False
    raise RuntimeError(f"Unrecognized boolean flag: {flag} should be either 'true' or 'false'")


def translate_bool_to_model(flag: bool) -> str:
    """Translate bool"""
    return str(flag).lower()


def translate_double_with_unit(num: DoubleWithUnit) -> float:
    """Translate double with unit"""
    return num.value


def translate_double_with_unit_to_model(num: float, units: common.UomType) -> LATEST_VERSION.DoubleWithUnit:
    """Translate double with unit"""
    return LATEST_VERSION.DoubleWithUnit(value=float(num), units=translate_uom_type_to_model(units))


def translate_float_with_unit(num: FloatWithUnit) -> float:
    """Translate float with unit"""
    return num.value


def translate_float_with_unit_to_model(num: float, units: common.UomType) -> LATEST_VERSION.FloatWithUnit:
    """Translate float"""
    return LATEST_VERSION.FloatWithUnit(value=float(num), units=translate_uom_type_to_model(units))


def translate_uom_type(unit: UomType) -> common.UomType:
    """Translate unit of measure"""
    return common.UomType[unit.name]


def translate_uom_type_to_model(unit: common.UomType) -> LATEST_VERSION.UomType:
    """Translate unit of measure"""
    return LATEST_VERSION.UomType[unit.name]


def translate_float_array_with_units(coefficients: FloatArrayWithUnits) -> list[float]:
    """Translate list of floats with units"""
    values = [float(v) for v in coefficients.value.split()]
    if len(values) != coefficients.count:
        raise RuntimeError(
            "Inconsistency in float array with units: "
            + f"{coefficients.value} and count: {coefficients.count} do not match"
        )
    return values


def translate_float_array_with_units_to_model(
    coefficients: list[float], units: common.UomType
) -> LATEST_VERSION.FloatArrayWithUnits:
    """Translate list of floats with units"""
    return LATEST_VERSION.FloatArrayWithUnits(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
        units=translate_uom_type_to_model(units),
    )


def translate_float_array(
    coefficients: FloatArray,
) -> list[float]:
    """Translate list of floats"""
    values = [float(v) for v in coefficients.value.split()]
    if len(values) != coefficients.count:
        raise RuntimeError(
            "Inconsistency in float array: " + f"{coefficients.value} and count: {coefficients.count} do not match"
        )
    return values


def translate_float_array_to_model(
    coefficients: list[float],
) -> LATEST_VERSION.FloatArray:
    """Translate list of floats"""
    return LATEST_VERSION.FloatArray(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
    )


def translate_int_array(coefficients: IntArray) -> list[int]:
    """Translate integer array model to a list and validate its count."""
    values = [int(value) for value in coefficients.value.split()]
    if len(values) != coefficients.count:
        raise RuntimeError(
            "Inconsistency in integer array: " + f"{coefficients.value} and count: {coefficients.count} do not match"
        )
    return values


def translate_int_array_to_model(coefficients: list[int]) -> LATEST_VERSION.IntArray:
    """Translate a list of integers to the annotation model."""
    return LATEST_VERSION.IntArray(
        count=len(coefficients),
        value=" ".join(str(int(value)) for value in coefficients),
    )


def translate_double_array(coefficients: DoubleArray) -> list[float]:
    """Translate list of double"""
    values = [float(v) for v in coefficients.value.split()]
    if len(values) != coefficients.count:
        raise RuntimeError(
            "Inconsistency in double array: " + f"'{coefficients.value}' and count: {coefficients.count} do not match"
        )
    return values


def translate_double_array_to_model(coefficients: list[float]) -> LATEST_VERSION.DoubleArray:
    """Translate list of double"""
    return LATEST_VERSION.DoubleArray(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
    )


def translate_double_array_with_units(coefficients: DoubleArrayWithUnits) -> list[float]:
    """Translate list of double with units"""
    values = [float(v) for v in coefficients.value.split()]
    if len(values) != coefficients.count:
        raise RuntimeError(
            "Inconsistency in double array with units: "
            + f"'{coefficients.value}' and count: {coefficients.count} do not match"
        )
    return values


def translate_double_array_with_units_to_model(
    coefficients: list[float], units: common.UomType
) -> LATEST_VERSION.DoubleArrayWithUnits:
    """Translate list of double with units"""
    return LATEST_VERSION.DoubleArrayWithUnits(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
        units=translate_uom_type_to_model(units),
    )


def translate_complex(number: Complex) -> complex:
    """Translate complex number"""
    return complex(real=number.re, imag=number.im)


def translate_complex_to_model(number: complex) -> LATEST_VERSION.Complex:
    """Translate complex number"""
    return LATEST_VERSION.Complex(re=number.real, im=number.imag)


def translate_interferometric_pair(interferometric_pair: InterferometricPairType) -> tuple[int, int]:
    """Translate an InteferometricPairType to a integer pair."""
    return (
        int(interferometric_pair.primary),
        int(interferometric_pair.secondary),
    )


def translate_interferometric_pair_to_model(
    interferometric_pair: tuple[int, int],
) -> LATEST_VERSION.InterferometricPairType:
    """Translate an integer pair into an InterferometricPairType."""
    return LATEST_VERSION.InterferometricPairType(
        primary=int(interferometric_pair[0]),
        secondary=int(interferometric_pair[1]),
    )


def translate_interferometric_pair_list(
    interferometric_pair_list: InterferometricPairListType,
) -> list[tuple[int, int]]:
    """Convert an InterferometricPairListType into a list of integer pairs."""
    return [translate_interferometric_pair(p) for p in interferometric_pair_list.interferometric_pairs]


def translate_interferometric_pair_list_to_model(
    interferometric_pair_list: list[tuple[int, int]],
) -> LATEST_VERSION.InterferometricPairListType:
    """Translate an a list of integer pairs into InterferometricPairListType."""
    return LATEST_VERSION.InterferometricPairListType(
        count=len(interferometric_pair_list),
        interferometric_pairs=[translate_interferometric_pair_to_model(p) for p in interferometric_pair_list],
    )


def translate_projection_type(projection: ProjectionType) -> common.ProjectionType:
    """Translate projection"""
    return common.ProjectionType[projection.name]


def translate_projection_type_to_model(projection: common.ProjectionType) -> LATEST_VERSION.ProjectionType:
    """Translate projection"""
    return LATEST_VERSION.ProjectionType[projection.name]


def translate_geodetic_reference_frame_type(frame: GeodeticReferenceFrameType) -> common.GeodeticReferenceFrameType:
    """Translate geodetic reference frame"""
    return common.GeodeticReferenceFrameType[frame.name]


def translate_geodetic_reference_frame_type_to_model(
    frame: common.GeodeticReferenceFrameType,
) -> LATEST_VERSION.GeodeticReferenceFrameType:
    """Translate geodetic reference frame"""
    return LATEST_VERSION.GeodeticReferenceFrameType[frame.name]


def translate_datum(datum: DatumType) -> common.DatumType:
    """Translate datum"""
    return common.DatumType(
        coordinate_reference_system=datum.coordinate_reference_system,
        geodetic_reference_frame=translate_geodetic_reference_frame_type(datum.geodetic_reference_frame),
    )


def translate_datum_to_model(datum: common.DatumType) -> LATEST_VERSION.DatumType:
    """Translate datum"""
    return LATEST_VERSION.DatumType(
        coordinate_reference_system=datum.coordinate_reference_system,
        geodetic_reference_frame=translate_geodetic_reference_frame_type_to_model(datum.geodetic_reference_frame),
    )


def translate_pixel_representation_type(representation: PixelRepresentationType) -> common.PixelRepresentationType:
    """Translate pixel representation"""
    return common.PixelRepresentationType[representation.name]


def translate_pixel_representation_type_to_model(
    representation: common.PixelRepresentationType,
) -> LATEST_VERSION.PixelRepresentationType:
    """Translate pixel representation"""
    return LATEST_VERSION.PixelRepresentationType[representation.name]


def translate_pixel_type_type(pixel_type: PixelTypeType) -> common.PixelTypeType:
    """Translate pixel type"""
    return common.PixelTypeType[pixel_type.name]


def translate_pixel_type_type_to_model(pixel_type: common.PixelTypeType) -> LATEST_VERSION.PixelTypeType:
    """Translate pixel type"""
    return LATEST_VERSION.PixelTypeType[pixel_type.name]


def translate_pixel_quantity_type(quantity: PixelQuantityType) -> common.PixelQuantityType:
    """Translate pixel quantity"""
    return common.PixelQuantityType[quantity.name]


def translate_pixel_quantity_type_to_model(quantity: common.PixelQuantityType) -> LATEST_VERSION.PixelQuantityType:
    """Translate pixel quantity"""
    return LATEST_VERSION.PixelQuantityType[quantity.name]


def translate_mission_type(mission: MissionType) -> common.MissionType:
    """Translate mission type"""
    return common.MissionType[mission.name]


def translate_mission_type_to_model(mission: common.MissionType) -> LATEST_VERSION.MissionType:
    """Translate  mission type"""
    return LATEST_VERSION.MissionType[mission.name]


def translate_swath_type(swath: SwathType) -> common.SwathType:
    """Translate swath"""
    return common.SwathType[swath.name]


def translate_swath_type_to_model(swath: common.SwathType) -> LATEST_VERSION.SwathType:
    """Translate swath"""
    return LATEST_VERSION.SwathType[swath.name]


def translate_product_type(product_type: ProductType) -> common.ProductType:
    """Translate product type"""
    return common.ProductType[product_type.name]


def translate_product_type_to_model(product_type: common.ProductType) -> LATEST_VERSION.ProductType:
    """Translate product type"""
    return LATEST_VERSION.ProductType[product_type.name]


def translate_mission_phase_id(phase: MissionPhaseIdtype) -> common.MissionPhaseIdtype:
    """Translate mission phase type"""
    return common.MissionPhaseIdtype[phase.name]


def translate_mission_phase_id_to_model(phase: common.MissionPhaseIdtype) -> LATEST_VERSION.MissionPhaseIdtype:
    """Translate mission phase"""
    return LATEST_VERSION.MissionPhaseIdtype[phase.name]


def translate_sensor_mode(mode: SensorModeType) -> common.SensorModeType:
    """Translate sensor mode"""
    return common.SensorModeType[mode.name]


def translate_sensor_mode_to_model(mode: common.SensorModeType) -> LATEST_VERSION.SensorModeType:
    """Translate sensor mode"""
    return LATEST_VERSION.SensorModeType[mode.name]


def translate_orbit_pass(direction: OrbitPassType) -> common.OrbitPassType:
    """Translate orbit pass"""
    return common.OrbitPassType[direction.name]


def translate_orbit_pass_to_model(
    direction: common.OrbitPassType,
) -> LATEST_VERSION.OrbitPassType:
    """Translate orbit pass"""
    return LATEST_VERSION.OrbitPassType[direction.name]


def translate_product_composition(composition: ProductCompositionType) -> common.ProductCompositionType:
    """Translate product composition"""
    return common.ProductCompositionType[composition.name]


def translate_product_composition_to_model(
    composition: common.ProductCompositionType,
) -> LATEST_VERSION.ProductCompositionType:
    """Translate product composition"""
    return LATEST_VERSION.ProductCompositionType[composition.name]


def translate_state_type(state: StateType) -> tuple[PreciseDateTime, float]:
    """Translate state type"""
    return translate_datetime(state.azimuth_time), translate_float_with_unit(state.value)


def translate_state_type_to_model(
    state: tuple[PreciseDateTime, float], unit: common.UomType
) -> LATEST_VERSION.StateType:
    """Translate state type"""
    time, value = state
    return LATEST_VERSION.StateType(
        azimuth_time=translate_datetime_to_model(time),
        value=translate_float_with_unit_to_model(value, unit),
    )


def translate_data_format_mode(mode: DataFormatModeType) -> common.DataFormatModeType:
    """Translate data format type"""
    return common.DataFormatModeType[mode.name]


def translate_data_format_mode_to_model(
    mode: common.DataFormatModeType,
) -> LATEST_VERSION.DataFormatModeType:
    """Translate data format type"""
    return LATEST_VERSION.DataFormatModeType[mode.name]


def translate_time_with_pol(pair: TimeTypeWithPolarisation) -> tuple[common.PolarisationType, PreciseDateTime]:
    """Translate time with pol"""
    return translate_polarisation_type(pair.polarisation), translate_datetime(pair.value)


def translate_time_with_pol_to_model(
    pair: tuple[common.PolarisationType, PreciseDateTime],
) -> LATEST_VERSION.TimeTypeWithPolarisation:
    """Translate time with pol"""
    pol, time = pair
    return LATEST_VERSION.TimeTypeWithPolarisation(
        polarisation=translate_polarisation_type_to_model(pol),
        value=translate_datetime_to_model(time),
    )


def translate_slant_range_polynomial(poly: SlantRangePolynomialType) -> common.SlantRangePolynomialType:
    """Translate slant-range polynomial"""
    return common.SlantRangePolynomialType(
        azimuth_time=translate_datetime(poly.azimuth_time),
        t0=translate_double_with_unit(poly.t0),
        polynomial=translate_double_array(poly.polynomial),
    )


def translate_slant_range_polynomial_to_model(
    poly: common.SlantRangePolynomialType,
) -> LATEST_VERSION.SlantRangePolynomialType:
    """Translate slant-range polynomial"""
    return LATEST_VERSION.SlantRangePolynomialType(
        azimuth_time=translate_datetime_to_model(poly.azimuth_time),
        t0=translate_double_with_unit_to_model(poly.t0, units=common.UomType.S),
        polynomial=translate_double_array_to_model(poly.polynomial),
    )


def translate_internal_calibration_source(
    source: InternalCalibrationSourceType,
) -> common.InternalCalibrationSourceType:
    """Translate internal calibration source"""
    return common.InternalCalibrationSourceType[source.name]


def translate_internal_calibration_source_to_model(
    source: common.InternalCalibrationSourceType,
) -> LATEST_VERSION.InternalCalibrationSourceType:
    """Translate internal calibration source"""
    return LATEST_VERSION.InternalCalibrationSourceType[source.name]


def translate_height_model_base(height_model: HeightModelBaseType) -> common.HeightModelBaseType:
    """Translate height model"""
    return common.HeightModelBaseType[height_model.name]


def translate_height_model_base_to_model(
    height_model: common.HeightModelBaseType,
) -> LATEST_VERSION.HeightModelBaseType:
    """Translate height model"""
    return LATEST_VERSION.HeightModelBaseType[height_model.name]


def translate_height_model(height_model: HeightModelType) -> common.HeightModelType:
    """Translate height model"""
    return common.HeightModelType(
        value=translate_height_model_base(height_model.value),
        version=height_model.version,
    )


def translate_height_model_to_model(
    height_model: common.HeightModelType,
) -> LATEST_VERSION.HeightModelType:
    """Translate height model"""
    return LATEST_VERSION.HeightModelType(
        value=translate_height_model_base_to_model(height_model.value),
        version=height_model.version,
    )


def translate_lut_layer(layer: LayerType) -> common.LayerType:
    """Translate layer"""
    return common.LayerType[layer.name]


def translate_lut_layer_to_model(
    layer: common.LayerType,
) -> LATEST_VERSION.LayerType:
    """Translate layer"""
    return LATEST_VERSION.LayerType[layer.name]


def translate_layer_list(layers: LayerListType) -> list[common.LayerType]:
    """Translate layer list"""

    if len(layers.layer) != layers.count:
        raise RuntimeError("Inconsistency in layer list: " + f"{layers.layer} and count: {layers.count} do not match")

    return [translate_lut_layer(lut_layer) for lut_layer in layers.layer]


def translate_layer_list_to_model(
    layers: list[common.LayerType],
) -> LATEST_VERSION.LayerListType:
    """Translate layer list"""
    layer_list = [translate_lut_layer_to_model(lut_layer) for lut_layer in layers]

    return LATEST_VERSION.LayerListType(layer=layer_list, count=len(layer_list))


def translate_polarisation_combination_method(
    method: PolarisationCombinationMethodType,
) -> common.PolarisationCombinationMethodType | None:
    """Translate the polarization combination method."""
    if method.value == "None":
        return None
    return common.PolarisationCombinationMethodType[method.name]


def translate_polarisation_combination_method_to_model(
    method: common.PolarisationCombinationMethodType | None,
) -> LATEST_VERSION.PolarisationCombinationMethodType:
    """Translate the polarization combination method."""
    if method is None:
        return LATEST_VERSION.PolarisationCombinationMethodType.NONE

    return LATEST_VERSION.PolarisationCombinationMethodType[method.name]


def translate_primary_image_selection_information(
    info: PrimaryImageSelectionInformationType,
) -> common.PrimaryImageSelectionInformationType:
    """Translate the primary image selection information"""
    return common.PrimaryImageSelectionInformationType[info.name]


def translate_primary_image_selection_information_to_model(
    info: common.PrimaryImageSelectionInformationType,
) -> LATEST_VERSION.PrimaryImageSelectionInformationType:
    """Translate the primary image selection information"""
    return LATEST_VERSION.PrimaryImageSelectionInformationType[info.name]


def translate_primary_image_selection_method(
    method: PrimaryImageSelectionMethodType,
) -> common.PrimaryImageSelectionMethodType:
    """Translate the primary image selection method"""
    return common.PrimaryImageSelectionMethodType[method.name]


def translate_primary_image_selection_method_to_model(
    method: common.PrimaryImageSelectionMethodType,
) -> LATEST_VERSION.PrimaryImageSelectionMethodType:
    """Translate the primary image selection method"""
    return LATEST_VERSION.PrimaryImageSelectionMethodType[method.name]


def translate_skp_phase_calibration_postprocessing_type(
    method: SkpPhaseCalibrationPostprocessingType,
) -> common.SkpPhaseCalibrationPostprocessingType:
    """Translate SKP calibration phase-screen postprocessing method."""
    return common.SkpPhaseCalibrationPostprocessingType[method.name]


def translate_skp_phase_calibration_postprocessing_type_to_model(
    method: common.SkpPhaseCalibrationPostprocessingType,
) -> LATEST_VERSION.SkpPhaseCalibrationPostprocessingType:
    """Translate SKP calibration phase-screen postprocessing method to model."""
    return LATEST_VERSION.SkpPhaseCalibrationPostprocessingType[method.name]


def translate_coregistration_method(method: CoregistrationMethodType) -> common.CoregistrationMethodType:
    """Translate the coregistration method."""
    return common.CoregistrationMethodType[method.name]


def translate_coregistration_method_to_model(
    method: common.CoregistrationMethodType,
) -> LATEST_VERSION.CoregistrationMethodType:
    """Translate the coregistration method."""
    return LATEST_VERSION.CoregistrationMethodType[method.name]


def translate_cross_talk_list(cross_talk_list: CrossTalkList) -> common.CrossTalkList:
    """Translate cross talk list"""
    return common.CrossTalkList(
        hv_rx=translate_complex(cross_talk_list.cross_talk_hvrx),
        vh_rx=translate_complex(cross_talk_list.cross_talk_vhrx),
        vh_tx=translate_complex(cross_talk_list.cross_talk_vhtx),
        hv_tx=translate_complex(cross_talk_list.cross_talk_hvtx),
    )


def translate_cross_talk_list_to_model(cross_talk_list: common.CrossTalkList) -> LATEST_VERSION.CrossTalkList:
    """Translate cross talk list"""

    return LATEST_VERSION.CrossTalkList(
        cross_talk_hvrx=translate_complex_to_model(cross_talk_list.hv_rx),
        cross_talk_vhrx=translate_complex_to_model(cross_talk_list.vh_rx),
        cross_talk_vhtx=translate_complex_to_model(cross_talk_list.vh_tx),
        cross_talk_hvtx=translate_complex_to_model(cross_talk_list.hv_tx),
    )


def translate_channel_imbalance_list(channel_imbalance_list: ChannelImbalanceList) -> common.ChannelImbalanceList:
    """Translate channel imbalance list"""
    return common.ChannelImbalanceList(
        hv_rx=translate_complex(channel_imbalance_list.channel_imbal_hvrx),
        hv_tx=translate_complex(channel_imbalance_list.channel_imbal_hvtx),
    )


def translate_channel_imbalance_list_to_model(
    channel_imbalance_list: common.ChannelImbalanceList,
) -> LATEST_VERSION.ChannelImbalanceList:
    """Translate channel imbalance list"""

    return LATEST_VERSION.ChannelImbalanceList(
        channel_imbal_hvrx=translate_complex_to_model(channel_imbalance_list.hv_rx),
        channel_imbal_hvtx=translate_complex_to_model(channel_imbalance_list.hv_tx),
    )


def translate_channel_type(channel: ChannelType) -> common.ChannelType:
    """Translate the channel type."""
    return common.ChannelType[channel.name]


def translate_channel_type_to_model(
    channel: common.ChannelType,
) -> LATEST_VERSION.ChannelType:
    """Translate the channel type."""
    return LATEST_VERSION.ChannelType[channel.name]
