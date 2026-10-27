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
from bps.transcoder.io.v450 import common as common_v450
from xsdata.models.datatype import XmlDateTime


def translate_min_max_type(values: common_v450.MinMaxType) -> common.MinMaxType:
    """Translate minimum and maximum values from the annotation model."""
    return common.MinMaxType(min=float(values.min), max=float(values.max))


def translate_min_max_type_to_model(values: common.MinMaxType) -> common_v450.MinMaxType:
    """Translate minimum and maximum values to the annotation model."""
    return common_v450.MinMaxType(min=float(values.min), max=float(values.max))


def translate_min_max_type_with_unit(values: common_v450.MinMaxTypeWithUnit) -> common.MinMaxTypeWithUnit:
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
) -> common_v450.MinMaxTypeWithUnit:
    """Translate minimum and maximum values with their unit to the annotation model."""
    return common_v450.MinMaxTypeWithUnit(
        min=translate_float_with_unit_to_model(values.min, values.units),
        max=translate_float_with_unit_to_model(values.max, values.units),
    )


def translate_processing_mode(
    mode: common_v450.ProcessingModeType,
) -> common.ProcessingModeType:
    """Translate nominal/parc enum"""
    return common.ProcessingModeType[mode.name]


def translate_processing_mode_to_model(
    mode: common.ProcessingModeType,
) -> common_v450.ProcessingModeType:
    """Translate nominal/parc enum"""
    return common_v450.ProcessingModeType[mode.name]


def translate_orbit_attitude_source(
    source: common_v450.OrbitAttitudeSourceType,
) -> common.OrbitAttitudeSourceType:
    """Translate downling/auxiliary enum"""
    return common.OrbitAttitudeSourceType[source.name]


def translate_orbit_attitude_source_to_model(
    source: common.OrbitAttitudeSourceType,
) -> common_v450.OrbitAttitudeSourceType:
    """Translate downling/auxiliary enum"""
    return common_v450.OrbitAttitudeSourceType[source.name]


def translate_rfi_mask_type(mask_type: common_v450.RfiMaskType) -> common.RfiMaskType:
    """Translate single/mutliple rfi mask enum"""
    return common.RfiMaskType[mask_type.name]


def translate_rfi_mask_type_to_model(
    mask_type: common.RfiMaskType,
) -> common_v450.RfiMaskType:
    """Translate single/mutliple rfi mask enum"""
    return common_v450.RfiMaskType[mask_type.name]


def translate_rfi_mitigation_method_type(
    method: common_v450.RfiMitigationMethodType,
) -> common.RfiMitigationMethodType:
    """Translate time/frequency/... enum"""
    return common.RfiMitigationMethodType[method.name]


def translate_rfi_mitigation_method_type_to_model(
    method: common.RfiMitigationMethodType,
) -> common_v450.RfiMitigationMethodType:
    """Translate time/frequency/... enum"""
    return common_v450.RfiMitigationMethodType[method.name]


def translate_rfi_mask_generation_method_type(
    method: common_v450.RfiMaskGenerationMethodType,
) -> common.RfiMaskGenerationMethodType:
    """Translate and/or enum"""
    return common.RfiMaskGenerationMethodType[method.name]


def translate_rfi_mask_generation_method_type_to_model(
    method: common.RfiMaskGenerationMethodType,
) -> common_v450.RfiMaskGenerationMethodType:
    """Translate and/or enum"""
    return common_v450.RfiMaskGenerationMethodType[method.name]


def translate_rfi_fmmitigation_method_type(
    method: common_v450.RfiFmmitigationMethodType,
) -> common.RfiFmmitigationMethodType:
    """Translate frequency-domain RFI mitigation method."""
    return common.RfiFmmitigationMethodType[method.name]


def translate_rfi_fmmitigation_method_type_to_model(
    method: common.RfiFmmitigationMethodType,
) -> common_v450.RfiFmmitigationMethodType:
    """Translate frequency-domain RFI mitigation method to the annotation model."""
    return common_v450.RfiFmmitigationMethodType[method.name]


def translate_range_compression_method_type(
    method: common_v450.RangeCompressionMethodType,
) -> common.RangeCompressionMethodType:
    """Translate matched filters/inverse filters enum"""
    return common.RangeCompressionMethodType[method.name]


def translate_range_compression_method_type_to_model(
    method: common.RangeCompressionMethodType,
) -> common_v450.RangeCompressionMethodType:
    """Translate matched filters/inverse filters enum"""
    return common_v450.RangeCompressionMethodType[method.name]


def translate_range_reference_function_type(
    source_type: common_v450.RangeReferenceFunctionType,
) -> common.RangeReferenceFunctionType:
    """Translate range reference source nominal/internal/replica enum"""
    return common.RangeReferenceFunctionType[source_type.name]


def translate_range_reference_function_type_to_model(
    source_type: common.RangeReferenceFunctionType,
) -> common_v450.RangeReferenceFunctionType:
    """Translate range reference source nominal/internal/replica enum"""
    return common_v450.RangeReferenceFunctionType[source_type.name]


def translate_dc_method_type(
    method: common_v450.DcMethodType,
) -> common.DcMethodType:
    """Translate dc method"""
    return common.DcMethodType[method.name]


def translate_dc_method_type_to_model(
    method: common.DcMethodType,
) -> common_v450.DcMethodType:
    """Translate dc method"""
    return common_v450.DcMethodType[method.name]


def translate_weighting_window_type(
    window_type: common_v450.WeightingWindowType,
) -> common.WeightingWindowType:
    """Translate weighting window"""
    return common.WeightingWindowType[window_type.name]


def translate_weighting_window_type_to_model(
    window_type: common.WeightingWindowType,
) -> common_v450.WeightingWindowType:
    """Translate weighting window"""
    return common_v450.WeightingWindowType[window_type.name]


def translate_bistatic_delay_correction_method_type(
    method: common_v450.BistaticDelayCorrectionMethodType,
) -> common.BistaticDelayCorrectionMethodType:
    """Translate bistatic delay correction"""
    return common.BistaticDelayCorrectionMethodType[method.name]


def translate_bistatic_delay_correction_method_type_to_model(
    method: common.BistaticDelayCorrectionMethodType,
) -> common_v450.BistaticDelayCorrectionMethodType:
    """Translate bistatic delay correction"""
    return common_v450.BistaticDelayCorrectionMethodType[method.name]


def translate_polarisation_type(
    polarisation: common_v450.PolarisationType,
) -> common.PolarisationType:
    """Translate polarisation type"""
    return common.PolarisationType[polarisation.name]


def translate_polarisation_type_to_model(
    polarisation: common.PolarisationType,
) -> common_v450.PolarisationType:
    """Translate polarisation type"""
    return common_v450.PolarisationType[polarisation.name]


def translate_float_with_polarisation(
    float_with_polarisation: common_v450.FloatWithPolarisation,
) -> common.FloatWithPolarisation:
    """Translate float with polarisation"""
    return float_with_polarisation.value, translate_polarisation_type(float_with_polarisation.polarisation)


def translate_float_with_polarisation_to_model(
    float_with_polarisation: common.FloatWithPolarisation,
) -> common_v450.FloatWithPolarisation:
    """Translate float with polarisation"""
    value, pol = float_with_polarisation
    return common_v450.FloatWithPolarisation(value=float(value), polarisation=translate_polarisation_type_to_model(pol))


def translate_ionosphere_height_estimation_method_type(
    method: common_v450.IonosphereHeightEstimationMethodType,
) -> common.IonosphereHeightEstimationMethodType:
    """Translate height estimation method"""
    return common.IonosphereHeightEstimationMethodType[method.name]


def translate_ionosphere_height_estimation_method_type_to_model(
    method: common.IonosphereHeightEstimationMethodType,
) -> common_v450.IonosphereHeightEstimationMethodType:
    """Translate height estimation method"""
    return common_v450.IonosphereHeightEstimationMethodType[method.name]


def translate_autofocus_method_type(
    method: common_v450.AutofocusMethodType,
) -> common.AutofocusMethodType:
    """Translate autofocus method"""
    return common.AutofocusMethodType[method.name]


def translate_autofocus_method_type_to_model(
    method: common.AutofocusMethodType,
) -> common_v450.AutofocusMethodType:
    """Translate autofocus method"""
    return common_v450.AutofocusMethodType[method.name]


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


def translate_double_with_unit(num: common_v450.DoubleWithUnit) -> float:
    """Translate double with unit"""
    return num.value


def translate_double_with_unit_to_model(num: float, units: common.UomType) -> common_v450.DoubleWithUnit:
    """Translate double with unit"""
    return common_v450.DoubleWithUnit(value=float(num), units=translate_uom_type_to_model(units))


def translate_float_with_unit(num: common_v450.FloatWithUnit) -> float:
    """Translate float with unit"""
    return num.value


def translate_float_with_unit_to_model(num: float, units: common.UomType) -> common_v450.FloatWithUnit:
    """Translate float"""
    return common_v450.FloatWithUnit(value=float(num), units=translate_uom_type_to_model(units))


def translate_uom_type(
    unit: common_v450.UomType,
) -> common.UomType:
    """Translate unit of measure"""
    return common.UomType[unit.name]


def translate_uom_type_to_model(
    unit: common.UomType,
) -> common_v450.UomType:
    """Translate unit of measure"""
    return common_v450.UomType[unit.name]


def translate_float_array_with_units(
    coefficients: common_v450.FloatArrayWithUnits,
) -> list[float]:
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
) -> common_v450.FloatArrayWithUnits:
    """Translate list of floats with units"""
    return common_v450.FloatArrayWithUnits(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
        units=translate_uom_type_to_model(units),
    )


def translate_float_array(
    coefficients: common_v450.FloatArray,
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
) -> common_v450.FloatArray:
    """Translate list of floats"""
    return common_v450.FloatArray(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
    )


def translate_int_array(coefficients: common_v450.IntArray) -> list[int]:
    """Translate integer array model to a list and validate its count."""
    values = [int(value) for value in coefficients.value.split()]
    if len(values) != coefficients.count:
        raise RuntimeError(
            "Inconsistency in integer array: " + f"{coefficients.value} and count: {coefficients.count} do not match"
        )
    return values


def translate_int_array_to_model(coefficients: list[int]) -> common_v450.IntArray:
    """Translate a list of integers to the annotation model."""
    return common_v450.IntArray(
        count=len(coefficients),
        value=" ".join(str(int(value)) for value in coefficients),
    )


def translate_double_array(
    coefficients: common_v450.DoubleArray,
) -> list[float]:
    """Translate list of double"""
    values = [float(v) for v in coefficients.value.split()]
    if len(values) != coefficients.count:
        raise RuntimeError(
            "Inconsistency in double array: " + f"'{coefficients.value}' and count: {coefficients.count} do not match"
        )
    return values


def translate_double_array_to_model(
    coefficients: list[float],
) -> common_v450.DoubleArray:
    """Translate list of double"""
    return common_v450.DoubleArray(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
    )


def translate_double_array_with_units(
    coefficients: common_v450.DoubleArrayWithUnits,
) -> list[float]:
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
) -> common_v450.DoubleArrayWithUnits:
    """Translate list of double with units"""
    return common_v450.DoubleArrayWithUnits(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
        units=translate_uom_type_to_model(units),
    )


def translate_complex(number: common_v450.Complex) -> complex:
    """Translate complex number"""
    return complex(real=number.re, imag=number.im)


def translate_complex_to_model(number: complex) -> common_v450.Complex:
    """Translate complex number"""
    return common_v450.Complex(re=number.real, im=number.imag)


def translate_interferometric_pair(
    interferometric_pair: common_v450.InterferometricPairType,
) -> tuple[int, int]:
    """Translate an InteferometricPairType to a integer pair."""
    return (
        int(interferometric_pair.primary),
        int(interferometric_pair.secondary),
    )


def translate_interferometric_pair_to_model(
    interferometric_pair: tuple[int, int],
) -> common_v450.InterferometricPairType:
    """Translate an integer pair into an InterferometricPairType."""
    return common_v450.InterferometricPairType(
        primary=int(interferometric_pair[0]),
        secondary=int(interferometric_pair[1]),
    )


def translate_interferometric_pair_list(
    interferometric_pair_list: common_v450.InterferometricPairListType,
) -> list[tuple[int, int]]:
    """Convert an InterferometricPairListType into a list of integer pairs."""
    return [translate_interferometric_pair(p) for p in interferometric_pair_list.interferometric_pairs]


def translate_interferometric_pair_list_to_model(
    interferometric_pair_list: list[tuple[int, int]],
) -> common_v450.InterferometricPairListType:
    """Translate an a list of integer pairs into InterferometricPairListType."""
    return common_v450.InterferometricPairListType(
        count=len(interferometric_pair_list),
        interferometric_pairs=[translate_interferometric_pair_to_model(p) for p in interferometric_pair_list],
    )


def translate_projection_type(
    projection: common_v450.ProjectionType,
) -> common.ProjectionType:
    """Translate projection"""
    return common.ProjectionType[projection.name]


def translate_projection_type_to_model(
    projection: common.ProjectionType,
) -> common_v450.ProjectionType:
    """Translate projection"""
    return common_v450.ProjectionType[projection.name]


def translate_geodetic_reference_frame_type(
    frame: common_v450.GeodeticReferenceFrameType,
) -> common.GeodeticReferenceFrameType:
    """Translate geodetic reference frame"""
    return common.GeodeticReferenceFrameType[frame.name]


def translate_geodetic_reference_frame_type_to_model(
    frame: common.GeodeticReferenceFrameType,
) -> common_v450.GeodeticReferenceFrameType:
    """Translate geodetic reference frame"""
    return common_v450.GeodeticReferenceFrameType[frame.name]


def translate_datum(datum: common_v450.DatumType) -> common.DatumType:
    """Translate datum"""
    return common.DatumType(
        coordinate_reference_system=datum.coordinate_reference_system,
        geodetic_reference_frame=translate_geodetic_reference_frame_type(datum.geodetic_reference_frame),
    )


def translate_datum_to_model(datum: common.DatumType) -> common_v450.DatumType:
    """Translate datum"""
    return common_v450.DatumType(
        coordinate_reference_system=datum.coordinate_reference_system,
        geodetic_reference_frame=translate_geodetic_reference_frame_type_to_model(datum.geodetic_reference_frame),
    )


def translate_pixel_representation_type(
    representation: common_v450.PixelRepresentationType,
) -> common.PixelRepresentationType:
    """Translate pixel representation"""
    return common.PixelRepresentationType[representation.name]


def translate_pixel_representation_type_to_model(
    representation: common.PixelRepresentationType,
) -> common_v450.PixelRepresentationType:
    """Translate pixel representation"""
    return common_v450.PixelRepresentationType[representation.name]


def translate_pixel_type_type(
    pixel_type: common_v450.PixelTypeType,
) -> common.PixelTypeType:
    """Translate pixel type"""
    return common.PixelTypeType[pixel_type.name]


def translate_pixel_type_type_to_model(
    pixel_type: common.PixelTypeType,
) -> common_v450.PixelTypeType:
    """Translate pixel type"""
    return common_v450.PixelTypeType[pixel_type.name]


def translate_pixel_quantity_type(
    quantity: common_v450.PixelQuantityType,
) -> common.PixelQuantityType:
    """Translate pixel quantity"""
    return common.PixelQuantityType[quantity.name]


def translate_pixel_quantity_type_to_model(
    quantity: common.PixelQuantityType,
) -> common_v450.PixelQuantityType:
    """Translate pixel quantity"""
    return common_v450.PixelQuantityType[quantity.name]


def translate_mission_type(
    mission: common_v450.MissionType,
) -> common.MissionType:
    """Translate mission type"""
    return common.MissionType[mission.name]


def translate_mission_type_to_model(
    mission: common.MissionType,
) -> common_v450.MissionType:
    """Translate  mission type"""
    return common_v450.MissionType[mission.name]


def translate_swath_type(
    swath: common_v450.SwathType,
) -> common.SwathType:
    """Translate swath"""
    return common.SwathType[swath.name]


def translate_swath_type_to_model(
    swath: common.SwathType,
) -> common_v450.SwathType:
    """Translate swath"""
    return common_v450.SwathType[swath.name]


def translate_product_type(
    product_type: common_v450.ProductType,
) -> common.ProductType:
    """Translate product type"""
    return common.ProductType[product_type.name]


def translate_product_type_to_model(
    product_type: common.ProductType,
) -> common_v450.ProductType:
    """Translate product type"""
    return common_v450.ProductType[product_type.name]


def translate_mission_phase_id(
    phase: common_v450.MissionPhaseIdtype,
) -> common.MissionPhaseIdtype:
    """Translate mission phase type"""
    return common.MissionPhaseIdtype[phase.name]


def translate_mission_phase_id_to_model(
    phase: common.MissionPhaseIdtype,
) -> common_v450.MissionPhaseIdtype:
    """Translate mission phase"""
    return common_v450.MissionPhaseIdtype[phase.name]


def translate_sensor_mode(
    mode: common_v450.SensorModeType,
) -> common.SensorModeType:
    """Translate sensor mode"""
    return common.SensorModeType[mode.name]


def translate_sensor_mode_to_model(
    mode: common.SensorModeType,
) -> common_v450.SensorModeType:
    """Translate sensor mode"""
    return common_v450.SensorModeType[mode.name]


def translate_orbit_pass(
    direction: common_v450.OrbitPassType,
) -> common.OrbitPassType:
    """Translate orbit pass"""
    return common.OrbitPassType[direction.name]


def translate_orbit_pass_to_model(
    direction: common.OrbitPassType,
) -> common_v450.OrbitPassType:
    """Translate orbit pass"""
    return common_v450.OrbitPassType[direction.name]


def translate_product_composition(
    composition: common_v450.ProductCompositionType,
) -> common.ProductCompositionType:
    """Translate product composition"""
    return common.ProductCompositionType[composition.name]


def translate_product_composition_to_model(
    composition: common.ProductCompositionType,
) -> common_v450.ProductCompositionType:
    """Translate product composition"""
    return common_v450.ProductCompositionType[composition.name]


def translate_state_type(
    state: common_v450.StateType,
) -> tuple[PreciseDateTime, float]:
    """Translate state type"""
    return translate_datetime(state.azimuth_time), translate_float_with_unit(state.value)


def translate_state_type_to_model(state: tuple[PreciseDateTime, float], unit: common.UomType) -> common_v450.StateType:
    """Translate state type"""
    time, value = state
    return common_v450.StateType(
        azimuth_time=translate_datetime_to_model(time),
        value=translate_float_with_unit_to_model(value, unit),
    )


def translate_data_format_mode(
    mode: common_v450.DataFormatModeType,
) -> common.DataFormatModeType:
    """Translate data format type"""
    return common.DataFormatModeType[mode.name]


def translate_data_format_mode_to_model(
    mode: common.DataFormatModeType,
) -> common_v450.DataFormatModeType:
    """Translate data format type"""
    return common_v450.DataFormatModeType[mode.name]


def translate_time_with_pol(
    pair: common_v450.TimeTypeWithPolarisation,
) -> tuple[common.PolarisationType, PreciseDateTime]:
    """Translate time with pol"""
    return translate_polarisation_type(pair.polarisation), translate_datetime(pair.value)


def translate_time_with_pol_to_model(
    pair: tuple[common.PolarisationType, PreciseDateTime],
) -> common_v450.TimeTypeWithPolarisation:
    """Translate time with pol"""
    pol, time = pair
    return common_v450.TimeTypeWithPolarisation(
        polarisation=translate_polarisation_type_to_model(pol),
        value=translate_datetime_to_model(time),
    )


def translate_slant_range_polynomial(
    poly: common_v450.SlantRangePolynomialType,
) -> common.SlantRangePolynomialType:
    """Translate slant-range polynomial"""
    return common.SlantRangePolynomialType(
        azimuth_time=translate_datetime(poly.azimuth_time),
        t0=translate_double_with_unit(poly.t0),
        polynomial=translate_double_array(poly.polynomial),
    )


def translate_slant_range_polynomial_to_model(
    poly: common.SlantRangePolynomialType,
) -> common_v450.SlantRangePolynomialType:
    """Translate slant-range polynomial"""
    return common_v450.SlantRangePolynomialType(
        azimuth_time=translate_datetime_to_model(poly.azimuth_time),
        t0=translate_double_with_unit_to_model(poly.t0, units=common.UomType.S),
        polynomial=translate_double_array_to_model(poly.polynomial),
    )


def translate_internal_calibration_source(
    source: common_v450.InternalCalibrationSourceType,
) -> common.InternalCalibrationSourceType:
    """Translate internal calibration source"""
    return common.InternalCalibrationSourceType[source.name]


def translate_internal_calibration_source_to_model(
    source: common.InternalCalibrationSourceType,
) -> common_v450.InternalCalibrationSourceType:
    """Translate internal calibration source"""
    return common_v450.InternalCalibrationSourceType[source.name]


def translate_height_model_base(
    height_model: common_v450.HeightModelBaseType,
) -> common.HeightModelBaseType:
    """Translate height model"""
    return common.HeightModelBaseType[height_model.name]


def translate_height_model_base_to_model(
    height_model: common.HeightModelBaseType,
) -> common_v450.HeightModelBaseType:
    """Translate height model"""
    return common_v450.HeightModelBaseType[height_model.name]


def translate_height_model(
    height_model: common_v450.HeightModelType,
) -> common.HeightModelType:
    """Translate height model"""
    return common.HeightModelType(
        value=translate_height_model_base(height_model.value),
        version=height_model.version,
    )


def translate_height_model_to_model(
    height_model: common.HeightModelType,
) -> common_v450.HeightModelType:
    """Translate height model"""
    return common_v450.HeightModelType(
        value=translate_height_model_base_to_model(height_model.value),
        version=height_model.version,
    )


def translate_lut_layer(
    layer: common_v450.LayerType,
) -> common.LayerType:
    """Translate layer"""
    return common.LayerType[layer.name]


def translate_lut_layer_to_model(
    layer: common.LayerType,
) -> common_v450.LayerType:
    """Translate layer"""
    return common_v450.LayerType[layer.name]


def translate_layer_list(layers: common_v450.LayerListType) -> list[common.LayerType]:
    """Translate layer list"""

    if len(layers.layer) != layers.count:
        raise RuntimeError("Inconsistency in layer list: " + f"{layers.layer} and count: {layers.count} do not match")

    return [translate_lut_layer(lut_layer) for lut_layer in layers.layer]


def translate_layer_list_to_model(
    layers: list[common.LayerType],
) -> common_v450.LayerListType:
    """Translate layer list"""
    layer_list = [translate_lut_layer_to_model(lut_layer) for lut_layer in layers]

    return common_v450.LayerListType(layer=layer_list, count=len(layer_list))


def translate_polarisation_combination_method(
    method: common_v450.PolarisationCombinationMethodType,
) -> common.PolarisationCombinationMethodType | None:
    """Translate the polarization combination method."""
    if method == common_v450.PolarisationCombinationMethodType.NONE:
        return None
    return common.PolarisationCombinationMethodType[method.name]


def translate_polarisation_combination_method_to_model(
    method: common.PolarisationCombinationMethodType | None,
) -> common_v450.PolarisationCombinationMethodType:
    """Translate the polarization combination method."""
    if method is None:
        return common_v450.PolarisationCombinationMethodType.NONE

    return common_v450.PolarisationCombinationMethodType[method.name]


def translate_primary_image_selection_information(
    info: common_v450.PrimaryImageSelectionInformationType,
) -> common.PrimaryImageSelectionInformationType:
    """Translate the primary image selection information"""
    return common.PrimaryImageSelectionInformationType[info.name]


def translate_primary_image_selection_information_to_model(
    info: common.PrimaryImageSelectionInformationType,
) -> common_v450.PrimaryImageSelectionInformationType:
    """Translate the primary image selection information"""
    return common_v450.PrimaryImageSelectionInformationType[info.name]


def translate_primary_image_selection_method(
    method: common_v450.PrimaryImageSelectionMethodType,
) -> common.PrimaryImageSelectionMethodType:
    """Translate the primary image selection method"""
    return common.PrimaryImageSelectionMethodType[method.name]


def translate_primary_image_selection_method_to_model(
    method: common.PrimaryImageSelectionMethodType,
) -> common_v450.PrimaryImageSelectionMethodType:
    """Translate the primary image selection method"""
    return common_v450.PrimaryImageSelectionMethodType[method.name]


def translate_skp_phase_calibration_postprocessing_type(
    method: common_v450.SkpPhaseCalibrationPostprocessingType,
) -> common.SkpPhaseCalibrationPostprocessingType:
    """Translate SKP calibration phase-screen postprocessing method."""
    return common.SkpPhaseCalibrationPostprocessingType[method.name]


def translate_skp_phase_calibration_postprocessing_type_to_model(
    method: common.SkpPhaseCalibrationPostprocessingType,
) -> common_v450.SkpPhaseCalibrationPostprocessingType:
    """Translate SKP calibration phase-screen postprocessing method to model."""
    return common_v450.SkpPhaseCalibrationPostprocessingType[method.name]


def translate_coregistration_method(
    method: common_v450.CoregistrationMethodType,
) -> common.CoregistrationMethodType:
    """Translate the coregistration method."""
    return common.CoregistrationMethodType[method.name]


def translate_coregistration_method_to_model(
    method: common.CoregistrationMethodType,
) -> common_v450.CoregistrationMethodType:
    """Translate the coregistration method."""
    return common_v450.CoregistrationMethodType[method.name]


def translate_cross_talk_list(
    cross_talk_list: common_v450.CrossTalkList,
) -> common.CrossTalkList:
    """Translate cross talk list"""
    return common.CrossTalkList(
        hv_rx=translate_complex(cross_talk_list.cross_talk_hvrx),
        vh_rx=translate_complex(cross_talk_list.cross_talk_vhrx),
        vh_tx=translate_complex(cross_talk_list.cross_talk_vhtx),
        hv_tx=translate_complex(cross_talk_list.cross_talk_hvtx),
    )


def translate_cross_talk_list_to_model(
    cross_talk_list: common.CrossTalkList,
) -> common_v450.CrossTalkList:
    """Translate cross talk list"""

    return common_v450.CrossTalkList(
        cross_talk_hvrx=translate_complex_to_model(cross_talk_list.hv_rx),
        cross_talk_vhrx=translate_complex_to_model(cross_talk_list.vh_rx),
        cross_talk_vhtx=translate_complex_to_model(cross_talk_list.vh_tx),
        cross_talk_hvtx=translate_complex_to_model(cross_talk_list.hv_tx),
    )


def translate_channel_imbalance_list(
    channel_imbalance_list: common_v450.ChannelImbalanceList,
) -> common.ChannelImbalanceList:
    """Translate channel imbalance list"""
    return common.ChannelImbalanceList(
        hv_rx=translate_complex(channel_imbalance_list.channel_imbal_hvrx),
        hv_tx=translate_complex(channel_imbalance_list.channel_imbal_hvtx),
    )


def translate_channel_imbalance_list_to_model(
    channel_imbalance_list: common.ChannelImbalanceList,
) -> common_v450.ChannelImbalanceList:
    """Translate channel imbalance list"""

    return common_v450.ChannelImbalanceList(
        channel_imbal_hvrx=translate_complex_to_model(channel_imbalance_list.hv_rx),
        channel_imbal_hvtx=translate_complex_to_model(channel_imbalance_list.hv_tx),
    )


def translate_channel_type(
    channel: common_v450.ChannelType,
) -> common.ChannelType:
    """Translate the channel type."""
    return common.ChannelType[channel.name]


def translate_channel_type_to_model(
    channel: common.ChannelType,
) -> common_v450.ChannelType:
    """Translate the channel type."""
    return common_v450.ChannelType[channel.name]
