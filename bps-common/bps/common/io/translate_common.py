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
from bps.common.io import common, common_types
from xsdata.models.datatype import XmlDateTime


def translate_processing_mode(
    mode: common_types.ProcessingModeType,
) -> common.ProcessingModeType:
    """Translate nominal/parc enum"""
    return common.ProcessingModeType[mode.name]


def translate_processing_mode_to_model(
    mode: common.ProcessingModeType,
) -> common_types.ProcessingModeType:
    """Translate nominal/parc enum"""
    return common_types.ProcessingModeType[mode.name]


def translate_orbit_attitude_source(
    source: common_types.OrbitAttitudeSourceType,
) -> common.OrbitAttitudeSourceType:
    """Translate downling/auxiliary enum"""
    return common.OrbitAttitudeSourceType[source.name]


def translate_orbit_attitude_source_to_model(
    source: common.OrbitAttitudeSourceType,
) -> common_types.OrbitAttitudeSourceType:
    """Translate downling/auxiliary enum"""
    return common_types.OrbitAttitudeSourceType[source.name]


def translate_rfi_mask_type(mask_type: common_types.RfiMaskType) -> common.RfiMaskType:
    """Translate single/mutliple rfi mask enum"""
    return common.RfiMaskType[mask_type.name]


def translate_rfi_mask_type_to_model(
    mask_type: common.RfiMaskType,
) -> common_types.RfiMaskType:
    """Translate single/mutliple rfi mask enum"""
    return common_types.RfiMaskType[mask_type.name]


def translate_rfi_mitigation_method_type(
    method: common_types.RfiMitigationMethodType,
) -> common.RfiMitigationMethodType:
    """Translate time/frequency/... enum"""
    return common.RfiMitigationMethodType[method.name]


def translate_rfi_mitigation_method_type_to_model(
    method: common.RfiMitigationMethodType,
) -> common_types.RfiMitigationMethodType:
    """Translate time/frequency/... enum"""
    return common_types.RfiMitigationMethodType[method.name]


def translate_rfi_mask_generation_method_type(
    method: common_types.RfiMaskGenerationMethodType,
) -> common.RfiMaskGenerationMethodType:
    """Translate and/or enum"""
    return common.RfiMaskGenerationMethodType[method.name]


def translate_rfi_mask_generation_method_type_to_model(
    method: common.RfiMaskGenerationMethodType,
) -> common_types.RfiMaskGenerationMethodType:
    """Translate and/or enum"""
    return common_types.RfiMaskGenerationMethodType[method.name]


def translate_range_compression_method_type(
    method: common_types.RangeCompressionMethodType,
) -> common.RangeCompressionMethodType:
    """Translate matched filters/inverse filters enum"""
    return common.RangeCompressionMethodType[method.name]


def translate_range_compression_method_type_to_model(
    method: common.RangeCompressionMethodType,
) -> common_types.RangeCompressionMethodType:
    """Translate matched filters/inverse filters enum"""
    return common_types.RangeCompressionMethodType[method.name]


def translate_range_reference_function_type(
    source_type: common_types.RangeReferenceFunctionType,
) -> common.RangeReferenceFunctionType:
    """Translate range reference source nominal/internal/replica enum"""
    return common.RangeReferenceFunctionType[source_type.name]


def translate_range_reference_function_type_to_model(
    source_type: common.RangeReferenceFunctionType,
) -> common_types.RangeReferenceFunctionType:
    """Translate range reference source nominal/internal/replica enum"""
    return common_types.RangeReferenceFunctionType[source_type.name]


def translate_dc_method_type(
    method: common_types.DcMethodType,
) -> common.DcMethodType:
    """Translate dc method"""
    return common.DcMethodType[method.name]


def translate_dc_method_type_to_model(
    method: common.DcMethodType,
) -> common_types.DcMethodType:
    """Translate dc method"""
    return common_types.DcMethodType[method.name]


def translate_weighting_window_type(
    window_type: common_types.WeightingWindowType,
) -> common.WeightingWindowType:
    """Translate weighting window"""
    return common.WeightingWindowType[window_type.name]


def translate_weighting_window_type_to_model(
    window_type: common.WeightingWindowType,
) -> common_types.WeightingWindowType:
    """Translate weighting window"""
    return common_types.WeightingWindowType[window_type.name]


def translate_bistatic_delay_correction_method_type(
    method: common_types.BistaticDelayCorrectionMethodType,
) -> common.BistaticDelayCorrectionMethodType:
    """Translate bistatic delay correction"""
    return common.BistaticDelayCorrectionMethodType[method.name]


def translate_bistatic_delay_correction_method_type_to_model(
    method: common.BistaticDelayCorrectionMethodType,
) -> common_types.BistaticDelayCorrectionMethodType:
    """Translate bistatic delay correction"""
    return common_types.BistaticDelayCorrectionMethodType[method.name]


def translate_polarisation_type(
    polarisation: common_types.PolarisationType,
) -> common.PolarisationType:
    """Translate polarisation type"""
    return common.PolarisationType[polarisation.name]


def translate_polarisation_type_to_model(
    polarisation: common.PolarisationType,
) -> common_types.PolarisationType:
    """Translate polarisation type"""
    return common_types.PolarisationType[polarisation.name]


def translate_float_with_polarisation(
    float_with_polarisation: common_types.FloatWithPolarisation,
) -> common.FloatWithPolarisation:
    """Translate float with polarisation"""
    assert float_with_polarisation.value is not None
    assert float_with_polarisation.polarisation is not None
    return float_with_polarisation.value, translate_polarisation_type(float_with_polarisation.polarisation)


def translate_float_with_polarisation_to_model(
    float_with_polarisation: common.FloatWithPolarisation,
) -> common_types.FloatWithPolarisation:
    """Translate float with polarisation"""
    value, pol = float_with_polarisation
    return common_types.FloatWithPolarisation(
        value=float(value), polarisation=translate_polarisation_type_to_model(pol)
    )


def translate_ionosphere_height_estimation_method_type(
    method: common_types.IonosphereHeightEstimationMethodType,
) -> common.IonosphereHeightEstimationMethodType:
    """Translate height estimation method"""
    return common.IonosphereHeightEstimationMethodType[method.name]


def translate_ionosphere_height_estimation_method_type_to_model(
    method: common.IonosphereHeightEstimationMethodType,
) -> common_types.IonosphereHeightEstimationMethodType:
    """Translate height estimation method"""
    return common_types.IonosphereHeightEstimationMethodType[method.name]


def translate_ionosphere_type(
    ionosphere_type: common_types.IonosphereTypeType,
) -> common.IonosphereType | None:
    """Translate the ionosphere type"""
    if ionosphere_type is common_types.IonosphereTypeType.NONE:
        return None
    return common.IonosphereType[ionosphere_type.name]


def translate_ionosphere_type_to_model(
    ionosphere_type: common.IonosphereType | None,
) -> common_types.IonosphereTypeType:
    """Translate the ionosphere type to mnodel"""
    if ionosphere_type is None:
        return common_types.IonosphereTypeType.NONE
    return common_types.IonosphereTypeType[ionosphere_type.name]


def translate_autofocus_method_type(
    method: common_types.AutofocusMethodType,
) -> common.AutofocusMethodType:
    """Translate autofocus method"""
    return common.AutofocusMethodType[method.name]


def translate_autofocus_method_type_to_model(
    method: common.AutofocusMethodType,
) -> common_types.AutofocusMethodType:
    """Translate autofocus method"""
    return common_types.AutofocusMethodType[method.name]


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


def translate_double_with_unit(num: common_types.DoubleWithUnit) -> float:
    """Translate double with unit"""
    assert num.value is not None
    return num.value


def translate_double_with_unit_to_model(num: float, units: common.UomType) -> common_types.DoubleWithUnit:
    """Translate double with unit"""
    return common_types.DoubleWithUnit(float(num), units=translate_uom_type_to_model(units))


def translate_float_with_unit(num: common_types.FloatWithUnit) -> float:
    """Translate float with unit"""
    assert num.value is not None
    return num.value


def translate_float_with_unit_to_model(num: float, units: common.UomType) -> common_types.FloatWithUnit:
    """Translate float"""
    return common_types.FloatWithUnit(float(num), units=translate_uom_type_to_model(units))


def translate_uom_type(
    unit: common_types.UomType,
) -> common.UomType:
    """Translate unit of measure"""
    return common.UomType[unit.name]


def translate_uom_type_to_model(
    unit: common.UomType,
) -> common_types.UomType:
    """Translate unit of measure"""
    return common_types.UomType[unit.name]


def translate_float_array_with_units(
    coefficients: common_types.FloatArrayWithUnits,
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
) -> common_types.FloatArrayWithUnits:
    """Translate list of floats with units"""
    return common_types.FloatArrayWithUnits(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
        units=translate_uom_type_to_model(units),
    )


def translate_float_array(
    coefficients: common_types.FloatArray,
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
) -> common_types.FloatArray:
    """Translate list of floats"""
    return common_types.FloatArray(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
    )


def translate_double_array(
    coefficients: common_types.DoubleArray,
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
) -> common_types.DoubleArray:
    """Translate list of double"""
    return common_types.DoubleArray(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
    )


def translate_double_array_with_units(
    coefficients: common_types.DoubleArrayWithUnits,
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
) -> common_types.DoubleArrayWithUnits:
    """Translate list of double with units"""
    return common_types.DoubleArrayWithUnits(
        count=len(coefficients),
        value=" ".join(str(float(c)) for c in coefficients),
        units=translate_uom_type_to_model(units),
    )


def translate_complex(number: common_types.Complex) -> complex:
    """Translate complex number"""
    assert number.im is not None
    assert number.re is not None
    return complex(real=number.re, imag=number.im)


def translate_complex_to_model(number: complex) -> common_types.Complex:
    """Translate complex number"""
    return common_types.Complex(re=number.real, im=number.imag)


def translate_interferometric_pair(
    interferometric_pair: common_types.InterferometricPairType,
) -> tuple[int, int]:
    """Translate an InteferometricPairType to a integer pair."""
    return (
        int(interferometric_pair.primary),
        int(interferometric_pair.secondary),
    )


def translate_interferometric_pair_to_model(
    interferometric_pair: tuple[int, int],
) -> common_types.InterferometricPairType:
    """Translate an integer pair into an InterferometricPairType."""
    return common_types.InterferometricPairType(
        primary=int(interferometric_pair[0]),
        secondary=int(interferometric_pair[1]),
    )


def translate_interferometric_pair_list(
    interferometric_pair_list: common_types.InterferometricPairListType,
) -> list[tuple[int, int]]:
    """Convert an InterferometricPairListType into a list of integer pairs."""
    return [translate_interferometric_pair(p) for p in interferometric_pair_list.interferometric_pairs]


def translate_interferometric_pair_list_to_model(
    interferometric_pair_list: list[tuple[int, int]],
) -> common_types.InterferometricPairListType:
    """Translate an a list of integer pairs into InterferometricPairListType."""
    return common_types.InterferometricPairListType(
        count=len(interferometric_pair_list),
        interferometric_pairs=[translate_interferometric_pair_to_model(p) for p in interferometric_pair_list],
    )


def translate_projection_type(
    projection: common_types.ProjectionType,
) -> common.ProjectionType:
    """Translate projection"""
    return common.ProjectionType[projection.name]


def translate_projection_type_to_model(
    projection: common.ProjectionType,
) -> common_types.ProjectionType:
    """Translate projection"""
    return common_types.ProjectionType[projection.name]


def translate_geodetic_reference_frame_type(
    frame: common_types.GeodeticReferenceFrameType,
) -> common.GeodeticReferenceFrameType:
    """Translate geodetic reference frame"""
    return common.GeodeticReferenceFrameType[frame.name]


def translate_geodetic_reference_frame_type_to_model(
    frame: common.GeodeticReferenceFrameType,
) -> common_types.GeodeticReferenceFrameType:
    """Translate geodetic reference frame"""
    return common_types.GeodeticReferenceFrameType[frame.name]


def translate_datum(datum: common_types.DatumType) -> common.DatumType:
    """Translate datum"""
    assert datum.coordinate_reference_system is not None
    assert datum.geodetic_reference_frame is not None
    return common.DatumType(
        coordinate_reference_system=datum.coordinate_reference_system,
        geodetic_reference_frame=translate_geodetic_reference_frame_type(datum.geodetic_reference_frame),
    )


def translate_datum_to_model(datum: common.DatumType) -> common_types.DatumType:
    """Translate datum"""
    return common_types.DatumType(
        coordinate_reference_system=datum.coordinate_reference_system,
        geodetic_reference_frame=translate_geodetic_reference_frame_type_to_model(datum.geodetic_reference_frame),
    )


def translate_pixel_representation_type(
    representation: common_types.PixelRepresentationType,
) -> common.PixelRepresentationType:
    """Translate pixel representation"""
    return common.PixelRepresentationType[representation.name]


def translate_pixel_representation_type_to_model(
    representation: common.PixelRepresentationType,
) -> common_types.PixelRepresentationType:
    """Translate pixel representation"""
    return common_types.PixelRepresentationType[representation.name]


def translate_pixel_type_type(
    pixel_type: common_types.PixelTypeType,
) -> common.PixelTypeType:
    """Translate pixel type"""
    return common.PixelTypeType[pixel_type.name]


def translate_pixel_type_type_to_model(
    pixel_type: common.PixelTypeType,
) -> common_types.PixelTypeType:
    """Translate pixel type"""
    return common_types.PixelTypeType[pixel_type.name]


def translate_pixel_quantity_type(
    quantity: common_types.PixelQuantityType,
) -> common.PixelQuantityType:
    """Translate pixel quantity"""
    return common.PixelQuantityType[quantity.name]


def translate_pixel_quantity_type_to_model(
    quantity: common.PixelQuantityType,
) -> common_types.PixelQuantityType:
    """Translate pixel quantity"""
    return common_types.PixelQuantityType[quantity.name]


def translate_mission_type(
    mission: common_types.MissionType,
) -> common.MissionType:
    """Translate mission type"""
    return common.MissionType[mission.name]


def translate_mission_type_to_model(
    mission: common.MissionType,
) -> common_types.MissionType:
    """Translate  mission type"""
    return common_types.MissionType[mission.name]


def translate_swath_type(
    swath: common_types.SwathType,
) -> common.SwathType:
    """Translate swath"""
    return common.SwathType[swath.name]


def translate_swath_type_to_model(
    swath: common.SwathType,
) -> common_types.SwathType:
    """Translate swath"""
    return common_types.SwathType[swath.name]


def translate_product_type(
    product_type: common_types.ProductType,
) -> common.ProductType:
    """Translate product type"""
    return common.ProductType[product_type.name]


def translate_product_type_to_model(
    product_type: common.ProductType,
) -> common_types.ProductType:
    """Translate product type"""
    return common_types.ProductType[product_type.name]


def translate_mission_phase_id(
    phase: common_types.MissionPhaseIdtype,
) -> common.MissionPhaseIdtype:
    """Translate mission phase type"""
    return common.MissionPhaseIdtype[phase.name]


def translate_mission_phase_id_to_model(
    phase: common.MissionPhaseIdtype,
) -> common_types.MissionPhaseIdtype:
    """Translate mission phase"""
    return common_types.MissionPhaseIdtype[phase.name]


def translate_sensor_mode(
    mode: common_types.SensorModeType,
) -> common.SensorModeType:
    """Translate sensor mode"""
    return common.SensorModeType[mode.name]


def translate_sensor_mode_to_model(
    mode: common.SensorModeType,
) -> common_types.SensorModeType:
    """Translate sensor mode"""
    return common_types.SensorModeType[mode.name]


def translate_orbit_pass(
    direction: common_types.OrbitPassType,
) -> common.OrbitPassType:
    """Translate orbit pass"""
    return common.OrbitPassType[direction.name]


def translate_orbit_pass_to_model(
    direction: common.OrbitPassType,
) -> common_types.OrbitPassType:
    """Translate orbit pass"""
    return common_types.OrbitPassType[direction.name]


def translate_product_composition(
    composition: common_types.ProductCompositionType,
) -> common.ProductCompositionType:
    """Translate product composition"""
    return common.ProductCompositionType[composition.name]


def translate_product_composition_to_model(
    composition: common.ProductCompositionType,
) -> common_types.ProductCompositionType:
    """Translate product composition"""
    return common_types.ProductCompositionType[composition.name]


def translate_state_type(
    state: common_types.StateType,
) -> tuple[PreciseDateTime, float]:
    """Translate state type"""
    assert state.azimuth_time is not None
    assert state.value is not None
    return translate_datetime(state.azimuth_time), translate_float_with_unit(state.value)


def translate_state_type_to_model(state: tuple[PreciseDateTime, float], unit: common.UomType) -> common_types.StateType:
    """Translate state type"""
    time, value = state
    return common_types.StateType(
        azimuth_time=translate_datetime_to_model(time),
        value=translate_float_with_unit_to_model(value, unit),
    )


def translate_data_format_mode(
    mode: common_types.DataFormatModeType,
) -> common.DataFormatModeType:
    """Translate data format type"""
    return common.DataFormatModeType[mode.name]


def translate_data_format_mode_to_model(
    mode: common.DataFormatModeType,
) -> common_types.DataFormatModeType:
    """Translate data format type"""
    return common_types.DataFormatModeType[mode.name]


def translate_time_with_pol(
    pair: common_types.TimeTypeWithPolarisation,
) -> tuple[common.PolarisationType, PreciseDateTime]:
    """Translate time with pol"""
    assert pair.polarisation is not None
    assert pair.value is not None
    return translate_polarisation_type(pair.polarisation), translate_datetime(pair.value)


def translate_time_with_pol_to_model(
    pair: tuple[common.PolarisationType, PreciseDateTime],
) -> common_types.TimeTypeWithPolarisation:
    """Translate time with pol"""
    pol, time = pair
    return common_types.TimeTypeWithPolarisation(
        polarisation=translate_polarisation_type_to_model(pol),
        value=translate_datetime_to_model(time),
    )


def translate_float_with_pol(
    pair: common_types.FloatWithPolarisation,
) -> tuple[common.PolarisationType, float]:
    """Translate float with pol"""
    assert pair.polarisation is not None
    assert pair.value is not None
    return translate_polarisation_type(pair.polarisation), pair.value


def translate_float_with_pol_to_model(
    pair: tuple[common.PolarisationType, float],
) -> common_types.FloatWithPolarisation:
    """Translate time with pol"""
    pol, value = pair
    return common_types.FloatWithPolarisation(
        polarisation=translate_polarisation_type_to_model(pol),
        value=float(value),
    )


def translate_slant_range_polynomial(
    poly: common_types.SlantRangePolynomialType,
) -> common.SlantRangePolynomialType:
    """Translate slant-range polynomial"""
    assert poly.azimuth_time is not None
    assert poly.t0 is not None
    assert poly.polynomial is not None
    return common.SlantRangePolynomialType(
        azimuth_time=translate_datetime(poly.azimuth_time),
        t0=translate_double_with_unit(poly.t0),
        polynomial=translate_double_array(poly.polynomial),
    )


def translate_slant_range_polynomial_to_model(
    poly: common.SlantRangePolynomialType,
) -> common_types.SlantRangePolynomialType:
    """Translate slant-range polynomial"""
    return common_types.SlantRangePolynomialType(
        azimuth_time=translate_datetime_to_model(poly.azimuth_time),
        t0=translate_double_with_unit_to_model(poly.t0, units=common.UomType.S),
        polynomial=translate_double_array_to_model(poly.polynomial),
    )


def translate_internal_calibration_source(
    source: common_types.InternalCalibrationSourceType,
) -> common.InternalCalibrationSourceType:
    """Translate internal calibration source"""
    return common.InternalCalibrationSourceType[source.name]


def translate_internal_calibration_source_to_model(
    source: common.InternalCalibrationSourceType,
) -> common_types.InternalCalibrationSourceType:
    """Translate internal calibration source"""
    return common_types.InternalCalibrationSourceType[source.name]


def translate_height_model_base(
    height_model: common_types.HeightModelBaseType,
) -> common.HeightModelBaseType:
    """Translate height model"""
    return common.HeightModelBaseType[height_model.name]


def translate_height_model_base_to_model(
    height_model: common.HeightModelBaseType,
) -> common_types.HeightModelBaseType:
    """Translate height model"""
    return common_types.HeightModelBaseType[height_model.name]


def translate_height_model(
    height_model: common_types.HeightModelType,
) -> common.HeightModelType:
    """Translate height model"""
    assert height_model.value is not None
    assert height_model.version is not None
    return common.HeightModelType(
        value=translate_height_model_base(height_model.value),
        version=height_model.version,
    )


def translate_height_model_to_model(
    height_model: common.HeightModelType,
) -> common_types.HeightModelType:
    """Translate height model"""
    return common_types.HeightModelType(
        value=translate_height_model_base_to_model(height_model.value),
        version=height_model.version,
    )


def translate_lut_layer(
    layer: common_types.LayerType,
) -> common.LayerType:
    """Translate layer"""
    return common.LayerType[layer.name]


def translate_lut_layer_to_model(
    layer: common.LayerType,
) -> common_types.LayerType:
    """Translate layer"""
    return common_types.LayerType[layer.name]


def translate_layer_list(layers: common_types.LayerListType) -> list[common.LayerType]:
    """Translate layer list"""

    if len(layers.layer) != layers.count:
        raise RuntimeError("Inconsistency in layer list: " + f"{layers.layer} and count: {layers.count} do not match")

    return [translate_lut_layer(lut_layer) for lut_layer in layers.layer]


def translate_layer_list_to_model(
    layers: list[common.LayerType],
) -> common_types.LayerListType:
    """Translate layer list"""
    layer_list = [translate_lut_layer_to_model(lut_layer) for lut_layer in layers]

    return common_types.LayerListType(layer=layer_list, count=len(layer_list))


def translate_polarisation_combination_method(
    method: common_types.PolarisationCombinationMethodType,
) -> common.PolarisationCombinationMethodType | None:
    """Translate the polarization combination method."""
    if method == common_types.PolarisationCombinationMethodType.NONE:
        return None
    return common.PolarisationCombinationMethodType[method.name]


def translate_polarisation_combination_method_to_model(
    method: common.PolarisationCombinationMethodType | None,
) -> common_types.PolarisationCombinationMethodType:
    """Translate the polarization combination method."""
    if method is None:
        return common_types.PolarisationCombinationMethodType.NONE

    return common_types.PolarisationCombinationMethodType[method.name]


def translate_primary_image_selection_information(
    info: common_types.PrimaryImageSelectionInformationType,
) -> common.PrimaryImageSelectionInformationType | None:
    """Translate the primary image selection information"""
    return common.PrimaryImageSelectionInformationType[info.name]


def translate_primary_image_selection_information_to_model(
    info: common.PrimaryImageSelectionInformationType,
) -> common_types.PrimaryImageSelectionInformationType:
    """Translate the primary image selection information"""
    return common_types.PrimaryImageSelectionInformationType[info.name]


def translate_coregistration_method(
    method: common_types.CoregistrationMethodType,
) -> common.CoregistrationMethodType:
    """Translate the coregistration method."""
    return common.CoregistrationMethodType[method.name]


def translate_coregistration_method_to_model(
    method: common.CoregistrationMethodType,
) -> common_types.CoregistrationMethodType:
    """Translate the coregistration method."""
    return common_types.CoregistrationMethodType[method.name]


def translate_coregistration_execution_policy(
    policy: common_types.CoregistrationExecutionPolicyType,
) -> common.CoregistrationExecutionPolicyType:
    """Translate the coregistration execution policy."""
    return common.CoregistrationExecutionPolicyType[policy.name]


def translate_coregistration_execution_policy_to_model(
    policy: common.CoregistrationExecutionPolicyType,
) -> common_types.CoregistrationExecutionPolicyType:
    """Translate the coregistration execution policy."""
    return common_types.CoregistrationExecutionPolicyType[policy.name]


def translate_cross_talk_list(
    cross_talk_list: common_types.CrossTalkList,
) -> common.CrossTalkList:
    """Translate cross talk list"""
    assert cross_talk_list.cross_talk_hvrx is not None
    assert cross_talk_list.cross_talk_vhrx is not None
    assert cross_talk_list.cross_talk_vhtx is not None
    assert cross_talk_list.cross_talk_hvtx is not None

    return common.CrossTalkList(
        hv_rx=translate_complex(cross_talk_list.cross_talk_hvrx),
        vh_rx=translate_complex(cross_talk_list.cross_talk_vhrx),
        vh_tx=translate_complex(cross_talk_list.cross_talk_vhtx),
        hv_tx=translate_complex(cross_talk_list.cross_talk_hvtx),
    )


def translate_cross_talk_list_to_model(
    cross_talk_list: common.CrossTalkList,
) -> common_types.CrossTalkList:
    """Translate cross talk list"""

    return common_types.CrossTalkList(
        cross_talk_hvrx=translate_complex_to_model(cross_talk_list.hv_rx),
        cross_talk_vhrx=translate_complex_to_model(cross_talk_list.vh_rx),
        cross_talk_vhtx=translate_complex_to_model(cross_talk_list.vh_tx),
        cross_talk_hvtx=translate_complex_to_model(cross_talk_list.hv_tx),
    )


def translate_channel_imbalance_list(
    channel_imbalance_list: common_types.ChannelImbalanceList,
) -> common.ChannelImbalanceList:
    """Translate channel imbalance list"""
    assert channel_imbalance_list.channel_imbal_hvrx is not None
    assert channel_imbalance_list.channel_imbal_hvtx is not None

    return common.ChannelImbalanceList(
        hv_rx=translate_complex(channel_imbalance_list.channel_imbal_hvrx),
        hv_tx=translate_complex(channel_imbalance_list.channel_imbal_hvtx),
    )


def translate_channel_imbalance_list_to_model(
    channel_imbalance_list: common.ChannelImbalanceList,
) -> common_types.ChannelImbalanceList:
    """Translate channel imbalance list"""

    return common_types.ChannelImbalanceList(
        channel_imbal_hvrx=translate_complex_to_model(channel_imbalance_list.hv_rx),
        channel_imbal_hvtx=translate_complex_to_model(channel_imbalance_list.hv_tx),
    )
