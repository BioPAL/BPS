# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Main annotation L1ab translation"""

from bps.common.io import translate_common
from bps.transcoder.io import (
    main_annotation_l1ab,
    main_annotation_models_l1ab,
    translate_common_annotation_l1,
)


def translate_main_annotation_l1ab(
    annotation: main_annotation_models_l1ab.MainAnnotation,
) -> main_annotation_l1ab.MainAnnotationL1ab:
    """Translate L1ab main annotation file"""
    assert annotation.acquisition_information is not None
    assert annotation.sar_image is not None
    assert annotation.instrument_parameters is not None
    assert annotation.raw_data_analysis is not None
    assert annotation.processing_parameters is not None
    assert annotation.internal_calibration is not None
    assert annotation.rfi_mitigation is not None
    assert annotation.doppler_parameters is not None
    assert annotation.radiometric_calibration is not None
    assert annotation.polarimetric_distortion is not None
    assert annotation.ionosphere_correction is not None
    assert annotation.geometry is not None
    assert annotation.quality is not None
    assert annotation.annotation_lut is not None

    return main_annotation_l1ab.MainAnnotationL1ab(
        acquisition_information=translate_common_annotation_l1.translate_acquisition_information(
            annotation.acquisition_information
        ),
        sar_image=translate_common_annotation_l1.translate_sar_image(annotation.sar_image),
        instrument_parameters=translate_common_annotation_l1.translate_instrument_parameters(
            annotation.instrument_parameters
        ),
        raw_data_analysis=translate_common_annotation_l1.translate_raw_data_analysis(annotation.raw_data_analysis),
        processing_parameters=translate_common_annotation_l1.translate_processing_parameters(
            annotation.processing_parameters
        ),
        internal_calibration=translate_common_annotation_l1.translate_internal_calibration(
            annotation.internal_calibration
        ),
        rfi_mitigation=translate_common_annotation_l1.translate_rfi_mitigation(annotation.rfi_mitigation),
        doppler_parameters=translate_common_annotation_l1.translate_doppler_parameters(annotation.doppler_parameters),
        radiometric_calibration=translate_common_annotation_l1.translate_radiometric_calibration(
            annotation.radiometric_calibration
        ),
        polarimetric_distortion=translate_common_annotation_l1.translate_polarimetric_distortion(
            annotation.polarimetric_distortion
        ),
        ionosphere_correction=translate_common_annotation_l1.translate_ionosphere_correction(
            annotation.ionosphere_correction
        ),
        geometry=translate_common_annotation_l1.translate_geometry(annotation.geometry),
        quality=translate_common_annotation_l1.translate_quality(annotation.quality),
        annotation_lut=translate_common.translate_layer_list(annotation.annotation_lut),
    )


def translate_main_annotation_l1ab_to_model(
    annotation: main_annotation_l1ab.MainAnnotationL1ab,
) -> main_annotation_models_l1ab.MainAnnotation:
    """Translate L1ab main annotation file"""

    return main_annotation_models_l1ab.MainAnnotation(
        acquisition_information=translate_common_annotation_l1.translate_acquisition_information_to_model(
            annotation.acquisition_information
        ),
        sar_image=translate_common_annotation_l1.translate_sar_image_to_model(annotation.sar_image),
        instrument_parameters=translate_common_annotation_l1.translate_instrument_parameters_to_model(
            annotation.instrument_parameters
        ),
        raw_data_analysis=translate_common_annotation_l1.translate_raw_data_analysis_to_model(
            annotation.raw_data_analysis
        ),
        processing_parameters=translate_common_annotation_l1.translate_processing_parameters_to_model(
            annotation.processing_parameters
        ),
        internal_calibration=translate_common_annotation_l1.translate_internal_calibration_to_model(
            annotation.internal_calibration
        ),
        rfi_mitigation=translate_common_annotation_l1.translate_rfi_mitigation_to_model(annotation.rfi_mitigation),
        doppler_parameters=translate_common_annotation_l1.translate_doppler_parameters_to_model(
            annotation.doppler_parameters
        ),
        radiometric_calibration=translate_common_annotation_l1.translate_radiometric_calibration_to_model(
            annotation.radiometric_calibration
        ),
        polarimetric_distortion=translate_common_annotation_l1.translate_polarimetric_distortion_to_model(
            annotation.polarimetric_distortion
        ),
        ionosphere_correction=translate_common_annotation_l1.translate_ionosphere_correction_to_model(
            annotation.ionosphere_correction
        ),
        geometry=translate_common_annotation_l1.translate_geometry_to_model(annotation.geometry),
        quality=translate_common_annotation_l1.translate_quality_to_model(annotation.quality),
        annotation_lut=translate_common.translate_layer_list_to_model(annotation.annotation_lut),
    )
