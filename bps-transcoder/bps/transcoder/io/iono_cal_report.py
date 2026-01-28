# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Ionospheric calibration report
------------------------------
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from bps.common.io.common import IonosphereHeightEstimationMethodType
from bps.transcoder.io.common_annotation_l1 import IonosphereCorrection


@dataclass
class IonosphericCalibrationReport:
    """Ionospheric calibration report"""

    ionosphere_correction: IonosphereCorrection
    phase_screen_correction_applied: bool
    group_delay_correction_applied: bool
    constant_sign_geomagnetic_field: bool


def _translate_method(method: str) -> IonosphereHeightEstimationMethodType:
    if method == "None":
        return IonosphereHeightEstimationMethodType.NA
    if method == "Auto":
        return IonosphereHeightEstimationMethodType.AUTOMATIC
    if method == "Model":
        return IonosphereHeightEstimationMethodType.MODEL
    if method == "FeatureTracking":
        return IonosphereHeightEstimationMethodType.FEATURE_TRACKING
    if method == "SquintSensitivity":
        return IonosphereHeightEstimationMethodType.SQUINT_SENSITIVITY

    raise RuntimeError(f"Unknown ionospheric height estimation method: {method}")


def _translate_bool(val: str) -> bool:
    return val == "true"


def read_iono_cal_report(report: Path) -> IonosphericCalibrationReport:
    """Read ionospheric calibration report"""

    text = report.read_text()

    content = {child.tag: child.text for child in ET.fromstring(text)}

    height_used_str = content.get("HeightUsed")
    height_estimated_str = content.get("HeightEstimated")
    height_estimation_method_selected_str = content.get("HeightEstimationMethodSelected")
    height_estimation_latitude_value_str = content.get("HeightEstimationLatitudeValue")
    height_estimation_flag_str = content.get("HeightEstimationFlag")
    height_estimation_method_used_str = content.get("HeightEstimationMethodUsed")
    gaussian_filter_computation_flag_str = content.get("GaussianFilterComputationFlag")
    faraday_rotation_correction_applied_str = content.get("FaradayRotationCorrectionApplied")
    phase_screen_correction_applied_str = content.get("PhaseScreenCorrectionApplied")
    group_delay_correction_applied_str = content.get("GroupDelayCorrectionApplied")
    constant_geomagnetic_field_str = content.get("ConstantSignGeomagneticField")

    if (
        height_used_str is None
        or height_estimated_str is None
        or height_estimation_method_selected_str is None
        or height_estimation_latitude_value_str is None
        or height_estimation_flag_str is None
        or height_estimation_method_used_str is None
        or gaussian_filter_computation_flag_str is None
        or faraday_rotation_correction_applied_str is None
        or phase_screen_correction_applied_str is None
        or group_delay_correction_applied_str is None
        or constant_geomagnetic_field_str is None
    ):
        raise RuntimeError(f"Missing mandatory field in Ionospheric Calibration report file: {report}")

    selected_method = _translate_method(height_estimation_method_selected_str)
    method_used = _translate_method(height_estimation_method_used_str)
    method_used = (
        method_used
        if method_used != IonosphereHeightEstimationMethodType.NA
        else IonosphereHeightEstimationMethodType.FIXED
    )

    assert selected_method in (
        IonosphereHeightEstimationMethodType.NA,
        IonosphereHeightEstimationMethodType.SQUINT_SENSITIVITY,
        IonosphereHeightEstimationMethodType.FEATURE_TRACKING,
    )

    # The empty value for the xml report is '0.0' while in the annotation '-1.0' is used.
    empty_float = -1.0
    ionosphere_height_estimated = (
        float(height_estimated_str) if selected_method != IonosphereHeightEstimationMethodType.NA else empty_float
    )

    return IonosphericCalibrationReport(
        ionosphere_correction=IonosphereCorrection(
            ionosphere_height_used=float(height_used_str),
            ionosphere_height_estimated=ionosphere_height_estimated,
            ionosphere_height_estimation_method_selected=selected_method,
            ionosphere_height_estimation_latitude_value=float(height_estimation_latitude_value_str),
            ionosphere_height_estimation_flag=_translate_bool(height_estimation_flag_str),
            ionosphere_height_estimation_method_used=method_used,
            gaussian_filter_computation_flag=_translate_bool(gaussian_filter_computation_flag_str),
            faraday_rotation_correction_applied=_translate_bool(faraday_rotation_correction_applied_str),
            autofocus_shifts_applied=False,
        ),
        phase_screen_correction_applied=_translate_bool(phase_screen_correction_applied_str),
        group_delay_correction_applied=_translate_bool(group_delay_correction_applied_str),
        constant_sign_geomagnetic_field=_translate_bool(constant_geomagnetic_field_str),
    )
