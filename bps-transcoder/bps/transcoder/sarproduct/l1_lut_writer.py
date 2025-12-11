# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""L1 LUT writer module"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
from arepytools.geometry.generalsarorbit import GSO3DCurveWrapper, create_general_sar_orbit
from arepytools.geometry.geometric_functions import compute_incidence_angles
from arepytools.io import metadata
from arepytools.math.genericpoly import create_sorted_poly_list
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.transcoder.sarproduct.biomass_l1product import BIOMASSL1Product
from bps.transcoder.sarproduct.generic_product import GenericProduct
from bps.transcoder.utils.constants import (
    ALPHAINV,
    AVERAGE_GROUND_VELOCITY,
    CARRIER_FREQUENCY,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    H,
)
from bps.transcoder.utils.production_model_utils import (
    translate_global_coverage_id,
    translate_major_cycle_id,
    translate_repeat_cycle_id,
)
from netCDF4 import Dataset
from scipy.constants import speed_of_light as LIGHTSPEED

ZETA = LIGHTSPEED * H / (4.0 * np.pi**2 * ELECTRON_MASS * ALPHAINV)
TEC_FACTOR = LIGHTSPEED * ELECTRON_MASS * CARRIER_FREQUENCY**2 / (ELEMENTARY_CHARGE * ZETA * 1e-9)
PHI_FACTOR = TEC_FACTOR * 4.0 * np.pi * ZETA / (LIGHTSPEED * CARRIER_FREQUENCY)
DELTA_R_FACTOR = LIGHTSPEED * ELECTRON_MASS / ELEMENTARY_CHARGE * 1e9


class ProductLUTID(Enum):
    """Products that can be exported in the LUTs"""

    SAR_DEM = auto()
    RFI_TIME_MASK = auto()
    RFI_FREQ_MASK = auto()
    FR = auto()
    FR_PLANE = auto()
    PHASE_SCREEN_BB = auto()
    PHASE_SCREEN_AF = auto()
    SLC_NESZ_MAP = auto()


@dataclass
class LutProducts:
    """List of lut products"""

    sar_dem: GenericProduct
    rfi_time_mask: GenericProduct | None
    rfi_freq_mask: GenericProduct | None
    fr: GenericProduct | None
    fr_plane: GenericProduct | None
    phase_screen_bb: GenericProduct | None
    phase_screen_af: GenericProduct | None
    slc_nesz_map: GenericProduct | None

    @classmethod
    def from_dict(cls, product_lut: dict[ProductLUTID, GenericProduct | None]):
        """Fill data class"""
        sar_dem_product = product_lut.get(ProductLUTID.SAR_DEM)
        if sar_dem_product is None:
            raise RuntimeError("SlantDEM intermediate product not available")

        return cls(
            sar_dem=sar_dem_product,
            rfi_time_mask=product_lut.get(ProductLUTID.RFI_TIME_MASK),
            rfi_freq_mask=product_lut.get(ProductLUTID.RFI_FREQ_MASK),
            fr=product_lut.get(ProductLUTID.FR),
            fr_plane=product_lut.get(ProductLUTID.FR_PLANE),
            phase_screen_bb=product_lut.get(ProductLUTID.PHASE_SCREEN_BB),
            phase_screen_af=product_lut.get(ProductLUTID.PHASE_SCREEN_AF),
            slc_nesz_map=product_lut.get(ProductLUTID.SLC_NESZ_MAP),
        )


def _generate_zeros_lut(shape, num_channels, dtype):
    return [np.full(shape, 0, dtype=dtype) for _ in range(num_channels)]


def _build_default_range_freq_axis(range_axis) -> np.ndarray:
    sampling_frequency = 1.0 / (range_axis[1] - range_axis[0])
    size = len(range_axis)
    delta_freq = sampling_frequency / size
    return delta_freq * np.arange(size)


def _phase_screen_to_tec(phase_screen: np.ndarray) -> np.ndarray:
    return TEC_FACTOR / PHI_FACTOR / 1e16 * phase_screen


def _phase_screen_to_range_shifts(phase_screen: np.ndarray) -> np.ndarray:
    return LIGHTSPEED / CARRIER_FREQUENCY / 4.0 / np.pi * phase_screen


def _phase_screen_to_azimuth_shifts(
    phase_screen: np.ndarray, rg_axis: np.ndarray, az_axis: np.ndarray, dr: metadata.DopplerRateVector
) -> np.ndarray:
    phase_screen_derivative = np.gradient(phase_screen, axis=0)
    dr_values = create_sorted_poly_list(dr).evaluate((dr.get_poly(0).t_ref_az, np.array(rg_axis)))
    dr_values = np.tile(dr_values, (phase_screen.shape[0], 1))
    return (
        1
        / 2
        / np.pi
        * phase_screen_derivative
        / (az_axis[1] - az_axis[0])
        / np.abs(dr_values)
        * AVERAGE_GROUND_VELOCITY
    )


def _get_lat_lon_h_luts(dem_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_shape = dem_xyz.shape[0:2]
    dem_llh = xyz2llh(dem_xyz.reshape(-1, 3).T)
    lat = np.rad2deg(dem_llh[0, :].reshape(output_shape))
    lon = np.rad2deg(dem_llh[1, :].reshape(output_shape))
    height = dem_llh[2, :].reshape(output_shape)
    return lat, lon, height


@dataclass
class SARDemData:
    """Data from SARDem product"""

    xyz: np.ndarray
    elevation_angles: np.ndarray
    incidence_angles: np.ndarray

    @classmethod
    def from_data_list(cls, data: list[np.ndarray]):
        """Map product channels to class"""
        return cls(xyz=np.stack(data[:3], axis=-1), elevation_angles=data[3], incidence_angles=data[4])


@dataclass
class GeometricLUTData:
    """Output LUTs"""

    lat: np.ndarray
    lon: np.ndarray
    height: np.ndarray
    incidence_angle_deg: np.ndarray
    elevation_angle_deg: np.ndarray
    terrain_slope: np.ndarray
    sigma_nought: np.ndarray
    gamma_nought: np.ndarray


def compute_geometric_lut(
    dem_data: SARDemData, trajectory: TwiceDifferentiable3DCurve, azimuth_axis: np.ndarray
) -> GeometricLUTData:
    """Compute geometric look up tables from SARDem product"""

    lat, lon, height = _get_lat_lon_h_luts(dem_data.xyz)

    sensor_positions = trajectory.evaluate(azimuth_axis)
    sensor_positions = np.broadcast_to(sensor_positions.reshape(azimuth_axis.size, 1, 3), dem_data.xyz.shape).reshape(
        -1, 3
    )

    # incidence angles on inflated ellipsoid
    incidence_angles = np.rad2deg(
        np.reshape(
            compute_incidence_angles(sensor_positions=sensor_positions, points=dem_data.xyz.reshape(-1, 3)),
            dem_data.xyz.shape[:2],
        )
    )

    elevation_angle = np.rad2deg(dem_data.elevation_angles)
    terrain_slope = incidence_angles - np.rad2deg(dem_data.incidence_angles)
    sigma_nought = np.sin(dem_data.incidence_angles)
    gamma_nought = np.tan(dem_data.incidence_angles)

    return GeometricLUTData(
        lat=lat,
        lon=lon,
        height=height,
        incidence_angle_deg=incidence_angles,
        elevation_angle_deg=elevation_angle,
        terrain_slope=terrain_slope,
        sigma_nought=sigma_nought,
        gamma_nought=gamma_nought,
    )


def write_lut_file(
    lut_file: Path,
    product: BIOMASSL1Product,
    lut_data: dict[ProductLUTID, GenericProduct | None],
) -> list[str]:
    """Write the netcdf lut file, return list of added LUTs"""

    ncfile = Dataset(str(lut_file), mode="w", format="NETCDF4", clobber=True)

    assert product.start_time is not None
    assert product.stop_time is not None
    assert product.mission_phase_id is not None
    assert product.global_coverage_id is not None
    assert product.major_cycle_id is not None
    assert product.repeat_cycle_id is not None
    assert product.orbit_number is not None
    assert product.track_number is not None
    assert product.orbit_direction is not None
    assert product.datatake_id is not None

    start_time = product.start_time.isoformat(timespec="microseconds")[:-1]

    # Reference time used for axis definition
    reference_time_iso = product.start_time.isoformat(timespec="microseconds")[:-1]
    reference_time = PreciseDateTime.fromisoformat(reference_time_iso)

    # Add ATTRIBUTES (global, common for all the LUT elements)
    ncfile.mission = product.mission
    ncfile.swath = product.swath_list[0]
    ncfile.productType = product.type
    ncfile.polarisationList = product.polarization_list
    ncfile.startTime = start_time
    ncfile.stopTime = product.stop_time.isoformat(timespec="microseconds")[:-1]
    ncfile.missionPhaseID = product.mission_phase_id[0:3]
    ncfile.driftPhaseFlag = "False"
    ncfile.sensorMode = "Measurement"
    ncfile.globalCoverageID = np.uint16(translate_global_coverage_id(product.global_coverage_id))
    ncfile.majorCycleID = np.uint16(translate_major_cycle_id(product.major_cycle_id))
    ncfile.repeatCycleID = np.uint16(translate_repeat_cycle_id(product.repeat_cycle_id))
    ncfile.absoluteOrbitNumber = np.uint16(product.orbit_number)
    ncfile.relativeOrbitNumber = np.uint16(product.track_number)
    ncfile.orbitPass = product.orbit_direction.title()
    ncfile.platformHeading = 0.0
    ncfile.dataTakeID = np.uint32(product.datatake_id)
    ncfile.frame = np.uint16(product.frame_number)
    ncfile.productComposition = product.frame_status.title()
    ncfile.noDataValueBool = "False"
    ncfile.noDataValueFloat = product.lut_parameters.no_pixel_value
    ncfile.referenceAzimuthTime = reference_time_iso

    products = LutProducts.from_dict(lut_data)

    # Dimensions
    def _add_axis(name: str, units: str, values: np.ndarray, dtype: np.dtype = np.float32):
        ncfile.createDimension(name, len(values))
        range_var = ncfile.createVariable(name, dtype, (name,))
        range_var.units = units
        range_var[:] = values

    def _add_axis_pair(
        azimuth_name: str,
        range_name: str,
        azimuth_times: np.ndarray,
        range_times: np.ndarray,
        *,
        range_var_units: str = "s",
        dtype: np.dtype = np.float32,
    ):
        """Add axis pair"""
        _add_axis(
            name=range_name,
            units=range_var_units,
            values=range_times,
            dtype=dtype,
        )

        _add_axis(
            name=azimuth_name,
            units="s",
            values=azimuth_times - reference_time,
            dtype=dtype,
        )

    # - RGC axes
    rgc_ax = ("relativeAzimuthTimeRGC", "slantRangeTimeRGC")
    azimuth_ax_rgc = products.sar_dem.lines_axis_list[0]
    range_ax_rgc = products.sar_dem.samples_axis_list[0]

    # - RAW axes
    raw_ax = ("relativeAzimuthTimeRAW", "slantRangeTimeRAW")
    reference_raw_product = products.rfi_time_mask or products.rfi_freq_mask
    if reference_raw_product is not None:
        azimuth_ax_raw = reference_raw_product.lines_axis_list[0]
        range_ax_raw = reference_raw_product.samples_axis_list[0]
        _add_axis_pair(raw_ax[0], raw_ax[1], azimuth_ax_raw, range_ax_raw)

    else:
        # Defaults to RGC axes
        _add_axis_pair(raw_ax[0], raw_ax[1], azimuth_ax_rgc, range_ax_rgc)

    # - RAW axes - range frequency
    values = None
    reference_raw_freq_product = products.rfi_freq_mask
    if reference_raw_freq_product is not None:
        values = reference_raw_freq_product.samples_axis_list[0]
    else:
        values = _build_default_range_freq_axis(azimuth_ax_rgc)
    assert values is not None
    raw_freq_ax = (raw_ax[0], "rangeFreqRAW")
    _add_axis(
        name=raw_freq_ax[1],
        units="Hz",
        values=values,
    )

    # Put here for proper ordering in the nc file
    _add_axis_pair(rgc_ax[0], rgc_ax[1], azimuth_ax_rgc, range_ax_rgc, dtype=np.float64)

    # SLC axes
    slc_ax = ("relativeAzimuthTimeSLC", "slantRangeTimeSLC")
    reference_slc_product = (
        products.phase_screen_bb
        if products.phase_screen_bb
        else (
            products.fr
            if products.fr
            else (products.phase_screen_af if products.phase_screen_af else products.slc_nesz_map)
        )
    )
    if reference_slc_product is not None:
        azimuth_ax_slc = reference_slc_product.lines_axis_list[0]
        range_ax_slc = reference_slc_product.samples_axis_list[0]
        _add_axis_pair(slc_ax[0], slc_ax[1], azimuth_ax_slc, range_ax_slc)
        az_slc_size = len(azimuth_ax_slc)
        rg_slc_size = len(range_ax_slc)
    else:
        # Defaults to rgc axis
        _add_axis_pair(slc_ax[0], slc_ax[1], azimuth_ax_rgc, range_ax_rgc)
        az_slc_size = len(azimuth_ax_rgc)
        rg_slc_size = len(range_ax_rgc)
    slc_shape = (az_slc_size, rg_slc_size)

    # (Groups of) Variables
    @dataclass
    class LutVar:
        """Variable for a LUT"""

        name: str
        description: str
        units: str | None
        data: np.ndarray
        data_type: Any
        axes: Any
        compression: str = "zlib"
        compression_level: int = 2

    @dataclass
    class _LutAdder:
        added_luts: list[str] = field(default_factory=list)

        def add_lut(self, group, lut_var: LutVar):
            """Add a look up table to the group"""
            var = group.createVariable(
                lut_var.name,
                lut_var.data_type,
                dimensions=lut_var.axes,
                compression=lut_var.compression,
                complevel=lut_var.compression_level,
                fill_value=0 if lut_var.data_type == "B" else product.lut_parameters.no_pixel_value,
            )
            var.description = lut_var.description
            if lut_var.units is not None:
                var.units = lut_var.units
            var[:, :] = lut_var.data
            self.added_luts.append(var.name)

    lut_adder = _LutAdder()

    rfi_group = None
    if products.rfi_time_mask is not None or products.rfi_freq_mask is not None:
        rfi_group = ncfile.createGroup("rfiMitigation")

    # - RFI time masks on RAW axes
    if products.rfi_time_mask is not None:
        bps_logger.debug("Exporting RFI time domain masks to LUT")
        rfi_time_data = {pol: data for pol, data in zip(product.polarization_list, products.rfi_time_mask.data_list)}

        rfi_time_vars = [
            LutVar(
                f"rfiTimeMask{pol.replace('/', '')}",
                f"RFI Time Domain Mask {pol}",
                None,
                data,
                "B",
                raw_ax,
            )
            for pol, data in rfi_time_data.items()
        ]

        assert rfi_group is not None
        for rfi_time_mask in rfi_time_vars:
            lut_adder.add_lut(rfi_group, rfi_time_mask)

    # - RFI frequency masks on RAW axes
    if products.rfi_freq_mask is not None:
        bps_logger.debug("Exporting RFI frequency domain masks to LUT")
        rfi_freq_data = {pol: data for pol, data in zip(product.polarization_list, products.rfi_freq_mask.data_list)}

        rfi_freq_vars = [
            LutVar(
                f"rfiFreqMask{pol.replace('/', '')}",
                f"RFI Frequency Domain Mask {pol}",
                None,
                data > 0,
                "B",
                raw_freq_ax,
            )
            for pol, data in rfi_freq_data.items()
        ]

        assert rfi_group is not None
        for rfi_freq_mask in rfi_freq_vars:
            lut_adder.add_lut(rfi_group, rfi_freq_mask)

    # - Ionosphere variables on SLC axes
    ionospheric_vars = []
    iono_var_info = (np.float32, slc_ax)

    if products.phase_screen_bb is not None:
        bb_data_list = products.phase_screen_bb.data_list
        bb_data = bb_data_list[0]
        ionospheric_vars.append(LutVar("phaseScreen", "Phase Screen (Bickel & Bates)", "rad", bb_data, *iono_var_info))
        ionospheric_vars.append(LutVar("tec", "TEC", "TECU", _phase_screen_to_tec(bb_data), *iono_var_info))
        ionospheric_vars.append(
            LutVar("rangeShifts", "Range Shifts", "m", _phase_screen_to_range_shifts(bb_data), *iono_var_info)
        )
        ionospheric_vars.append(
            LutVar(
                "azimuthShifts",
                "Azimuth Shifts",
                "m",
                _phase_screen_to_azimuth_shifts(bb_data, range_ax_slc, azimuth_ax_slc, product.dr_vector_list[0]),
                *iono_var_info,
            )
        )

    if products.fr_plane is not None:
        fp_data_list = products.fr_plane.data_list
        fp_data = fp_data_list[0]
        ionospheric_vars.append(
            LutVar("faradayRotationPlane", "Faraday Rotation Plane", "rad", fp_data, *iono_var_info)
        )

    if products.fr is not None:
        fr_data_list = products.fr.data_list
        fr_data = fr_data_list[0]
        ionospheric_vars.extend(
            [
                LutVar("faradayRotation", "Faraday Rotation", "rad", fr_data, *iono_var_info),
                LutVar(
                    "faradayRotationStd",
                    "Faraday Rotation Std",
                    "rad",
                    np.full(slc_shape, product.lut_parameters.no_pixel_value, dtype=float),
                    *iono_var_info,
                ),
            ]
        )

    if products.phase_screen_af is not None:
        af_data_list = products.phase_screen_af.data_list
        af_data = af_data_list[0]
        ionospheric_vars.extend(
            [
                LutVar(
                    "autofocusPhaseScreen",
                    "Phase Screen (Autofocus)",
                    "rad",
                    af_data,
                    *iono_var_info,
                ),
                LutVar(
                    "autofocusPhaseScreenStd",
                    "Phase Screen (Autofocus) Std",
                    "rad",
                    np.full(slc_shape, product.lut_parameters.no_pixel_value, dtype=float),
                    *iono_var_info,
                ),
            ]
        )

    if len(ionospheric_vars) > 0:
        bps_logger.debug("Exporting Ionospheric calibration related data to LUT")
        group = ncfile.createGroup("ionosphereCorrection")
        for iono_var in ionospheric_vars:
            lut_adder.add_lut(group, iono_var)

    # - Denoising vectors on SLC axes
    bps_logger.debug("Exporting Denoising data to LUT")
    noise_data = (
        products.slc_nesz_map.data_list
        if products.slc_nesz_map is not None
        else _generate_zeros_lut(slc_shape, len(product.polarization_list), float)
    )

    noise_data = {
        pol: np.abs(data[0 : slc_shape[0], 0 : slc_shape[1]])
        for pol, data in zip(product.polarization_list, noise_data)
    }

    noise_vars = [
        LutVar(
            f"denoising{pol.replace('/', '')}",
            f"Denoising {pol}",
            None,
            data,
            np.float32,
            slc_ax,
        )
        for pol, data in noise_data.items()
    ]

    group = ncfile.createGroup("denoising")
    for noise_var in noise_vars:
        lut_adder.add_lut(group, noise_var)

    # - Geometry-based info on RGC axes
    bps_logger.debug("Exporting Geometric (DEM) data to LUT")
    trajectory = GSO3DCurveWrapper(
        create_general_sar_orbit(product.general_sar_orbit[0], ignore_anx_after_orbit_start=True)
    )
    sardem_data = SARDemData.from_data_list(products.sar_dem.data_list)
    geometric_data = compute_geometric_lut(sardem_data, trajectory, azimuth_ax_rgc)

    llh_var_info = np.float64, rgc_ax  # NOTE: LLh floating precision is critical.
    geo_var_info = np.float32, rgc_ax
    dem_vars = [
        LutVar("latitude", "Latitude", "deg", geometric_data.lat, *llh_var_info),
        LutVar("longitude", "Longitude", "deg", geometric_data.lon, *llh_var_info),
        LutVar("height", "Height", "m", geometric_data.height, *llh_var_info),
        LutVar("incidenceAngle", "Incidence Angle", "deg", geometric_data.incidence_angle_deg, *geo_var_info),
        LutVar("elevationAngle", "Elevation Angle", "deg", geometric_data.elevation_angle_deg, *geo_var_info),
        LutVar("terrainSlope", "Terrain Slope", None, geometric_data.terrain_slope, *geo_var_info),
    ]

    group = ncfile.createGroup("geometry")
    for dem_var in dem_vars:
        lut_adder.add_lut(group, dem_var)

    radio_var_info = (np.float32, rgc_ax)
    dem_vars = [
        LutVar("sigmaNought", "Sigma Nought", None, geometric_data.sigma_nought, *radio_var_info),
        LutVar("gammaNought", "Gamma Nought", None, geometric_data.gamma_nought, *radio_var_info),
    ]

    group = ncfile.createGroup("radiometry")
    for dem_var in dem_vars:
        lut_adder.add_lut(group, dem_var)

    # Write LUT annotation file: Finalize Dataset
    ncfile.close()

    return lut_adder.added_luts
