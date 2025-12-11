# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Reader for the BIOMASS L1c Products
-----------------------------------
"""

import xml.etree.ElementTree as ET
from numbers import Number
from pathlib import Path
from typing import Any

import arepytools.io.metadata as metadata
import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.constants import LIGHT_SPEED
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common_types
from bps.common.io.parsing import parse
from bps.common.io.translate_common import translate_datetime
from bps.transcoder.auxiliaryfiles.aux_attitude import read_attitude_file
from bps.transcoder.io import (
    aux_orb_models,
    main_annotation_models_l1c,
    translate_common_annotation_l1,
)
from bps.transcoder.sarproduct.biomass_stackproduct import (
    LUT_LAYERS,
    REQUIRED_LUT_LAYERS,
    BIOMASSStackCoregistrationParameters,
    BIOMASSStackInSARParameters,
    BIOMASSStackProcessingParameters,
    BIOMASSStackProduct,
    BIOMASSStackProductConfiguration,
    BIOMASSStackQuality,
    InvalidBIOMASSStackProductError,
)
from bps.transcoder.sarproduct.footprint_utils import parse_footprint_string
from bps.transcoder.sarproduct.mph import MPH_NAMESPACES
from bps.transcoder.sarproduct.navigation_files_utils import (
    translate_attitude_file_to_attitude_info,
)
from bps.transcoder.sarproduct.sarproduct import SARProduct
from bps.transcoder.sarproduct.sta.product_content import BIOMASSStackProductStructure
from bps.transcoder.utils.gdal_utils import read_geotiff
from bps.transcoder.utils.product_name import parse_l1product_name
from netCDF4 import Dataset
from osgeo import gdal

gdal.UseExceptions()


class BIOMASSStackProductReader:
    """Read a BIOMASS L1c product (Stack product)."""

    def __init__(self, product_path: Path, *, nodata_fill_value: Number | None = None):
        """
        Instantiate an L1c product reader.

        Parameters
        ----------
        product_path: Path
            The path to an existing (valid) L1c product.

        nodata_fill_value: Optional[Number] = np.nan
            Optionally a user-defined provided value to fill the no-data
            pixels. If None, the no-data value of the L1c product is used.

        """
        self.product_path = Path(product_path)
        self.parsed_product_name = parse_l1product_name(self.product_path.name)
        self.is_monitoring = self.parsed_product_name.is_monitoring
        self.baseline_id = self.parsed_product_name.baseline
        self.global_coverage_id = self.parsed_product_name.coverage
        self.repeat_cycle_id = self.parsed_product_name.repeat_cycle
        self.major_cycle_id = self.parsed_product_name.major_cycle
        self.frame_number = self.parsed_product_name.frame_number
        self.track_number = self.parsed_product_name.track_number
        self.product_primary = SARProduct()
        self.product = SARProduct()
        self.stack_processing_parameters = None
        self.stack_coregistration_parameters = None
        self.stack_in_sarparameters = None
        self.stack_quality = None
        self.nodata_fill_value = nodata_fill_value
        self.product_structure = BIOMASSStackProductStructure(
            self.product_path,
            is_monitoring=self.is_monitoring,
            exists_ok=True,
        )

    def read(self, *, annotation_only: bool = False) -> BIOMASSStackProduct:
        """
        Read data associated to the L1c product.

        Arguments
        ---------
        annotation_only: bool = False
            Skip reading the measurement (data).

        Return
        ------
        BIOMASSStackProduct
            The parsed L1c product.

        """
        self.read_as_sarproduct(annotation_only=annotation_only)
        return BIOMASSStackProduct(
            product=self.product,
            product_primary=self.product_primary,
            stack_processing_parameters=self.stack_processing_parameters,
            stack_coregistration_parameters=self.stack_coregistration_parameters,
            stack_in_sarparameters=self.stack_in_sarparameters,
            stack_quality=self.stack_quality,
            is_monitoring=self.is_monitoring,
            configuration=BIOMASSStackProductConfiguration(
                frame_number=self.frame_number,
                frame_status=self.frame_status,
                product_baseline=self.baseline_id,
                product_nodata_value=self.product_nodata_value,
            ),
            mission_phase_id=self.mission_phase_id,
            datatake_id=self.datatake_id,
            orbit_number=self.orbit_number,
            global_coverage_id=self.global_coverage_id,
            major_cycle_id=self.major_cycle_id,
            repeat_cycle_id=self.repeat_cycle_id,
            track_number=self.track_number,
            platform_heading=self.platform_heading,
            first_sample_sr_time=self.first_sample_sr_time,
            first_line_az_time=self.first_line_az_time,
            rg_time_interval=self.rg_time_interval,
            az_time_interval=self.az_time_interval,
            number_of_samples=self.number_of_samples,
            number_of_lines=self.number_of_lines,
            stack_id=self.product.stack_id,
        )

    def read_as_sarproduct(self, *, annotation_only: bool = False) -> SARProduct:
        """
        Read the object's SAR information.

        Parameters
        ----------
        annotation_only: bool = False
            Skip reading the measurement (data).

        Return
        ------
        SARProduct
            The L1c product.

        """
        bps_logger.debug(
            "Reading BIOMASS L1c %s product",
            "monitoring" if self.is_monitoring else "standard",
        )

        # Read annotation files for primary and coregistered image.
        bps_logger.debug("Reading annotation files of primary image")
        self.__read_main_annotation_file(is_primary_image=True)

        bps_logger.debug("Reading annotation files of coregistered image")
        self.__read_main_annotation_file(is_primary_image=False)

        # Read the MPH file.
        self.__read_mph_file()

        # Read orbit and attitude files for primary and coregistered image.
        bps_logger.debug("Reading orbit and attitude files of primary image")
        self.__read_orbit_file(is_primary_image=True)
        self.__read_attitude_file(is_primary_image=True)

        bps_logger.debug("Reading orbit and attitude files of coregistered image")
        self.__read_orbit_file(is_primary_image=False)
        self.__read_attitude_file(is_primary_image=False)

        # If monitoring, we are done here.
        if self.is_monitoring:
            return self.product

        # Read the GCP list.
        bps_logger.debug("Reading GCP list")
        self.__read_gcp_list()

        # Finally, read the measurements.
        if not annotation_only:
            bps_logger.debug("Reading measurements")
            self.__read_measurement_files()
            self.__warn_if_data_annotation_inconsistent()

        return self.product

    def read_processing_parameters(self) -> tuple[dict, dict]:
        """Read the object's processing parameters."""
        parameters_dict_primary = self.__read_processing_parameters(True)
        parameters_dict_coregistered = self.__read_processing_parameters(False)
        return parameters_dict_primary, parameters_dict_coregistered

    def read_lut_annotation(
        self,
        *,
        fit_to_l1c_raster: bool = False,
    ) -> tuple[
        dict[str, npt.NDArray],
        tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]],
        tuple[None, None],
    ]:
        """
        Read the annotation as look-up tables.

        Parameters
        ----------
        fit_to_l1c_raster: bool = False
            If true, return the LUTs upsampled onto the raster of the L1c product.

        Returns
        ------
        dict[str, npt.NDArray]
            The LUT data, possibly upsampled to the L1c raster. To upsample the LUTs,
            a linear interpolator is used.

        tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]] [UTC], [s]
            The absolute azimuth and range axes of the LUTs (irrespectively
            of as to whether the LUTs were upsampled or not).

        tuple[None, None]
            A pair of None is retuned to allow backward compatibility of the
            reader. !! SOON TO BE DEPRECATED !!

        """
        # Read LUT annotation file.
        lut_annotation_path = Path(self.product_structure.lut_annotation2_file)
        lut_annotation = Dataset(lut_annotation_path, mode="r")

        # Read the no-data value.
        lut_nodata_value = self.__read_lut_attribute(
            dataset=lut_annotation,
            attribute="noDataValue",
            dtype=np.float64,
        )

        # Read the time axis.
        lut_axes_main = (
            self.__read_lut_dimension(
                dataset=lut_annotation,
                dimension="relativeAzimuthTime",
                warn_on_missing=True,
                dtype=np.float64,
            )
            + PreciseDateTime.from_utc_string(
                self.__read_lut_attribute(
                    dataset=lut_annotation,
                    attribute="referenceAzimuthTime",
                    dtype=str,
                )
            ),
            self.__read_lut_dimension(
                dataset=lut_annotation,
                dimension="slantRangeTime",
                warn_on_missing=True,
                dtype=np.float64,
            ),
        )

        if fit_to_l1c_raster:
            l1c_raster_info = self.read(annotation_only=True).raster_info_list[0]
            l1c_axes = (
                np.arange(l1c_raster_info.lines) * l1c_raster_info.lines_step,
                np.arange(l1c_raster_info.samples) * l1c_raster_info.samples_step,
            )
            lut_axes = (
                (lut_axes_main[0] - lut_axes_main[0][0]).astype(np.float64),
                (lut_axes_main[1] - lut_axes_main[1][0]).astype(np.float64),
            )

        # Read the LUTs.
        lut_dict = {}
        for lut_group_name, lut_group in lut_annotation.groups.items():
            for lut_layer_name in lut_group.variables:
                lut_dict[lut_layer_name] = self.__read_lut_group_variable(
                    dataset=lut_annotation,
                    group=lut_group_name,
                    variable=lut_layer_name,
                    lut_nodata_value=lut_nodata_value,
                    dtype=np.float64,
                )
                if fit_to_l1c_raster:
                    if lut_layer_name == "skpCalibrationPhaseScreen":
                        lut_dict[lut_layer_name] = np.arctan2(
                            sp.interpolate.RectBivariateSpline(
                                lut_axes[0],
                                lut_axes[1],
                                np.sin(lut_dict[lut_layer_name]),
                                kx=1,
                                ky=1,
                                s=0,
                            )(l1c_axes[0], l1c_axes[1]),
                            sp.interpolate.RectBivariateSpline(
                                lut_axes[0],
                                lut_axes[1],
                                np.cos(lut_dict[lut_layer_name]),
                                kx=1,
                                ky=1,
                                s=0,
                            )(l1c_axes[0], l1c_axes[1]),
                        )
                    else:
                        lut_dict[lut_layer_name] = sp.interpolate.RectBivariateSpline(
                            lut_axes[0],
                            lut_axes[1],
                            lut_dict[lut_layer_name],
                            kx=1,
                            ky=1,
                            s=0,
                        )(l1c_axes[0], l1c_axes[1])

        if any(lut_layer_name not in LUT_LAYERS for lut_layer_name in lut_dict):
            raise InvalidBIOMASSStackProductError(
                "{:s} contains unexpected LUTs ({})".format(
                    self.product_path.name,
                    [ln for ln in lut_dict if ln not in LUT_LAYERS],
                )
            )

        return (
            lut_dict,
            lut_axes_main,
            (None, None),  # See docstring. Soon to be deprecated.
        )

    def __read_mph_file(self):
        """Read the MPH file."""
        mph_root = ET.parse(self.product_structure.mph_file).getroot()
        self.product.stack_id = mph_root.find(
            "om:procedure/eop:EarthObservationEquipment/eop:acquisitionParameters/bio:Acquisition/bio:stackID",
            MPH_NAMESPACES,
        ).text

    def __read_gcp_list(self):
        """Read the GCP list."""
        abs_file_path = str(self.product_structure.measurement_files["abs"])
        geotiff = read_geotiff(abs_file_path, skip_data=True)
        self.product.gcp_list = geotiff.gcp_list

    def __read_measurement_files(self):
        """Read the measurement files (abs and phase)."""
        # Read measurement file (abs part).
        abs_file_path = str(self.product_structure.measurement_files["abs"])
        geotiff = read_geotiff(abs_file_path)

        self.product.data_list = geotiff.data_list
        if len(geotiff.data_list) == 0:
            bps_logger.warning("Provided L1c product has no measurement/data.")
            return

        # Initialize populating the no-data mask.
        nodata_mask = np.full(geotiff.data_list[0].shape, False)
        for data, nodata_val in zip(geotiff.data_list, geotiff.nodata_values):
            nodata_mask |= data == nodata_val

        # Read measurement file (phase part).
        phase_file_path = str(self.product_structure.measurement_files["phase"])
        geotiff = read_geotiff(phase_file_path)
        assert len(geotiff.data_list) == len(self.product.data_list) == self.product.channels, (
            "invalid number of channels in geotiff"
        )

        # Incorporate the phase part into the data and update the no-data mask.
        for i, (data, nodata_val) in enumerate(zip(geotiff.data_list, geotiff.nodata_values)):
            self.product.data_list[i] = np.exp(1j * data) * self.product.data_list[i]
            nodata_mask |= data == nodata_val

        # Set the selected no-data value where needed.
        for data, nodata_val in zip(self.product.data_list, geotiff.nodata_values):
            data[nodata_mask] = _value_or(self.nodata_fill_value, default_value=nodata_val)

    def __read_main_annotation_file(self, is_primary_image: bool = True):
        """Read the main annotation file."""
        # Read main annotation file.
        # If is_primary_immage is True, then read main_annotation1_file,
        # otherwise read main_annotation2_file for coregistered image.
        if is_primary_image:
            main_annotation_path = Path(self.product_structure.main_annotation1_file)
        else:
            main_annotation_path = Path(self.product_structure.main_annotation2_file)
        main_annotation_model: main_annotation_models_l1c.MainAnnotation = parse(
            main_annotation_path.read_text(encoding="utf-8"),
            main_annotation_models_l1c.MainAnnotation,
        )

        # Fill attributes.
        temp_product = SARProduct()
        temp_product.name = self.product_path.name
        temp_product.mission = main_annotation_model.acquisition_information.mission.value
        temp_product.acquisition_mode = "STRIPMAP"
        temp_product.type = "STA"
        temp_product.swath_list = [main_annotation_model.acquisition_information.swath.value]
        temp_product.polarization_list = [
            p.value[0] + "/" + p.value[1]
            for p in main_annotation_model.acquisition_information.polarisation_list.polarisation
        ]
        temp_product.start_time = translate_datetime(main_annotation_model.acquisition_information.start_time)
        temp_product.stop_time = translate_datetime(main_annotation_model.acquisition_information.stop_time)

        footprint = main_annotation_model.sar_image.footprint.value
        if len(footprint.split(" ")) > 1:
            temp_product.footprint = parse_footprint_string(footprint, closed=True)
        else:
            temp_product.footprint = [[0.0, 0.0]] * 5

        temp_product.channels = main_annotation_model.acquisition_information.polarisation_list.count

        if len(temp_product.polarization_list) != temp_product.channels:
            raise InvalidBIOMASSStackProductError(
                "Channels and polarizations mismatch: pols={}, channels={}".format(
                    temp_product.polarization_list,
                    temp_product.channels,
                )
            )

        for channel in range(temp_product.channels):
            # - RasterInfo
            samples = main_annotation_model.sar_image.number_of_samples
            samples_start = main_annotation_model.sar_image.first_sample_slant_range_time.value
            samples_start_unit = "s"
            samples_step = main_annotation_model.sar_image.range_time_interval.value
            samples_step_unit = samples_start_unit

            lines = main_annotation_model.sar_image.number_of_lines
            lines_start = translate_datetime(main_annotation_model.sar_image.first_line_azimuth_time)
            lines_start_unit = "Utc"
            lines_step = main_annotation_model.sar_image.azimuth_time_interval.value
            lines_step_unit = "s"
            cell_type = "FLOAT_COMPLEX"
            metadata_ri = metadata.RasterInfo(
                lines,
                samples,
                celltype=cell_type,
                filename=None,
                header_offset_bytes=0,
                row_prefix_bytes=0,
                byteorder="LITTLEENDIAN",
            )
            metadata_ri.set_lines_axis(lines_start, lines_start_unit, lines_step, lines_step_unit)
            metadata_ri.set_samples_axis(samples_start, samples_start_unit, samples_step, samples_step_unit)
            temp_product.raster_info_list.append(metadata_ri)

            # - BurstInfo
            burst_count = 1
            metadata_bi = metadata.BurstInfo()
            for _ in range(burst_count):
                metadata_bi.add_burst(
                    samples_start,
                    lines_start,
                    lines,
                    burst_center_azimuth_shift_i=None,
                )
            temp_product.burst_info_list.append(metadata_bi)

            # - DataSetInfo
            if channel == 0:
                sensor_name = main_annotation_model.acquisition_information.mission.value
                acquisition_mode = "STRIPMAP"
                projection = main_annotation_model.sar_image.projection.value.upper()
                fc_hz = main_annotation_model.instrument_parameters.radar_carrier_frequency.value
                metadata_di = metadata.DataSetInfo(acquisition_mode, fc_hz)
                metadata_di.sensor_name = sensor_name
                metadata_di.description = ""
                metadata_di.sense_date = translate_datetime(main_annotation_model.sar_image.first_line_azimuth_time)
                if temp_product.type == "SLC":
                    metadata_di.image_type = "AZIMUTH FOCUSED RANGE COMPENSATED"
                elif temp_product.type == "GRD":
                    metadata_di.image_type = "MULTILOOK"
                metadata_di.projection = projection
                metadata_di.acquisition_station = ""
                metadata_di.processing_center = ""
                metadata_di.processing_date = PreciseDateTime.now()
                metadata_di.processing_software = ""
                metadata_di.side_looking = "LEFT"
                temp_product.dataset_info.append(metadata_di)

            # - SwathInfo
            swath = main_annotation_model.acquisition_information.swath.value
            polarization = main_annotation_model.acquisition_information.polarisation_list.polarisation[channel].value
            polarization = polarization[0] + "/" + polarization[1]
            acquisition_prf = main_annotation_model.instrument_parameters.prf_list.prf[0].value.value
            metadata_si = metadata.SwathInfo(swath, polarization, acquisition_prf)
            metadata_si.rank = main_annotation_model.instrument_parameters.rank
            metadata_si.range_delay_bias = 0
            metadata_si.acquisition_start_time = translate_datetime(
                main_annotation_model.acquisition_information.start_time
            )
            metadata_si.azimuth_steering_rate_reference_time = 0
            metadata_si.azimuth_steering_rate_pol = (0, 0, 0)
            metadata_si.echoes_per_burst = main_annotation_model.sar_image.number_of_lines
            metadata_si.rx_gain = 1
            temp_product.swath_info_list.append(metadata_si)

            # - SamplingConstants
            frg_hz = 1 / main_annotation_model.sar_image.range_time_interval.value
            brg_hz = main_annotation_model.processing_parameters.range_processing_parameters.look_bandwidth.value
            faz_hz = 1 / main_annotation_model.sar_image.azimuth_time_interval.value
            baz_hz = main_annotation_model.processing_parameters.azimuth_processing_parameters.look_bandwidth.value
            metadata_sc = metadata.SamplingConstants(frg_hz, brg_hz, faz_hz, baz_hz)
            temp_product.sampling_constants_list.append(metadata_sc)

            # - AcquisitionTimeline
            missing_lines_number = 0
            missing_lines_azimuth_times = None
            swp_list = main_annotation_model.instrument_parameters.swp_list
            swp_changes_number = swp_list.count
            swp_changes_azimuth_times = [
                translate_datetime(swp.azimuth_time)
                - translate_datetime(main_annotation_model.sar_image.first_line_azimuth_time)
                for swp in swp_list.swp
            ]
            swp_changes_values = [float(swp.value.value) for swp in swp_list.swp]
            noise_packets_number = 0
            noise_packets_azimuth_times = None
            internal_calibration_number = 0
            internal_calibration_azimuth_times = None
            swl_list = main_annotation_model.instrument_parameters.swl_list
            swl_changes_number = swl_list.count
            swl_changes_azimuth_times = [
                translate_datetime(swl.azimuth_time)
                - translate_datetime(main_annotation_model.sar_image.first_line_azimuth_time)
                for swl in swl_list.swl
            ]
            swl_changes_values = [float(swl.value.value) for swl in swl_list.swl]
            metadata_at = metadata.AcquisitionTimeLine(
                missing_lines_number,
                missing_lines_azimuth_times,
                swp_changes_number,
                swp_changes_azimuth_times,
                swp_changes_values,
                noise_packets_number,
                noise_packets_azimuth_times,
                internal_calibration_number,
                internal_calibration_azimuth_times,
                swl_changes_number,
                swl_changes_azimuth_times,
                swl_changes_values,
            )
            temp_product.acquisition_timeline_list.append(metadata_at)

            # - DataStatistics
            i_num_samples = (
                main_annotation_model.sar_image.number_of_samples * main_annotation_model.sar_image.number_of_lines
            )
            i_max_i = 0
            i_max_q = 0
            i_min_i = 0
            i_min_q = 0
            i_sum_i = 0
            i_sum_q = 0
            i_sum_2_i = 0
            i_sum_2_q = 0
            i_std_dev_i = 0
            i_std_dev_q = 0
            metadata_ds = metadata.DataStatistics(
                i_num_samples,
                i_max_i,
                i_max_q,
                i_min_i,
                i_min_q,
                i_sum_i,
                i_sum_q,
                i_sum_2_i,
                i_sum_2_q,
                i_std_dev_i,
                i_std_dev_q,
            )
            temp_product.data_statistics_list.append(metadata_ds)

            # - DopplerCentroidVector
            dcest_count = main_annotation_model.doppler_parameters.dc_estimate_list.count
            dcest_method = main_annotation_model.processing_parameters.dc_method.value
            metadata_dcl = []
            for dc in range(dcest_count):
                dcest = main_annotation_model.doppler_parameters.dc_estimate_list.dc_estimate[dc]
                ref_az = translate_datetime(dcest.azimuth_time)
                ref_rg = dcest.t0.value
                if dcest_method == common_types.DcMethodType.GEOMETRY.value:
                    coefficients = dcest.geometry_dcpolynomial.value.split(" ")
                elif (
                    dcest_method == common_types.DcMethodType.COMBINED.value
                    or dcest_method == common_types.DcMethodType.FIXED.value
                ):
                    coefficients = dcest.combined_dcpolynomial.value.split(" ")
                coefficients = [float(i) for i in coefficients]
                if len(coefficients) < 7:
                    coefficients = coefficients[0:2] + [0.0, 0.0] + coefficients[2:]
                metadata_dc = metadata.DopplerCentroid(ref_az, ref_rg, coefficients)
                metadata_dcl.append(metadata_dc)
            metadata_dcv = metadata.DopplerCentroidVector(metadata_dcl)
            temp_product.dc_vector_list.append(metadata_dcv)

            # - DopplerRateVector
            fmest_count = main_annotation_model.doppler_parameters.fm_rate_estimate_list.count
            metadata_fml = []
            for fm in range(fmest_count):
                fmest = main_annotation_model.doppler_parameters.fm_rate_estimate_list.fm_rate_estimate[fm]
                ref_az = translate_datetime(fmest.azimuth_time)
                ref_rg = fmest.t0.value
                coefficients = fmest.polynomial.value.split(" ")
                coefficients = [float(i) for i in coefficients]
                if len(coefficients) < 7:
                    coefficients = coefficients[0:2] + [0.0, 0.0] + coefficients[2:]
                metadata_fm = metadata.DopplerRate(ref_az, ref_rg, coefficients)
                metadata_fml.append(metadata_fm)
            metadata_fmv = metadata.DopplerRateVector(metadata_fml)
            temp_product.dr_vector_list.append(metadata_fmv)

            # - Slant2Ground
            # - Ground2Slant
            cc_count = main_annotation_model.sar_image.range_coordinate_conversion.count
            metadata_fml = []
            metadata_sgl = []
            metadata_gsl = []
            for cc in range(cc_count):
                coordinate_conversion = (
                    main_annotation_model.sar_image.range_coordinate_conversion.coordinate_conversion[cc]
                )
                ref_az = translate_datetime(coordinate_conversion.azimuth_time)
                ref_grrg = coordinate_conversion.gr0.value
                grcoefficients = coordinate_conversion.ground_to_slant_coefficients.value.split(" ")
                grcoefficients = [float(i) for i in grcoefficients]
                grcoefficients = grcoefficients[0:2] + [0.0, 0.0] + grcoefficients[2:]
                metadata_gs = metadata.GroundToSlant(ref_az, ref_grrg, grcoefficients)
                metadata_gsl.append(metadata_gs)
                ref_slrg = coordinate_conversion.t0.value
                srcoefficients = coordinate_conversion.slant_to_ground_coefficients.value.split(" ")
                if srcoefficients[0] == "":
                    srcoefficients = [0.0] * len(grcoefficients)
                else:
                    srcoefficients = [float(i) for i in srcoefficients] / (LIGHT_SPEED / 2)
                    srcoefficients = srcoefficients[0:2] + [0.0, 0.0] + srcoefficients[2:]
                metadata_sg = metadata.SlantToGround(ref_az, ref_slrg, srcoefficients)
                metadata_sgl.append(metadata_sg)
            metadata_sgv = metadata.SlantToGroundVector(metadata_sgl)
            metadata_gsv = metadata.GroundToSlantVector(metadata_gsl)
            temp_product.slant_to_ground_list.append(metadata_sgv)
            temp_product.ground_to_slant_list.append(metadata_gsv)

            # - Pulse
            tx_pulse = main_annotation_model.instrument_parameters.tx_pulse_list.tx_pulse[0]
            pulse_length = tx_pulse.tx_pulse_length.value
            pulse_bandwidth = (
                main_annotation_model.processing_parameters.range_processing_parameters.processing_bandwidth.value
            )
            pulse_sampling_rate = 1 / main_annotation_model.sar_image.range_time_interval.value
            pulse_energy = 1.0
            pulse_start_frequency = tx_pulse.tx_pulse_start_frequency.value
            pulse_start_phase = tx_pulse.tx_pulse_start_phase.value
            pulse_direction = "UP"
            metadata_p = metadata.Pulse(
                pulse_length,
                pulse_bandwidth,
                pulse_sampling_rate,
                pulse_energy,
                pulse_start_frequency,
                pulse_start_phase,
                pulse_direction,
            )
            temp_product.pulse_list.append(metadata_p)

        if is_primary_image:
            self.product_primary = temp_product
        else:
            self.product = temp_product
            self.stack_processing_parameters = BIOMASSStackProcessingParameters.from_l1c_main_annotation(
                main_annotation_model
            )
            self.stack_coregistration_parameters = BIOMASSStackCoregistrationParameters.from_l1c_main_annotation(
                main_annotation_model
            )
            self.stack_in_sarparameters = BIOMASSStackInSARParameters.from_l1c_main_annotation(main_annotation_model)
            self.stack_quality = BIOMASSStackQuality.from_l1c_main_annotation(main_annotation_model)

        self.mission_phase_id = main_annotation_model.acquisition_information.mission_phase_id.value
        self.datatake_id = main_annotation_model.acquisition_information.data_take_id
        self.orbit_number = main_annotation_model.acquisition_information.absolute_orbit_number
        self.frame_status = main_annotation_model.acquisition_information.product_composition.value.upper()
        self.platform_heading = main_annotation_model.acquisition_information.platform_heading.value
        self.first_sample_sr_time = main_annotation_model.sar_image.first_sample_slant_range_time.value
        self.first_line_az_time = translate_datetime(main_annotation_model.sar_image.first_line_azimuth_time)
        if self.first_line_az_time < temp_product.start_time:
            raise InvalidBIOMASSStackProductError(
                "First azm time and start time mismatch: first azm={}, start time={}".format(
                    self.first_line_az_time, temp_product.start_time
                )
            )

        self.rg_time_interval = main_annotation_model.sar_image.range_time_interval.value
        self.az_time_interval = main_annotation_model.sar_image.azimuth_time_interval.value
        self.number_of_samples = main_annotation_model.sar_image.number_of_samples
        self.number_of_lines = main_annotation_model.sar_image.number_of_lines
        self.product_pixel_type = main_annotation_model.sar_image.pixel_type
        self.product_nodata_value = main_annotation_model.sar_image.no_data_value

    def __read_processing_parameters(self, is_primary_image: bool = True):
        """Read the processing parameters."""
        # Read main annotation file.
        # If is_primary_immage is True, then read main_annotation1_file,
        # otherwise read main_annotation2_file for coregistered image.
        if is_primary_image:
            main_annotation_path = Path(self.product_structure.main_annotation1_file)
        else:
            main_annotation_path = Path(self.product_structure.main_annotation2_file)
        main_annotation_model: main_annotation_models_l1c.MainAnnotation = parse(
            main_annotation_path.read_text(encoding="utf-8"),
            main_annotation_models_l1c.MainAnnotation,
        )

        processing_parameters = translate_common_annotation_l1.translate_processing_parameters(
            main_annotation_model.processing_parameters
        )
        rg_params = processing_parameters.range_processing_parameters
        az_params = processing_parameters.azimuth_processing_parameters

        # Read processing parameters.
        parameters_dict = {}

        parameters_dict["range_window_type"] = rg_params.window_type.value
        parameters_dict["range_window_coefficient"] = rg_params.window_coefficient
        parameters_dict["range_window_bandwidth"] = rg_params.look_bandwidth

        parameters_dict["azimuth_window_type"] = az_params.window_type.value
        parameters_dict["azimuth_window_coefficient"] = az_params.window_coefficient
        parameters_dict["azimuth_window_bandwidth"] = az_params.look_bandwidth

        return parameters_dict

    def __read_lut_attribute(
        self,
        *,
        dataset: Dataset,
        attribute: str,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray | None:
        """Read a LUT attribute."""
        return dtype(dataset.__dict__[attribute])

    def __read_lut_dimension(
        self,
        *,
        dataset: Dataset,
        dimension: str,
        warn_on_missing: bool,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[float] | None:
        """Read a LUT dimension. Optionally, warn on missing."""
        try:
            return np.array(dataset.variables[dimension][:], dtype=dtype)
        except Exception as e:
            if warn_on_missing:
                bps_logger.warning("Failed to read dimension %s: %s", dimension, str(e))
            return None

    def __read_lut_group_variable(
        self,
        *,
        dataset: Dataset,
        group: str,
        variable: str,
        lut_nodata_value: float,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray | None:
        """Read a LUT. Warn the user if a required attribute is missing."""
        try:
            values = np.asarray(dataset.groups[group][variable][:])
            values[values == lut_nodata_value] = _value_or(self.nodata_fill_value, default_value=lut_nodata_value)
            return values.astype(dtype)
        except Exception as e:
            if variable not in dataset.groups[group].__dict__.keys() and variable in REQUIRED_LUT_LAYERS:
                bps_logger.warning("%s is missing from %s/%s: %s", variable, self.product_path.name, group, str(e))
            return None

    def __read_orbit_file(self, is_primary_image: bool = True):
        """Read the orbit file."""
        # If is_primary_immage is True, then read orbit1_file,
        # otherwise read orbit2_file for coregistered image.
        if is_primary_image:
            earth_observation_file_path = Path(self.product_structure.orbit1_file)
        else:
            earth_observation_file_path = Path(self.product_structure.orbit2_file)
        earth_observation_file_model: aux_orb_models.EarthObservationFile = parse(
            earth_observation_file_path.read_text(encoding="utf-8"),
            aux_orb_models.EarthObservationFile,
        )

        # Fill attributes.
        # - StateVectors
        osvs = earth_observation_file_model.data_block.list_of_osvs
        sv_count = osvs.count
        position_sv = np.zeros((sv_count, 3))
        velocity_sv = np.zeros((sv_count, 3))
        for sv in range(sv_count):
            osv = osvs.osv[sv]
            position_sv[sv][0] = osv.x.value
            position_sv[sv][1] = osv.y.value
            position_sv[sv][2] = osv.z.value
            velocity_sv[sv][0] = osv.vx.value
            velocity_sv[sv][1] = osv.vy.value
            velocity_sv[sv][2] = osv.vz.value
        reference_time = PreciseDateTime().fromisoformat(osvs.osv[0].utc[4:])
        delta_time = PreciseDateTime().fromisoformat(osvs.osv[1].utc[4:]) - reference_time
        metadata_sv = metadata.StateVectors(position_sv, velocity_sv, reference_time, delta_time)

        if is_primary_image:
            self.product_primary.orbit_number = 1
            if velocity_sv[0][2] > 0:
                self.product_primary.orbit_direction = "ASCENDING"
            else:
                self.product_primary.orbit_direction = "DESCENDING"

            self.product_primary.general_sar_orbit.append(metadata_sv)
        else:
            self.product.orbit_number = 1
            if velocity_sv[0][2] > 0:
                self.product.orbit_direction = "ASCENDING"
            else:
                self.product.orbit_direction = "DESCENDING"

            self.product.general_sar_orbit.append(metadata_sv)

    def __read_attitude_file(self, is_primary_image: bool = True):
        """Read the attitude file."""
        # If is_primary_image is True, then read attitude1_file,
        # otherwise read attitude2_file for coregistered image
        earth_observation_file_path = Path(
            self.product_structure.attitude1_file if is_primary_image else self.product_structure.attitude2_file
        )

        attitude_model = read_attitude_file(earth_observation_file_path)

        state_vectors = (
            self.product_primary.general_sar_orbit[0] if is_primary_image else self.product.general_sar_orbit[0]
        )
        attitude_info = translate_attitude_file_to_attitude_info(attitude_model, state_vectors)

        if is_primary_image:
            self.product_primary.attitude_info.append(attitude_info)
        else:
            self.product.attitude_info.append(attitude_info)

    def __warn_if_data_annotation_inconsistent(self):
        """Warn if data and annotations are inconsistent."""
        for pol, raster_info, data in zip(
            self.product.polarization_list,
            self.product.raster_info_list,
            self.product.data_list,
        ):
            if data.shape != (raster_info.lines, raster_info.samples):
                raise InvalidBIOMASSStackProductError(
                    "invalid raster shape for pol={}: data={}, raster={}".format(
                        pol,
                        data.shape,
                        (raster_info.lines, raster_info.samples),
                    )
                )


def _value_or(arg: Any | None, *, default_value: Any) -> Any:
    """Return `x` if `x` is not None, `default_value` otherwise."""
    return arg if arg is not None else default_value
