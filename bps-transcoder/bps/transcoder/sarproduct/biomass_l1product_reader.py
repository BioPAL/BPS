# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import os
from numbers import Number
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import arepytools.io.metadata as metadata
import numpy as np
from arepytools.constants import LIGHT_SPEED
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common
from bps.common.io.parsing import parse
from bps.transcoder.auxiliaryfiles.aux_attitude import read_attitude_file
from bps.transcoder.io import (
    common_annotation_l1,
    main_annotation_l1ab,
    main_annotation_models_l1ab,
    translate_main_annotation_l1ab,
)
from bps.transcoder.orbit.eoorbit import EOOrbit
from bps.transcoder.sarproduct import l1_annotations
from bps.transcoder.sarproduct.biomass_l1product import (
    BIOMASSL1ProcessingParameters,
    BIOMASSL1Product,
    BIOMASSL1ProductConfiguration,
    QualityParameters,
    SARImageParameters,
)
from bps.transcoder.sarproduct.l1.product_content import L1ProductContent
from bps.transcoder.sarproduct.l1_lut_reader import read_lut_file
from bps.transcoder.sarproduct.mph import get_acquisition, get_metadata, get_product_information
from bps.transcoder.sarproduct.navigation_files_utils import (
    translate_attitude_file_to_attitude_info,
)
from bps.transcoder.sarproduct.sarproduct import SARProduct
from bps.transcoder.utils.gdal_utils import read_geotiff
from bps.transcoder.utils.product_name import parse_l1product_name
from osgeo import gdal

gdal.UseExceptions()


def _read_main_annotation_l1ab(
    file: Path,
) -> main_annotation_l1ab.MainAnnotationL1ab:
    """Read main annotation of l1ab products"""
    main_annotation_model: main_annotation_models_l1ab.MainAnnotation = parse(
        file.read_text(encoding="utf-8"),
        main_annotation_models_l1ab.MainAnnotation,
    )

    return translate_main_annotation_l1ab.translate_main_annotation_l1ab(main_annotation_model)


class BIOMASSL1ProductReader:
    def __init__(self, product_path, nodata_fill_value: Number | None = None) -> None:
        self.product_path = Path(product_path)
        self.product = SARProduct()
        self.content = L1ProductContent.from_name(self.product_path.name)
        self.nodata_fill_value = nodata_fill_value
        self.is_monitoring = parse_l1product_name(self.product_path.name).is_monitoring

    def _read_gcp_list(self):
        assert self.content.abs_raster is not None
        abs_file_path = self.product_path.joinpath(self.content.abs_raster)
        geotiff = read_geotiff(abs_file_path, skip_data=True)
        self.product.gcp_list = geotiff.gcp_list

    def _read_measurement_files(self):
        # Read the geotiff of the abs.
        assert self.content.abs_raster is not None
        abs_file_path = self.product_path.joinpath(self.content.abs_raster)
        geotiff = read_geotiff(abs_file_path)

        self.product.data_list = geotiff.data_list
        if len(self.product.data_list) == 0:
            bps_logger.warning("Provided L1a product has no measurement/data.")
            return

        # Populate the no-data masks for the abs part.
        nodata_mask = np.full(geotiff.data_list[0].shape, False)
        for data, nodata_val in zip(geotiff.data_list, geotiff.nodata_values):
            nodata_mask |= data == nodata_val

        if self.product.type == "SLC":
            # Read measurement file (phase part).
            assert self.content.phase_raster is not None
            phase_file_path = self.product_path.joinpath(self.content.phase_raster)
            geotiff = read_geotiff(phase_file_path)
            assert len(geotiff.data_list) == len(self.product.data_list)

            for i, (data, nodata_val) in enumerate(zip(geotiff.data_list, geotiff.nodata_values)):
                self.product.data_list[i] = np.exp(1j * data) * self.product.data_list[i]
                nodata_mask |= data == nodata_val

        # Populate the no data values.
        for data, nodata_val in zip(self.product.data_list, geotiff.nodata_values):
            data[nodata_mask] = self.nodata_fill_value if self.nodata_fill_value is None else nodata_val

    def _read_mph_file(self):
        tree = ElementTree.parse(self.product_path.joinpath(self.content.mph_file))
        root = tree.getroot()

        acquisition = get_acquisition(root)
        product_information = get_product_information(root)
        metadata = get_metadata(root)

        self.product.orbit_number = acquisition.orbit_number
        self.product.orbit_direction = acquisition.orbit_direction
        self.product.track_number = acquisition.track_number
        self.product.slice_number = acquisition.slice_number
        self.product.mission_phase_id = acquisition.mission_phase_id
        self.product.instrument_configuration_id = acquisition.instrument_configuration_id
        self.product.datatake_id = acquisition.datatake_id
        self.product.orbit_drift_flag = acquisition.orbit_drift_flag
        self.product.global_coverage_id = acquisition.global_coverage_id
        self.product.major_cycle_id = acquisition.major_cycle_id
        self.product.repeat_cycle_id = acquisition.repeat_cycle_id

        self.product.baseline_id = product_information.baseline_id
        self.product.doi = metadata.doi

    def _read_main_annotation_file(self):
        """Read annotation file"""
        main_annotation_path = Path(self.product_path.joinpath(self.content.main_annotation))

        annotation = _read_main_annotation_l1ab(main_annotation_path)

        self.product.name = os.path.basename(self.product_path)
        self.product.mission = annotation.acquisition_information.mission.value
        self.product.acquisition_mode = "STRIPMAP"

        self.product.processing_parameters = BIOMASSL1ProcessingParameters.from_l1_annotation(
            annotation.processing_parameters
        )
        self.product.sar_image_parameters = SARImageParameters.from_l1_annotation(annotation.sar_image)
        self.product.quality_parameters = QualityParameters(
            max_isp_gap=annotation.quality.quality_parameters_list[0].max_ispgap,
            raw_mean_expected=annotation.quality.quality_parameters_list[0].raw_mean_expected,
            raw_mean_threshold=annotation.quality.quality_parameters_list[0].raw_mean_threshold,
            raw_std_expected=annotation.quality.quality_parameters_list[0].raw_std_expected,
            raw_std_threshold=annotation.quality.quality_parameters_list[0].raw_std_threshold,
            max_rfi_tm_percentage=annotation.quality.quality_parameters_list[0].rfi_tmfraction,
            max_rfi_fm_percentage=annotation.quality.quality_parameters_list[0].rfi_fmfraction,
            max_drift_amplitude_std_fraction=0.0,
            max_drift_phase_std_fraction=0.0,
            max_drift_amplitude_error=0.0,
            max_drift_phase_error=0.0,
            max_invalid_drift_fraction=annotation.quality.quality_parameters_list[0].max_invalid_drift_fraction,
            dc_rmserror_threshold=annotation.quality.quality_parameters_list[0].dc_rmserror_threshold,
        )

        def _product_type_to_sar(
            product_type: common.ProductType,
        ) -> Literal["SLC", "GRD"]:
            if product_type == common.ProductType.SCS:
                return "SLC"
            if product_type == common.ProductType.DGM:
                return "GRD"
            raise RuntimeError(f"product type '{product_type.name} is not a valida l1ab type")

        self.product.type = _product_type_to_sar(annotation.acquisition_information.product_type)
        self.product.swath_list = [annotation.acquisition_information.swath.value]
        self.product.polarization_list = [
            p.value[0] + "/" + p.value[1] for p in annotation.acquisition_information.polarisation_list
        ]

        default_height_model = common.HeightModelType(
            value=common.HeightModelBaseType.ELLIPSOID,
            version="",
        )
        self.product.height_model = (
            annotation.geometry.height_model if annotation.geometry.height_model_used_flag else default_height_model
        )

        self.product.datum = annotation.sar_image.datum

        self.product.start_time = annotation.acquisition_information.start_time
        self.product.stop_time = annotation.acquisition_information.stop_time

        if len(annotation.sar_image.footprint) > 1:
            if len(annotation.sar_image.footprint) != 8:
                raise RuntimeError(f"Invalid footprint list size: {len(annotation.sar_image.footprint)} != 8")
            self.product.footprint = np.reshape(annotation.sar_image.footprint, (4, 2)).tolist()
        else:
            self.product.footprint = [[0.0, 0.0]] * 4

        self.product.channels = len(annotation.acquisition_information.polarisation_list)

        self.product.overall_product_quality_index = annotation.quality.overall_product_quality_index

        self.product.sensor_mode = annotation.acquisition_information.sensor_mode.value.upper()
        self.product.platform_heading = annotation.acquisition_information.platform_heading
        self.product.frame_number = annotation.acquisition_information.frame
        self.product.frame_status = annotation.acquisition_information.product_composition.value.upper()

        if annotation.sar_image.pixel_type != common.PixelTypeType.VALUE_32_BIT_FLOAT:
            raise RuntimeError(f"Unexpected pixel type: {annotation.sar_image.pixel_type} in product annotation")

        self.product.rg_time_interval = annotation.sar_image.range_time_interval
        self.product.az_time_interval = annotation.sar_image.azimuth_time_interval
        self.product.number_of_samples = annotation.sar_image.number_of_samples
        self.product.number_of_lines = annotation.sar_image.number_of_lines
        self.product.product_pixel_type = common.PixelTypeType.VALUE_32_BIT_FLOAT
        self.product.product_nodata_value = annotation.sar_image.no_data_value
        self.product.first_line_az_time = annotation.sar_image.first_line_azimuth_time
        self.product.first_sample_sr_time = annotation.sar_image.first_sample_slant_range_time

        for channel in range(self.product.channels):
            # - RasterInfo
            if self.product.type == "SLC":
                raster_info = l1_annotations.fill_raster_info_from_sar_image_slc(annotation.sar_image)
            else:
                assert self.product.type == "GRD"
                raster_info = l1_annotations.fill_raster_info_from_sar_image_grd(annotation.sar_image)

            # - BurstInfo
            burst_info = metadata.BurstInfo()
            burst_info.add_burst(
                raster_info.samples_start,
                raster_info.lines_start,
                raster_info.lines,
                burst_center_azimuth_shift_i=None,
            )

            # - DataSetInfo
            if channel == 0:
                sensor_name = annotation.acquisition_information.mission.value
                acquisition_mode = "STRIPMAP"
                projection = annotation.sar_image.projection.value.upper()
                fc_hz = annotation.instrument_parameters.radar_carrier_frequency
                metadata_di = metadata.DataSetInfo(acquisition_mode, fc_hz)
                metadata_di.sensor_name = sensor_name
                metadata_di.description = ""
                metadata_di.sense_date = annotation.sar_image.first_line_azimuth_time
                if self.product.type == "SLC":
                    metadata_di.image_type = "AZIMUTH FOCUSED RANGE COMPENSATED"
                elif self.product.type == "GRD":
                    metadata_di.image_type = "MULTILOOK"
                metadata_di.projection = projection
                metadata_di.acquisition_station = ""
                metadata_di.processing_center = ""
                metadata_di.processing_date = PreciseDateTime.now()
                metadata_di.processing_software = ""
                metadata_di.side_looking = "LEFT"
                # metadata_di.external_calibration_factor = 1.0
                # metadata_di.data_take_id = "1"
                self.product.dataset_info.append(metadata_di)

            # - SwathInfo
            swath = annotation.acquisition_information.swath.value
            polarization = annotation.acquisition_information.polarisation_list[channel].value
            polarization = polarization[0] + "/" + polarization[1]
            acquisition_prf = annotation.instrument_parameters.prf_list[0][1]
            swath_info = metadata.SwathInfo(swath, polarization, acquisition_prf)
            swath_info.rank = annotation.instrument_parameters.rank
            swath_info.range_delay_bias = 0
            swath_info.acquisition_start_time = list(
                annotation.instrument_parameters.first_line_sensing_time_list.values()
            )[0]

            swath_info.azimuth_steering_rate_reference_time = 0
            swath_info.azimuth_steering_rate_pol = (0, 0, 0)
            swath_info.echoes_per_burst = annotation.sar_image.number_of_lines
            swath_info.rx_gain = 1

            # - SamplingConstants
            frg_hz = 1 / annotation.sar_image.range_time_interval
            brg_hz = annotation.processing_parameters.range_processing_parameters.look_bandwidth
            faz_hz = 1 / annotation.sar_image.azimuth_time_interval
            baz_hz = annotation.processing_parameters.azimuth_processing_parameters.look_bandwidth
            sampling_constants = metadata.SamplingConstants(frg_hz, brg_hz, faz_hz, baz_hz)

            # - AcquisitionTimeline
            missing_lines_number = 0
            missing_lines_azimuth_times = None

            swp_changes_number = len(annotation.instrument_parameters.swp_list)
            swp_changes_azimuth_times = [
                time - annotation.sar_image.first_line_azimuth_time
                for time, _ in annotation.instrument_parameters.swp_list
            ]
            swp_changes_values = [value for _, value in annotation.instrument_parameters.swp_list]

            noise_packets_number = 0
            noise_packets_azimuth_times = None

            internal_calibration_number = 0
            internal_calibration_azimuth_times = None

            swl_changes_number = len(annotation.instrument_parameters.swl_list)
            swl_changes_azimuth_times = [
                time - annotation.sar_image.first_line_azimuth_time
                for time, _ in annotation.instrument_parameters.swl_list
            ]
            swl_changes_values = [value for _, value in annotation.instrument_parameters.swl_list]

            acquisition_timeline = metadata.AcquisitionTimeLine(
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

            # - DataStatistics
            data_statistics = metadata.DataStatistics(
                i_num_samples=raster_info.samples * raster_info.lines,
                i_max_i=0,
                i_max_q=0,
                i_min_i=0,
                i_min_q=0,
                i_sum_i=0,
                i_sum_q=0,
                i_sum_2_i=0,
                i_sum_2_q=0,
                i_std_dev_i=0,
                i_std_dev_q=0,
            )

            # - dcValue
            self.product.dc_value = annotation.processing_parameters.dc_value

            # - DopplerCentroidVector
            metadata_dcl = []
            metadata_edcl = []
            for dcest in annotation.doppler_parameters.dc_estimate_list:
                if annotation.processing_parameters.dc_method == common.DcMethodType.GEOMETRY:
                    coefficients = dcest.geometry_dcpolynomial
                elif annotation.processing_parameters.dc_method in (
                    common.DcMethodType.COMBINED,
                    common.DcMethodType.FIXED,
                ):
                    coefficients = dcest.combined_dcpolynomial
                else:
                    raise RuntimeError(f"Unknown dc method: {annotation.processing_parameters.dc_method}")

                if len(coefficients) < 7:
                    coefficients = coefficients[0:2] + [0.0, 0.0] + coefficients[2:]

                eff_coefficients = coefficients
                if annotation.processing_parameters.dc_method == common.DcMethodType.FIXED:
                    eff_coefficients = [self.product.dc_value]

                metadata_dc = metadata.DopplerCentroid(dcest.azimuth_time, dcest.t0, coefficients)
                metadata_edc = metadata.DopplerCentroid(dcest.azimuth_time, dcest.t0, eff_coefficients)
                metadata_dcl.append(metadata_dc)
                metadata_edcl.append(metadata_edc)

            doppler_centroid = metadata.DopplerCentroidVector(metadata_dcl)
            effective_doppler_centroid = metadata.DopplerCentroidVector(metadata_edcl)

            # - DopplerRateVector
            metadata_fml = []
            for fmest in annotation.doppler_parameters.fm_rate_estimate_list:
                coefficients = fmest.polynomial
                if len(coefficients) < 7:
                    coefficients = coefficients[0:2] + [0.0, 0.0] + coefficients[2:]
                    # coefficients = [coefficients[0], coefficients[1], 0, 0, coefficients[2], 0, 0, 0, 0, 0, 0]
                metadata_fm = metadata.DopplerRate(fmest.azimuth_time, fmest.t0, coefficients)
                metadata_fml.append(metadata_fm)
            doppler_rate = metadata.DopplerRateVector(metadata_fml)

            # - Slant2Ground
            # - Ground2Slant
            metadata_sgl = []
            metadata_gsl = []
            for coordinate_conversion in annotation.sar_image.range_coordinate_conversion:
                # Ground2Slant
                grcoefficients = coordinate_conversion.ground_to_slant_coefficients
                grcoefficients = grcoefficients[0:2] + [0.0, 0.0] + grcoefficients[2:]
                metadata_gs = metadata.GroundToSlant(
                    coordinate_conversion.azimuth_time,
                    coordinate_conversion.gr0,
                    grcoefficients,
                )
                metadata_gsl.append(metadata_gs)

                srcoefficients = coordinate_conversion.slant_to_ground_coefficients
                if len(srcoefficients) == 0:
                    srcoefficients = [0.0] * len(grcoefficients)
                else:
                    srcoefficients = [i / (LIGHT_SPEED / 2) for i in srcoefficients]
                    srcoefficients = srcoefficients[0:2] + [0.0, 0.0] + srcoefficients[2:]
                    # srcoefficients = [srcoefficients[0], srcoefficients[1], 0, 0, srcoefficients[2], 0, 0, 0, 0, 0, 0]
                metadata_sg = metadata.SlantToGround(
                    coordinate_conversion.azimuth_time,
                    coordinate_conversion.t0,
                    srcoefficients,
                )
                metadata_sgl.append(metadata_sg)

            slant_to_ground = metadata.SlantToGroundVector(metadata_sgl)
            ground_to_slant = metadata.GroundToSlantVector(metadata_gsl)

            # - Pulse
            tx_pulse = annotation.instrument_parameters.tx_pulse_list[0]
            metadata_p = metadata.Pulse(
                i_pulse_length=tx_pulse.tx_pulse_length,
                i_bandwidth=(annotation.processing_parameters.range_processing_parameters.processing_bandwidth),
                i_pulse_sampling_rate=1 / annotation.sar_image.range_time_interval,
                i_pulse_energy=1.0,
                i_pulse_start_frequency=tx_pulse.tx_pulse_start_frequency,
                i_pulse_start_phase=tx_pulse.tx_pulse_start_phase,
                i_pulse_direction="UP",
            )
            self.product.raster_info_list.append(raster_info)
            self.product.burst_info_list.append(burst_info)
            self.product.swath_info_list.append(swath_info)
            self.product.sampling_constants_list.append(sampling_constants)
            self.product.acquisition_timeline_list.append(acquisition_timeline)
            self.product.data_statistics_list.append(data_statistics)
            self.product.dc_vector_list.append(doppler_centroid)
            self.product.dc_eff_vector_list.append(effective_doppler_centroid)
            self.product.dr_vector_list.append(doppler_rate)
            self.product.slant_to_ground_list.append(slant_to_ground)
            self.product.ground_to_slant_list.append(ground_to_slant)
            self.product.pulse_list.append(metadata_p)

    def read_ionosphere_correction(self) -> common_annotation_l1.IonosphereCorrection:
        """Read ionosphere correction from main annotation file"""
        main_annotation_path = Path(self.product_path.joinpath(self.content.main_annotation))

        annotation = _read_main_annotation_l1ab(main_annotation_path)

        return annotation.ionosphere_correction

    def read_processing_parameters(self) -> dict[str, Any]:
        """Read processing parameters from main annotation file"""
        main_annotation_path = Path(self.product_path.joinpath(self.content.main_annotation))

        annotation = _read_main_annotation_l1ab(main_annotation_path)

        rg_params = annotation.processing_parameters.range_processing_parameters
        az_params = annotation.processing_parameters.azimuth_processing_parameters

        return {
            "range_window_type": rg_params.window_type.value,
            "range_window_coefficient": rg_params.window_coefficient,
            "range_window_bandwidth": rg_params.look_bandwidth,
            "range_window_total_bandwidth": rg_params.total_bandwidth,
            "azimuth_window_type": az_params.window_type.value,
            "azimuth_window_coefficient": az_params.window_coefficient,
            "azimuth_window_bandwidth": az_params.look_bandwidth,
            "azimuth_window_total_bandwidth": az_params.total_bandwidth,
        }

    def read_lut_annotation(self):
        lut_annotation_path = Path(self.product_path.joinpath(self.content.lut))
        return read_lut_file(lut_annotation_path)

    def _read_orbit_file(self):
        """Read orbit file"""
        orbit_path = Path(self.product_path.joinpath(self.content.orbit))

        eo_orbit = EOOrbit(orbit_path)

        metadata_sv = metadata.StateVectors(
            eo_orbit.position_sv,
            eo_orbit.velocity_sv,
            eo_orbit.reference_time,
            eo_orbit.delta_time,
        )

        self.product.orbit_direction = "ASCENDING" if eo_orbit.velocity_sv[0][2] > 0 else "DESCENDING"

        self.product.general_sar_orbit.append(metadata_sv)

    def _read_attitude_file(self):
        attitude_model = read_attitude_file(self.product_path.joinpath(self.content.attitude))

        state_vectors = self.product.general_sar_orbit[0]
        attitude_info = translate_attitude_file_to_attitude_info(attitude_model, state_vectors)

        self.product.attitude_info.append(attitude_info)

    def read(self, *, annotation_only: bool = False):
        self.read_as_sarproduct(annotation_only=annotation_only)
        return BIOMASSL1Product(
            product=self.product,
            is_monitoring=self.is_monitoring,
            source=self.product,
            configuration=BIOMASSL1ProductConfiguration(
                l1a_doi=self.product.doi,
                l1b_doi=self.product.doi,
                frame_id=self.product.frame_number,
                frame_status=self.product.frame_status,
                product_baseline=self.product.baseline_id,
                acquisition_raster_info=None,
                acquisition_timeline=None,
                processing_parameters=self.product.processing_parameters,
                sar_image_parameters=self.product.sar_image_parameters,
                lut_parameters=None,
                quicklook_parameters=None,
                quality_parameters=self.product.quality_parameters,
            ),
        )

    def read_as_sarproduct(self, *, annotation_only: bool = False):
        bps_logger.debug("Reading BIOMASS L1 product..")

        # Read annotation files.
        bps_logger.debug("..annotation files")
        self._read_main_annotation_file()

        # Read orbit and attitude files.
        bps_logger.debug("..orbit and attitude files")
        self._read_orbit_file()
        self._read_attitude_file()

        # Read the manifest file.
        bps_logger.debug("..reading manifest file")
        self._read_mph_file()

        # If monitoring, we are done here.
        if self.is_monitoring:
            bps_logger.debug("..done")
            return self.product

        # Read the GPC list.
        bps_logger.debug("..read GCP list")
        self._read_gcp_list()

        # Finally, read the measurements.
        if not annotation_only:
            bps_logger.debug("..read measurements")
            self._read_measurement_files()

        bps_logger.debug("..done")
        return self.product
