# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 core processor run
-------------------------
"""

from __future__ import annotations

from pathlib import Path

from arepyextras.runner import Environment
from arepytools.io import (
    iter_channels,
    open_product_folder,
    read_metadata,
    write_metadata,
)
from bps.common import bps_logger
from bps.common.decorators import log_elapsed_time
from bps.common.runner_helper import run_application
from bps.l1_core_processor.interface import (
    L1CoreProcessorInterface,
    write_l1_coreproc_input_file,
    write_l1_coreproc_options_file,
    write_l1_coreproc_parameters_file,
)
from bps.l1_core_processor.processing_options import BPSL1CoreProcessorStep
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsProduct
from bps.l1_pre_processor.aux_ins.swst_bias import retrieve_swst_bias
from bps.l1_processor.core.channel_imbalance import parse_channel_imbalance
from bps.l1_processor.core.doppler_centroid_poly import write_doppler_centroid_poly_file
from bps.l1_processor.core.filter_bank.filter_bank import write_resampling_filter
from bps.l1_processor.core.input_file_utils import fill_bps_l1_core_processor_input_file
from bps.l1_processor.core.interface import L1CoreProcessorInterfaceFiles
from bps.l1_processor.core.processing_options_utils import (
    fill_bps_l1_core_processor_processing_options,
)
from bps.l1_processor.core.processing_parameters_utils import (
    fill_sarfoc_processing_parameters,
)
from bps.l1_processor.core.wgs84_fallback_utils import (
    write_slant_dem_product_on_ellipsoid,
)
from bps.l1_processor.io.l0_mph_utils import L0MainProductHeader
from bps.l1_processor.processor_interface.aux_pp1 import (
    AuxProcessingParametersL1,
    DopplerEstimationConf,
    GeneralConf,
)
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder
from bps.l1_processor.settings.intermediate_names import IntermediateProductID
from bps.l1_processor.settings.l1_binaries import BPS_L1COREPROC_EXE_NAME
from bps.l1_processor.settings.l1_intermediates import (
    L1CoreProcessorOutputProducts,
    L1PreProcessorOutputProducts,
)


def update_grd_metadata(grd_product: Path, slc_product: Path) -> None:
    """Update missing metadata information in the GRD product"""
    grd_pf = open_product_folder(grd_product)
    slc_pf = open_product_folder(slc_product)
    for (grd_ch, grd_metadata), (_, slc_metadata) in zip(iter_channels(grd_pf), iter_channels(slc_pf)):
        grd_dc = grd_metadata.get_doppler_centroid()
        slc_dc = slc_metadata.get_doppler_centroid()
        for poly_idx in range(slc_dc.get_number_of_poly()):
            grd_dc.add_poly(slc_dc.get_poly(poly_idx))

        grd_dr = grd_metadata.get_doppler_rate()
        slc_dr = slc_metadata.get_doppler_rate()
        for poly_idx in range(slc_dr.get_number_of_poly()):
            grd_dr.add_poly(slc_dr.get_poly(poly_idx))

        grd_metadata.get_swath_info().acquisition_prf = slc_metadata.get_swath_info().acquisition_prf

        write_metadata(grd_metadata, grd_pf.get_channel_metadata(grd_ch))


def write_slant_dem_product_on_ellipsoid_from_rgc(
    core_outputs: L1CoreProcessorOutputProducts, aux_pp1: AuxProcessingParametersL1
):
    """Write slant dem product based on RGC raster info"""
    rgc_product_info = core_outputs.output_products.get(IntermediateProductID.RGC_DC_FR_ESTIMATOR)
    if rgc_product_info is None:
        raise RuntimeError("Missing RGC intermediate product")

    rgc_pf = open_product_folder(rgc_product_info.path)
    reference_channel = rgc_pf.get_channels_list()[0]
    reference_metadata_file = rgc_pf.get_channel_metadata(reference_channel)
    reference_metadata = read_metadata(reference_metadata_file)

    slant_dem_path = core_outputs.output_products.get(IntermediateProductID.SLANT_DEM)
    assert slant_dem_path is not None

    write_slant_dem_product_on_ellipsoid(
        reference_metadata,
        slant_dem_path.path,
        range_decimation_factor=aux_pp1.l1_product_export.lut_range_decimation_factor.dem_based_quantity,
        azimuth_decimation_factor=aux_pp1.l1_product_export.lut_azimuth_decimation_factor.dem_based_quantity,
    )


@log_elapsed_time("L1CoreProcessor")
def run_l1_core_processing(
    job_order: L1JobOrder,
    l1_pre_processor_outputs: L1PreProcessorOutputProducts,
    l1_core_processor_outputs: L1CoreProcessorOutputProducts,
    core_processing_steps: dict[BPSL1CoreProcessorStep, bool],
    chirp_replica_product: Path | None,
    per_line_correction_factors_product: Path | None,
    channel_imbalance_file: Path | None,
    geomagnetic_field_folder: Path | None,
    tec_map_field_folder: Path | None,
    ionospheric_height_model_file: Path | None,
    input_faraday_rotation_product: Path | None,
    input_phase_screen_product: Path | None,
    core_processor_files: L1CoreProcessorInterfaceFiles,
    bps_conf_file: Path,
    bps_logger_file: Path,
    aux_pp1: AuxProcessingParametersL1,
    dem_path: Path | None,
    env: Environment,
):
    """Execute L1 core processing"""

    intermediate_files = {id: info.path for id, info in l1_core_processor_outputs.output_products.items()}

    # channel imbalance parsing
    channel_imbalance = (
        parse_channel_imbalance(channel_imbalance_file.read_text()) if channel_imbalance_file is not None else None
    )

    aux_ins_file = AuxInsProduct.from_product(job_order.auxiliary_files.instrument_parameters).instrument_file
    instrument_swst_bias = retrieve_swst_bias(aux_ins_file)

    if aux_pp1.azimuth_compression.filter_type not in ("SINC", "FIR"):
        raise RuntimeError(f"Filter type not supported: {aux_pp1.azimuth_compression.filter_type}")

    pf = open_product_folder(l1_pre_processor_outputs.extracted_raw_product)
    sampling_frequency = 1.0 / read_metadata(pf.get_channel_metadata(1)).get_raster_info().lines_step

    write_resampling_filter(
        l1_core_processor_outputs.resampling_filter,
        aux_pp1.azimuth_compression.filter_type,
        aux_pp1.azimuth_compression.filter_bandwidth,
        sampling_frequency,
        aux_pp1.azimuth_compression.filter_length,
        aux_pp1.azimuth_compression.number_of_filters,
    )

    input_dc_poly_file = None
    if aux_pp1.doppler_estimation.method == DopplerEstimationConf.Method.FIXED:
        mph_swath_info = L0MainProductHeader.from_product(job_order.io_products.input.input_standard)

        write_doppler_centroid_poly_file(
            mph_swath_info,
            aux_pp1.doppler_estimation,
            core_processor_files.dc_poly_file,
        )

        assert core_processor_files.dc_poly_file.exists()

        input_dc_poly_file = core_processor_files.dc_poly_file

    input_tec_map_product = tec_map_field_folder.absolute() if tec_map_field_folder else None

    core_processor_interface = L1CoreProcessorInterface(
        input_file=fill_bps_l1_core_processor_input_file(
            job_order=job_order,
            input_raw_product=l1_pre_processor_outputs.extracted_raw_product.absolute(),
            input_chirp_replica_product=chirp_replica_product,
            input_per_line_correction_factors_product=per_line_correction_factors_product,
            input_processing_dc_poly_file_name=input_dc_poly_file,
            input_d1h_pattern_product=l1_pre_processor_outputs.antenna_patterns.ant_d1_h_product.absolute(),
            input_d1v_pattern_product=l1_pre_processor_outputs.antenna_patterns.ant_d1_v_product.absolute(),
            input_d2h_pattern_product=l1_pre_processor_outputs.antenna_patterns.ant_d2_h_product.absolute(),
            input_d2v_pattern_product=l1_pre_processor_outputs.antenna_patterns.ant_d2_v_product.absolute(),
            input_tx_power_tracking_product=l1_pre_processor_outputs.tx_power_tracking_product.absolute(),
            input_noise_product=l1_pre_processor_outputs.est_noise_product.absolute(),
            input_geomagnetic_field_product=(
                geomagnetic_field_folder.absolute() if geomagnetic_field_folder is not None else None
            ),
            input_tec_map_product=input_tec_map_product,
            input_climatological_model_file=(
                ionospheric_height_model_file.absolute() if ionospheric_height_model_file is not None else None
            ),
            input_faraday_rotation_product=input_faraday_rotation_product,
            input_phase_screen_product=input_phase_screen_product,
            processing_options=core_processor_files.options_file.absolute(),
            processing_parameters=core_processor_files.params_file.absolute(),
            bps_configuration_file=bps_conf_file.absolute(),
            bps_log_file=bps_logger_file.absolute(),
            output_dir=l1_core_processor_outputs.directory.absolute(),
        ),
        options=fill_bps_l1_core_processor_processing_options(
            dem_path=dem_path,
            aux_pp1_conf=aux_pp1,
            steps=core_processing_steps,
            intermediate_files=intermediate_files,
            resampling_filter_product=l1_core_processor_outputs.resampling_filter,
            remove_sarfoc_intermediates=True,
        ),
        params=fill_sarfoc_processing_parameters(
            pp1=aux_pp1, channel_imbalance=channel_imbalance, instrument_swst_bias=instrument_swst_bias
        ),
    )

    write_l1_coreproc_input_file(core_processor_interface.input_file, core_processor_files.input_file)
    write_l1_coreproc_options_file(core_processor_interface.options, core_processor_files.options_file)
    write_l1_coreproc_parameters_file(core_processor_interface.params, core_processor_files.params_file)

    if not aux_pp1.autofocus.autofocus_flag:
        # Standard run
        run_application(
            env,
            BPS_L1COREPROC_EXE_NAME,
            str(core_processor_files.input_file.absolute()),
            1,
        )

    else:
        # pylint: disable-next=import-outside-toplevel
        from bps.l1_processor.core.run_autofocus import run_with_autofocus

        run_with_autofocus(env, core_processor_interface, core_processor_files)

    # Update iGRD metadata
    grd_product = l1_core_processor_outputs.output_products.get(IntermediateProductID.GRD)
    if grd_product is not None and grd_product.path.exists():
        slc_id = l1_core_processor_outputs.main_slc_id
        assert slc_id is not None
        slc_product = l1_core_processor_outputs.output_products.get(slc_id)
        assert slc_product is not None and slc_product.path.exists()
        update_grd_metadata(grd_product.path, slc_product.path)

    # Write a slant dem product based on WGS84 ellipsoid
    if aux_pp1.general.height_model == GeneralConf.EarthModel.ELLIPSOID:
        bps_logger.info("Exporting SlantDEM intermediate product based on Ellipsoid")
        write_slant_dem_product_on_ellipsoid_from_rgc(l1_core_processor_outputs, aux_pp1)

    if aux_pp1.doppler_estimation.method == DopplerEstimationConf.Method.GEOMETRY:
        # Delete unnecessary empty combined dc grid product
        combined_dcgrid_proudct = l1_core_processor_outputs.output_products.get(
            IntermediateProductID.DOPPLER_CENTROID_ESTIMATOR_GRID
        )
        if combined_dcgrid_proudct is not None and combined_dcgrid_proudct.path.exists():
            open_product_folder(combined_dcgrid_proudct.path).delete()
