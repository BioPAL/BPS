# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 post processor run
-------------------------
"""

from pathlib import Path

from arepytools.io import open_product_folder
from bps.common.decorators import log_elapsed_time
from bps.l1_processor.post.interface import (
    export_in_bps_format,
    fill_bps_transcoder_configuration,
)
from bps.l1_processor.post.rfi_masks_statistics import compute_rfi_masks_statistics
from bps.l1_processor.processor_interface.aux_pp1 import AuxProcessingParametersL1, DopplerEstimationConf
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1JobOrder,
    L1RXOnlyOutputProducts,
    L1StripmapOutputProducts,
)
from bps.l1_processor.settings.intermediate_names import IntermediateProductID
from bps.l1_processor.settings.l1_intermediates import L1CoreProcessorOutputProducts
from bps.transcoder.io.iono_cal_report import read_iono_cal_report
from bps.transcoder.io.preprocessor_report import parse_annotations
from bps.transcoder.sarproduct.l1_annotations import DCAnnotations


def generate_export_products_list(
    l1_core_processor_outputs: L1CoreProcessorOutputProducts,
    requested_products: L1StripmapOutputProducts | L1RXOnlyOutputProducts,
) -> list[tuple[Path, bool]]:
    """Generate list of products to export"""

    if l1_core_processor_outputs.main_slc_id is None:
        raise RuntimeError("Main SLC intermediate ID not defined")

    slc_product = l1_core_processor_outputs.output_products.get(l1_core_processor_outputs.main_slc_id)
    assert slc_product is not None

    grd_product = l1_core_processor_outputs.output_products.get(IntermediateProductID.GRD)

    if isinstance(requested_products, L1RXOnlyOutputProducts):
        monitoring_required = False
        assert slc_product

        return [
            (
                slc_product.path,
                monitoring_required,
            )
        ]

    assert isinstance(requested_products, L1StripmapOutputProducts)

    export_products: list[tuple[Path, bool]] = []
    if requested_products.scs_standard_required:
        assert slc_product

        export_products.append(
            (
                slc_product.path,
                requested_products.scs_monitoring_required,
            )
        )

    if requested_products.dgm_standard_required:
        monitoring_required = False
        assert grd_product

        export_products.append((grd_product.path, monitoring_required))

    return export_products


@log_elapsed_time("L1PostProcessor")
def run_l1_post_processing(
    l1_core_processor_outputs: L1CoreProcessorOutputProducts,
    job_order: L1JobOrder,
    aux_pp1: AuxProcessingParametersL1,
):
    """Execute L1 post processing: export of the output products in the bps format"""

    export_products = generate_export_products_list(l1_core_processor_outputs, job_order.io_products.output)

    dc_fallback_activated = False
    dc_annotations = None
    if aux_pp1.doppler_estimation.method == DopplerEstimationConf.Method.COMBINED:
        assert l1_core_processor_outputs.geometric_doppler.exists()
        dc_grid_product = l1_core_processor_outputs.output_products.get(
            IntermediateProductID.DOPPLER_CENTROID_ESTIMATOR_GRID
        )
        if dc_grid_product is not None and dc_grid_product.path.exists():
            if len(open_product_folder(dc_grid_product.path).get_channels_list()) > 0:
                dc_annotations = DCAnnotations.from_products(
                    geometric_dc_product=l1_core_processor_outputs.geometric_doppler,
                    combined_dc_grid=dc_grid_product.path,
                )
            else:
                aux_pp1.doppler_estimation.method = DopplerEstimationConf.Method.GEOMETRY
                dc_fallback_activated = True

    bps_transcoder_configuration = fill_bps_transcoder_configuration(
        job_order, aux_pp1, l1_core_processor_outputs.extracted_raw_annotation
    )

    intermediate_files = {id.name: info.path for id, info in l1_core_processor_outputs.output_products.items()}

    auxiliary_files = job_order.auxiliary_files.get_all_aux_products_paths()

    l1_pre_proc_report = parse_annotations(l1_core_processor_outputs.preprocessor_report.read_text())

    rfi_masks_statistics = compute_rfi_masks_statistics(
        rfi_frequency_mask=intermediate_files.get(IntermediateProductID.RFI_FREQ_MASK.name),
        rfi_time_mask=intermediate_files.get(IntermediateProductID.RFI_TIME_MASK.name),
    )

    iono_cal_report_file = intermediate_files.get(IntermediateProductID.IONO_CAL_REPORT.name, None)
    iono_cal_report = read_iono_cal_report(iono_cal_report_file) if iono_cal_report_file else None

    for product, add_monitoring in export_products:
        export_in_bps_format(
            input_product_path=product,
            source_product_path=job_order.io_products.input.input_standard,
            source_monitoring_product_path=job_order.io_products.input.input_monitoring,
            auxiliary_files=auxiliary_files,
            intermediate_products_dict=intermediate_files,
            output_folder=job_order.io_products.output.output_directory,
            configuration=bps_transcoder_configuration,
            l1_pre_proc_report=l1_pre_proc_report,
            rfi_masks_statistics=rfi_masks_statistics,
            dc_annotations=dc_annotations,
            dc_fallback_activated=dc_fallback_activated,
            add_monitoring_product=add_monitoring,
            gdal_num_threads=job_order.device_resources.num_threads,
            iono_cal_report=iono_cal_report if iono_cal_report else None,
        )
