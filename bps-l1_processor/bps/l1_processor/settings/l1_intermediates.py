# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Intermediate products settings
------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from arepytools.io import open_product_folder
from arepytools.io.productfolder_layout import MANIFEST_NAME
from bps.common import bps_logger
from bps.l1_processor.settings import intermediate_names as names


def _remove_additional_files(product: Path):
    rule = f"^{product.name}(_[0-9]{{4}}\\.png)"

    for file in product.iterdir():
        if re.search(rule, file.name) or file.name in [
            names.FOOTPRINT_FILE_NAME,
            MANIFEST_NAME,
        ]:
            file.unlink()


def _delete(product: Path) -> None:
    bps_logger.debug("Removing: %s", product)
    if product.exists():
        try:
            open_product_folder(product).delete()
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:
            bps_logger.warning(
                "An error occurred during removal of %s intermediate product: '%s: %s'",
                product,
                exc.__class__.__name__,
                exc,
            )
        else:
            if product.exists():
                # product still exists: some file are left inside
                _remove_additional_files(product)

                if not list(product.iterdir()):
                    product.rmdir()


@dataclass
class ProductInfo:
    """Path and removal info"""

    path: Path
    to_be_removed: bool


@dataclass
class BPSAntennaPatternsProducts:
    """Antenna patterns products"""

    ant_d1_h_product: Path
    ant_d1_v_product: Path
    ant_d2_h_product: Path
    ant_d2_v_product: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> BPSAntennaPatternsProducts:
        """Setup the paths of the antenna patterns products

        Parameters
        ----------
        base_dir : Path
            Aux directory where to save the antenna patterns products

        Returns
        -------
        BPSAntennaPatternsProducts
            Struct with the paths to the products
        """
        return cls(
            ant_d1_h_product=base_dir.joinpath(names.ANT_D1_H_PRODUCT),
            ant_d1_v_product=base_dir.joinpath(names.ANT_D1_V_PRODUCT),
            ant_d2_h_product=base_dir.joinpath(names.ANT_D2_H_PRODUCT),
            ant_d2_v_product=base_dir.joinpath(names.ANT_D2_V_PRODUCT),
        )

    def delete(self) -> None:
        """Delete the aux products"""
        for product_path in [
            self.ant_d1_h_product,
            self.ant_d1_v_product,
            self.ant_d2_h_product,
            self.ant_d2_v_product,
        ]:
            _delete(product_path)


@dataclass
class L1PreProcessorOutputProducts:
    """L1 Pre Processor output products paths"""

    extracted_raw_product: Path
    extracted_dyncal_product: Path
    pgp_product: Path
    amp_phase_drift_product: Path
    chirp_replica_product: Path
    internal_delays_file: Path
    channel_imbalance_file: Path
    tx_power_tracking_product: Path
    est_noise_product: Path
    repaired_attitude: Path
    antenna_patterns: BPSAntennaPatternsProducts
    ssp_headers_file: Path
    report_file: Path

    directory: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> L1PreProcessorOutputProducts:
        """Setup the paths of the output products

        Parameters
        ----------
        output_dir : Path
            Output directory where to save the output products

        Returns
        -------
        L1PreProcessorOutputProducts
            Struct with the paths to the products
        """
        return cls(
            extracted_raw_product=output_dir.joinpath(names.EXTRACTED_RAW_PRODUCT),
            extracted_dyncal_product=output_dir.joinpath(names.EXTRACTED_DYNCAL_PRODUCT),
            pgp_product=output_dir.joinpath(names.PGP_PRODUCT),
            amp_phase_drift_product=output_dir.joinpath(names.AMP_PHASE_DRIFT_PRODUCT),
            chirp_replica_product=output_dir.joinpath(names.CHIRP_REPLICA_PRODUCT),
            internal_delays_file=output_dir.joinpath(names.INTERNAL_DELAYS_FILE),
            channel_imbalance_file=output_dir.joinpath(names.CHANNEL_IMBALANCE_FILE),
            tx_power_tracking_product=output_dir.joinpath(names.TX_POWER_TRACKING_PRODUCT),
            est_noise_product=output_dir.joinpath(names.EST_NOISE_PRODUCT),
            repaired_attitude=output_dir.joinpath(names.REPAIRED_ATTITUDE),
            antenna_patterns=BPSAntennaPatternsProducts.from_base_dir(output_dir),
            ssp_headers_file=output_dir.joinpath(names.SSP_HEADERS_FILE),
            report_file=output_dir.joinpath(names.L1_PREPROC_REPORT),
            directory=output_dir,
        )

    def delete(self) -> None:
        """Delete the output products"""
        for product_path in self.list_products():
            _delete(product_path)
        self.repaired_attitude.unlink(missing_ok=True)
        self.internal_delays_file.unlink(missing_ok=True)
        self.channel_imbalance_file.unlink(missing_ok=True)
        self.ssp_headers_file.unlink(missing_ok=True)
        self.report_file.unlink(missing_ok=True)
        self.antenna_patterns.delete()
        if len(list(self.directory.iterdir())) == 0:
            self.directory.rmdir()

    def list_products(self) -> list[Path]:
        """Get list of output products"""
        return [
            self.extracted_raw_product,
            self.extracted_dyncal_product,
            self.pgp_product,
            self.amp_phase_drift_product,
            self.chirp_replica_product,
            self.tx_power_tracking_product,
            self.est_noise_product,
        ]


@dataclass
class L1CoreProcessorOutputProducts:
    """L1 Core Processor output products paths"""

    ionospheric_height_model_file: Path
    resampling_filter: Path
    status_file: Path

    directory: Path

    output_products: dict[names.IntermediateProductID, ProductInfo]
    geometric_doppler: Path
    extracted_raw_annotation: Path
    preprocessor_report: Path

    main_slc_id: names.IntermediateProductID | None = None

    def add_output_if_not_present(self, product_id: names.IntermediateProductID, to_be_removed: bool):
        """Add output with default name if not already present"""
        if not self.output_products.get(product_id):
            self.output_products[product_id] = ProductInfo(
                self.directory.joinpath(product_id.to_name()),
                to_be_removed=to_be_removed,
            )

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> L1CoreProcessorOutputProducts:
        """Setup the paths of the output products

        Parameters
        ----------
        output_dir : Path
            Output directory where to save the output products

        Returns
        -------
        L1CoreProcessorOutputProducts
            Struct with the paths to the products
        """

        status_file = output_dir.joinpath(names.BPS_L1_CORE_PROCESSOR_STATUS_FILE_NAME)

        return cls(
            ionospheric_height_model_file=output_dir.joinpath(names.IONOSPHERIC_HEIGHT_MODEL_FILE),
            resampling_filter=output_dir.joinpath(names.RESAMPLING_FILTER_PRODUCT),
            status_file=status_file,
            directory=output_dir,
            output_products={},
            geometric_doppler=output_dir.joinpath(names.GEOMETRIC_DC_PRODUCT),
            extracted_raw_annotation=output_dir.joinpath(names.EXTRACTED_RAW_ANNOTATION),
            preprocessor_report=output_dir.joinpath(names.L1_PREPROC_REPORT),
        )

    def update_grd_output(self):
        """If not already promoted, promote BPSL1CoreProcessor GRD output"""
        self.add_output_if_not_present(names.IntermediateProductID.GRD, to_be_removed=True)

    def update_main_slc_output(
        self,
        polarimetric_compensation_enabled: bool,
        autofocus_enabled: bool,
    ):
        """If not already promoted, promote BPSL1CoreProcessor main SLC output"""

        if autofocus_enabled:
            self.main_slc_id = names.IntermediateProductID.SLC_AF_CORRECTED
            self.add_output_if_not_present(names.IntermediateProductID.SLC_IONO_CORRECTED, to_be_removed=True)
        elif polarimetric_compensation_enabled:
            self.main_slc_id = names.IntermediateProductID.SLC_IONO_CORRECTED
        else:
            self.main_slc_id = names.IntermediateProductID.SLC

        self.add_output_if_not_present(self.main_slc_id, to_be_removed=True)

    def update_parc_required_outputs(self):
        """Update BPSL1CoreProcessor output products with those later required for PARC processing"""
        self.add_output_if_not_present(names.IntermediateProductID.FR, to_be_removed=True)
        self.add_output_if_not_present(names.IntermediateProductID.PHASE_SCREEN_BB, to_be_removed=True)

    def update_lut_outputs(
        self,
        rfi_mitigation_enabled: bool,
        ionospheric_calibration_enabled: bool,
        autofocus_enabled: bool,
        nesz_map_generation_enabled: bool,
    ):
        """Update BPSL1CoreProcessor auxiliary products to outputs for look up tables and annotations"""
        self.add_output_if_not_present(names.IntermediateProductID.SLANT_DEM, to_be_removed=True)

        if rfi_mitigation_enabled:
            self.add_output_if_not_present(names.IntermediateProductID.RFI_TIME_MASK, to_be_removed=True)

            self.add_output_if_not_present(names.IntermediateProductID.RFI_FREQ_MASK, to_be_removed=True)

        # There is no doppler centroid estimator grid domain lut, but it is requested for annotation
        self.add_output_if_not_present(
            names.IntermediateProductID.DOPPLER_CENTROID_ESTIMATOR_GRID,
            to_be_removed=True,
        )

        if ionospheric_calibration_enabled:
            self.add_output_if_not_present(names.IntermediateProductID.IONO_CAL_REPORT, to_be_removed=True)
            self.add_output_if_not_present(names.IntermediateProductID.FR, to_be_removed=True)
            self.add_output_if_not_present(names.IntermediateProductID.FR_PLANE, to_be_removed=True)
            self.add_output_if_not_present(names.IntermediateProductID.PHASE_SCREEN_BB, to_be_removed=True)

        if autofocus_enabled:
            self.add_output_if_not_present(names.IntermediateProductID.PHASE_SCREEN_AF, to_be_removed=True)

        if nesz_map_generation_enabled:
            self.add_output_if_not_present(names.IntermediateProductID.SLC_NESZ_MAP, to_be_removed=True)

    def delete(self) -> None:
        """Delete the output products"""

        products_to_delete = [
            product_info.path for _, product_info in self.output_products.items() if product_info.to_be_removed
        ]
        products_to_delete.append(self.resampling_filter)

        for product_path in products_to_delete:
            if product_path.is_dir():
                _delete(product_path)
            else:
                product_path.unlink(missing_ok=True)

        self.ionospheric_height_model_file.unlink(missing_ok=True)
        self.status_file.unlink(missing_ok=True)
        if self.geometric_doppler.exists():
            for annotation_files in self.geometric_doppler.iterdir():
                annotation_files.unlink()

            self.geometric_doppler.rmdir()
        self.extracted_raw_annotation.unlink(missing_ok=True)
        self.preprocessor_report.unlink(missing_ok=True)

        if len(list(self.directory.iterdir())) == 0:
            self.directory.rmdir()

    def list_products(self) -> list[Path]:
        """Get list of output products"""
        return [product_info.path for _, product_info in self.output_products.items()]


def retrieve_additional_core_outputs(
    promotion_dir: Path,
) -> dict[names.IntermediateProductID, ProductInfo]:
    """intermediates promoted"""

    to_be_promoted = {
        names.IntermediateProductID.RAW_MITIGATED,
        names.IntermediateProductID.RGC_DC_FR_ESTIMATOR,
        names.IntermediateProductID.SLC,
        names.IntermediateProductID.SLC_IONO_CORRECTED,
        names.IntermediateProductID.SRD_MULTILOOKED,
        names.IntermediateProductID.SRD_DENOISED,
        names.IntermediateProductID.GRD,
    }

    products = {
        product_id: ProductInfo(
            path=promotion_dir.joinpath(names.L1_CORE_PROC_OUTPUT_FOLDER, product_id.to_name()),
            to_be_removed=False,
        )
        for product_id in to_be_promoted
    }

    return products


@dataclass
class L1ParcCoreAdditionalInputFiles:
    """L1 Parc Core Processor interface files"""

    faraday_rotation_product: Path
    phase_screen_product: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> L1ParcCoreAdditionalInputFiles:
        """Setup class from base directory"""
        return cls(
            faraday_rotation_product=base_dir.joinpath("Input" + names.IntermediateProductID.FR.to_name()),
            phase_screen_product=base_dir.joinpath("Input" + names.IntermediateProductID.PHASE_SCREEN_BB.to_name()),
        )

    def delete(self):
        """Delete files"""
        _delete(self.faraday_rotation_product)
        _delete(self.phase_screen_product)
