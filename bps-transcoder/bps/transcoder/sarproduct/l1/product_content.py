# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""BPS L1 product content"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bps.transcoder.utils.product_name import parse_l1product_name


@dataclass(frozen=True)
class L1ProductContent:
    """Relative paths to product content. Some may be None depending on standard/monitoring and on scs/dgm"""

    mph_file: Path

    measurement_folder: Path | None
    abs_raster: Path | None
    phase_raster: Path | None
    vrt: Path | None

    annotation_folder: Path
    main_annotation: Path
    lut: Path

    navigation_folder: Path
    orbit: Path
    attitude: Path

    preview_folder: Path
    quicklook: Path
    overlay: Path

    schema_folder: Path
    l1ab_main_ann_xsd: Path
    l1_annotation_xsd: Path
    l1_overlay_xsd: Path
    l1_overlay_support_xsd: Path
    common_types_xsd: Path
    aux_orb_xsd: Path
    aux_att_xsd: Path
    vrt_xsd: Path | None

    @classmethod
    def from_name(cls, name: str) -> L1ProductContent:
        """Build all paths regardless of product existance"""

        product_info = parse_l1product_name(name)
        standard_product = not product_info.is_monitoring
        assert product_info.processor_id is not None
        scs_product = product_info.processor_id[0:3] == "SCS"

        lower_name = name.lower()
        name_root = lower_name[:-10]

        mph_file = Path(lower_name + ".xml")

        measurement_folder = None
        abs_raster = None
        phase_raster = None
        vrt = None
        if standard_product:
            measurement_folder = Path("measurement")
            abs_raster = measurement_folder.joinpath(name_root + "_i_abs.tiff")

            if scs_product:
                phase_raster = measurement_folder.joinpath(name_root + "_i_phase.tiff")
                vrt = measurement_folder.joinpath(name_root + "_i.vrt")

        annotation_folder = Path("annotation")
        main_annotation = annotation_folder.joinpath(name_root + "_annot.xml")
        lut = annotation_folder.joinpath(name_root + "_lut.nc")

        navigation_folder = annotation_folder.joinpath("navigation")

        orbit = navigation_folder.joinpath(name_root + "_orb.xml")
        attitude = navigation_folder.joinpath(name_root + "_att.xml")

        preview_folder = Path("preview")
        quicklook = preview_folder.joinpath(name_root + "_ql.png")
        overlay = preview_folder.joinpath(name_root + "_map.kml")

        schema_folder = Path("schema")

        l1ab_main_ann_xsd = schema_folder.joinpath("bio-l1ab-main-annotation.xsd")
        l1_annotation_xsd = schema_folder.joinpath("bio-l1-annotations.xsd")
        l1_overlay_xsd = schema_folder.joinpath("bio-l1-overlay.xsd")
        l1_overlay_support_xsd = schema_folder.joinpath("bio-l1-overlay-support.xsd")
        common_types_xsd = schema_folder.joinpath("bio-common-types.xsd")
        aux_orb_xsd = schema_folder.joinpath("bio-aux-orbit.xsd")
        aux_att_xsd = schema_folder.joinpath("bio-aux-attitude.xsd")

        vrt_xsd = None
        if vrt is not None:
            vrt_xsd = schema_folder.joinpath("bio-l1-vrt.xsd")

        return L1ProductContent(
            mph_file=mph_file,
            measurement_folder=measurement_folder,
            abs_raster=abs_raster,
            phase_raster=phase_raster,
            vrt=vrt,
            annotation_folder=annotation_folder,
            main_annotation=main_annotation,
            lut=lut,
            navigation_folder=navigation_folder,
            orbit=orbit,
            attitude=attitude,
            preview_folder=preview_folder,
            quicklook=quicklook,
            overlay=overlay,
            schema_folder=schema_folder,
            l1ab_main_ann_xsd=l1ab_main_ann_xsd,
            l1_annotation_xsd=l1_annotation_xsd,
            l1_overlay_xsd=l1_overlay_xsd,
            l1_overlay_support_xsd=l1_overlay_support_xsd,
            common_types_xsd=common_types_xsd,
            aux_orb_xsd=aux_orb_xsd,
            aux_att_xsd=aux_att_xsd,
            vrt_xsd=vrt_xsd,
        )

    @property
    def folders(self) -> list[Path]:
        """All folders and sub-folders of the product. Relative to product dir. Including product dir."""
        folders = (
            Path("."),
            self.annotation_folder,
            self.navigation_folder,
            self.preview_folder,
            self.schema_folder,
            self.measurement_folder,
        )

        return [folder for folder in folders if folder is not None]

    @property
    def files_and_validators_if_any(self) -> list[tuple[Path, Path | None]]:
        """All files in product, paired with the corresponding XSD if any. Relative to product dir."""
        files = [
            (self.abs_raster, None),
            (self.phase_raster, None),
            (
                self.vrt,
                self.vrt_xsd,
            ),
            (
                self.main_annotation,
                self.l1ab_main_ann_xsd,
            ),
            (self.lut, None),
            (
                self.orbit,
                self.aux_orb_xsd,
            ),
            (
                self.attitude,
                self.aux_att_xsd,
            ),
            (
                self.quicklook,
                None,
            ),
            (
                self.overlay,
                self.l1_overlay_xsd,
            ),
        ]

        return [(file, schema) for file, schema in files if file is not None]

    @property
    def xsd_schema_files(self) -> list[Path]:
        """All schema files in product. Relative to product dir."""
        xsd_files = [
            self.l1ab_main_ann_xsd,
            self.l1_annotation_xsd,
            self.l1_overlay_xsd,
            self.l1_overlay_support_xsd,
            self.common_types_xsd,
            self.aux_orb_xsd,
            self.aux_att_xsd,
            self.vrt_xsd,
        ]
        return [xsd_file for xsd_file in xsd_files if xsd_file is not None]
