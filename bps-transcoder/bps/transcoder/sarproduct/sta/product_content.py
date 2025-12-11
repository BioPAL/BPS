# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1c Product Structure
-------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bps.transcoder.utils.product_name import (
    InvalidBIOMASSProductName,
    is_l1_product_name_valid,
    parse_l1product_name,
)


# Handle an invalid L1c product.
class InvalidBIOMASSStackProductStructureError(RuntimeError):
    """Handle an invalid L1c product."""


# Handle an invalid L1c product name.
class InvalidBIOMASSStackProductName(RuntimeError):
    """Handle an invalid L1c product name."""


@dataclass
class BIOMASSStackProductStructure:
    """
    Object that store a BIOMASS L1c product structure. Standard (S) L1c products
    have the following structure.

    BIO_S1_STA__1S_20170206T164349_20170206T164400_I_G03_M03_C03_T000_F001_01_CP8J3M/
    ├── annotation_coregistered
    │   ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_annot.xml
    │   ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_lut.nc
    │   └── navigation
    │       ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_att.xml
    │       └── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_orb.xml
    ├── annotation_primary
    │   ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_annot.xml
    │   └── navigation
    │       ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_att.xml
    │       └── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_orb.xml
    ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_01_cp8j3m.xml
    ├── measurement
    │   ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_i_abs.tiff
    │   ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_i_phase.tiff
    │   └── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_i.vrt
    ├── preview
    │   ├── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_map.kml
    │   └── bio_s1_sta__1s_20170206t164349_20170206t164400_i_g03_m03_c03_t000_f001_ql.png
    └── schema
        ├── bio-aux-attitude.xsd
        ├── bio-aux-orbit.xsd
        ├── bio-common-types.xsd
        ├── bio-l1ab-main-annotation.xsd
        ├── bio-l1-annotations.xsd
        ├── bio-l1c-main-annotation.xsd
        ├── bio-l1-overlay-support.xsd
        ├── bio-l1-overlay.xsd
        └── bio-l1-vrt.xsd

    Monitoring (M) products do not contain the 'measurement' folder.

    """

    product_path: Path
    """Path to the L1c product."""

    mph_file: Path
    """Path to the MPH file (XML)."""

    main_annotation1_file: Path
    """Path to the primary annotation file (XML)."""

    main_annotation2_file: Path
    """Path to the secondary (coregistered) annotation file (XML)."""

    lut_annotation2_file: Path
    """Path to the secondary (coregistered) LUT file (NetCDF4)."""

    orbit1_file: Path
    """Path to the primary orbit file (XML)."""

    attitude1_file: Path
    """Path to the primary attitude file (XML)."""

    orbit2_file: Path
    """Path to the secondary (coregistered) orbit file (XML)."""

    attitude2_file: Path
    """Path to the secondary (coregistered) attitude file (XML)."""

    quicklook_file: Path
    """Path to the quick-look preview file (png)."""

    overlay_file: Path
    """Path to the KML overlay preview file (KML)."""

    schema_files: dict[str, Path]
    """Paths to the XSD schema file (XSDs)."""

    measurement_files: dict[str, Path] = field(init=False)
    """Dictionary containing the measurement (GeoTIFFs)."""

    vrt_file: Path = field(init=False)
    """Path to the VRT file (VRT)."""

    def __init__(
        self,
        product_path: str | Path,
        *,
        is_monitoring: bool,
        exists_ok: bool,
    ):
        """
        Initialize the BIOMASS Stack product structure.

        Arguments
        ---------
        product_path: Union[str, Path]
            The product path, can be existing or a destination path.

        is_monitoring: bool
            Whether the product is a monitoring one.

        exists_ok: bool
            If False, raise an error if the product already exists.

        Raises
        ------
        FileExistsError
            If the product already exists (unless exists_ok is True).

        InvalidBIOMASSStackProductStructureError
            If the product exists but the product structure is not conformant
            to the BIOMASS L1c product structure specifications.

        InvalidBIOMASSProductName:
            If the product name is not conformant to the BIOMASS L1c product
            naming conventions.

        """
        self.product_path = Path(product_path)
        self.__init_product_structure(is_monitoring)

    def __init_product_structure(self, is_monitoring: bool):
        """Initialize the product internal paths."""
        if self.product_path.exists():
            # Initialize the product paths by querying the product structure.
            self.__load_product_structure_from_product_path(is_monitoring)
        else:
            # Initialize the (virtual) structure using the name as a seed.
            self.__init_product_structure_from_product_name(is_monitoring)

    def __load_product_structure_from_product_path(self, is_monitoring: bool):
        """Initialize the product structure from a path to an existing L1c product."""
        # Check if the product is actually a folder.
        if not self.product_path.is_dir():
            raise InvalidBIOMASSStackProductStructureError("{self.product_path} is not a BIOMASS L1c product")

        # Raise if product name is invalid.
        if not is_l1_product_name_valid(Path(self.product_path).name):
            raise InvalidBIOMASSProductName(f"'{Path(self.product_path).name}' is not a valid BIOMASS L1c product name")

        product_name = self.product_path.name.lower()
        product_name_stem = product_name[:-10]

        # MPH file.
        self.mph_file = _find_files(
            self.product_path,
            pattern=f"{product_name}.xml",
            expected_num_files=1,
        )[0]

        # Measurement files.
        if not is_monitoring:
            self.measurement_files = {
                "abs": _find_files(
                    self.measurement_dir,
                    pattern=f"{product_name_stem}*_abs.tiff",
                    expected_num_files=1,
                )[0],
                "phase": _find_files(
                    self.measurement_dir,
                    pattern=f"{product_name_stem}*_phase.tiff",
                    expected_num_files=1,
                )[0],
            }
            self.vrt_file = _find_files(
                self.measurement_dir,
                pattern=f"{product_name_stem}*.vrt",
                expected_num_files=1,
            )[0]

        # Primary image Annotation file.
        self.main_annotation1_file = _find_files(
            self.annotation_primary_dir,
            pattern=f"{product_name_stem}*_annot.xml",
            expected_num_files=1,
        )[0]

        # Secondary image annotation file.
        self.main_annotation2_file = _find_files(
            self.annotation_coregistered_dir,
            pattern=f"{product_name_stem}*_annot.xml",
            expected_num_files=1,
        )[0]
        self.lut_annotation2_file = _find_files(
            self.annotation_coregistered_dir,
            pattern=f"{product_name_stem}*_lut.nc",
            expected_num_files=1,
        )[0]

        # Primary orbit and attitude files.
        self.orbit1_file = _find_files(
            self.navigation_primary_dir,
            pattern=f"{product_name_stem}*_orb.xml",
            expected_num_files=1,
        )[0]
        self.attitude1_file = _find_files(
            self.navigation_primary_dir,
            pattern=f"{product_name_stem}*_att.xml",
            expected_num_files=1,
        )[0]

        # Secondary orbit and attitude files.
        self.orbit2_file = _find_files(
            self.navigation_coregistered_dir,
            pattern=f"{product_name_stem}*_orb.xml",
            expected_num_files=1,
        )[0]
        self.attitude2_file = _find_files(
            self.navigation_coregistered_dir,
            pattern=f"{product_name_stem}*_att.xml",
            expected_num_files=1,
        )[0]

        # Quick-look and overview file.
        self.quicklook_file = _find_files(
            self.preview_dir,
            pattern=f"{product_name_stem}*_ql.png",
            expected_num_files=1,
        )[0]
        self.overlay_file = _find_files(
            self.preview_dir,
            pattern=f"{product_name_stem}*_map.kml",
            expected_num_files=1,
        )[0]

        # Schema files.
        self.schema_files = _find_files(
            self.xsd_schema_dir,
            pattern="*.xsd",
            expected_num_files=8 if is_monitoring else 9,
        )

    def __init_product_structure_from_product_name(self, is_monitoring: bool):
        """Initialize the product structure from a L1c product name."""
        # Parse the product name to make sure it's a valid name. In case it's
        # not, this will throw a InvalidBIOMASSL1ProductName
        parse_l1product_name(self.product_path.name)
        if "_STA_" not in self.product_path.name:
            raise InvalidBIOMASSStackProductName("BIOMASS Stack (L1c) product names must contain the 'STA' tag.")
        product_name = self.product_path.name.lower()
        product_name_stem = product_name[:-10]

        # MPH file.
        self.mph_file = self.product_path / f"{product_name}.xml"

        # Measurement files.
        if not is_monitoring:
            self.measurement_files = {
                "abs": self.measurement_dir / f"{product_name_stem}_i_abs.tiff",
                "phase": self.measurement_dir / f"{product_name_stem}_i_phase.tiff",
            }
            self.vrt_file = self.measurement_dir / f"{product_name_stem}_i.vrt"

        # - Annotation files
        self.main_annotation1_file = self.annotation_primary_dir / f"{product_name_stem}_annot.xml"

        self.main_annotation2_file = self.annotation_coregistered_dir / f"{product_name_stem}_annot.xml"
        self.lut_annotation2_file = self.annotation_coregistered_dir / f"{product_name_stem}_lut.nc"

        # Orbit and attitude files.
        self.orbit1_file = self.navigation_primary_dir / f"{product_name_stem}_orb.xml"
        self.attitude1_file = self.navigation_primary_dir / f"{product_name_stem}_att.xml"

        self.orbit2_file = self.navigation_coregistered_dir / f"{product_name_stem}_orb.xml"
        self.attitude2_file = self.navigation_coregistered_dir / f"{product_name_stem}_att.xml"

        # Quick-look and overlay file.
        self.quicklook_file = self.preview_dir / f"{product_name_stem}_ql.png"
        self.overlay_file = self.preview_dir / f"{product_name_stem}_map.kml"

        # - Schema files
        self.schema_files = {
            "l1ab_main_ann_xsd": self.xsd_schema_dir / "bio-l1ab-main-annotation.xsd",
            "l1c_main_ann_xsd": self.xsd_schema_dir / "bio-l1c-main-annotation.xsd",
            "l1_ann_xsd": self.xsd_schema_dir / "bio-l1-annotations.xsd",
            "l1_ovr_xsd": self.xsd_schema_dir / "bio-l1-overlay.xsd",
            "l1_ovr_support_xsd": self.xsd_schema_dir / "bio-l1-overlay-support.xsd",
            "common_types_xsd": self.xsd_schema_dir / "bio-common-types.xsd",
            "aux_orb_xsd": self.xsd_schema_dir / "bio-aux-orbit.xsd",
            "aux_att_xsd": self.xsd_schema_dir / "bio-aux-attitude.xsd",
        }
        if not is_monitoring:
            self.schema_files.update({"l1_vrt_xsd": self.xsd_schema_dir / "bio-l1-vrt.xsd"})

    @property
    def measurement_dir(self) -> Path:
        """Path to the measurement directory."""
        return self.product_path / "measurement"

    @property
    def annotation_primary_dir(self) -> Path:
        """Path to the primary annotation directory."""
        return self.product_path / "annotation_primary"

    @property
    def navigation_primary_dir(self) -> Path:
        """Path to the navigation primary directory."""
        return self.annotation_primary_dir / "navigation"

    @property
    def annotation_coregistered_dir(self) -> Path:
        """Path to the coregistered (secondary) annotation directory."""
        return self.product_path / "annotation_coregistered"

    @property
    def navigation_coregistered_dir(self) -> Path:
        """Path to the navigation primary directory."""
        return self.annotation_coregistered_dir / "navigation"

    @property
    def preview_dir(self) -> Path:
        """Path to the preview directory."""
        return self.product_path / "preview"

    @property
    def xsd_schema_dir(self) -> Path:
        """Path to the XSD schema directory."""
        return self.product_path / "schema"

    def mkdirs(self, is_monitoring: bool, **mkdir_options: dict):
        """Create the product directories."""
        self.product_path.mkdir(**mkdir_options)
        if not is_monitoring:
            self.measurement_dir.mkdir(**mkdir_options)
        self.annotation_primary_dir.mkdir(**mkdir_options)
        self.navigation_primary_dir.mkdir(**mkdir_options)
        self.annotation_coregistered_dir.mkdir(**mkdir_options)
        self.navigation_coregistered_dir.mkdir(**mkdir_options)
        self.preview_dir.mkdir(**mkdir_options)
        self.xsd_schema_dir.mkdir(**mkdir_options)


def _find_files(root_dir: Path, *, pattern: str, expected_num_files: int) -> list[Path | None]:
    """Possibly find file from root directory matching pattern."""
    matched_paths = list(root_dir.glob(pattern))
    if len(matched_paths) != expected_num_files:
        raise InvalidBIOMASSStackProductStructureError(f"Missing '{pattern}' file from {root_dir}")
    return matched_paths
