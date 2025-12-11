# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 Core Processor interface
---------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IonexFiles:
    """Ionex file"""

    decompressed_1: Path
    decompressed_2: Path
    ionex_file: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> IonexFiles:
        base_name = "L1CoreProcessor"
        return IonexFiles(
            decompressed_1=base_dir.joinpath(f"{base_name}DecompressedIonexFile1.txt"),
            decompressed_2=base_dir.joinpath(f"{base_name}DecompressedIonexFile2.txt"),
            ionex_file=base_dir.joinpath(f"{base_name}DecompressedIonexFile.txt"),
        )

    def delete(self):
        """Delete files"""
        self.decompressed_1.unlink(missing_ok=True)
        self.decompressed_2.unlink(missing_ok=True)
        self.ionex_file.unlink(missing_ok=True)


@dataclass
class L1CoreProcessorInterfaceFiles:
    """L1 Core Processor interface files"""

    input_file: Path
    options_file: Path
    params_file: Path
    dc_poly_file: Path
    ionex_files: IonexFiles
    directory: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> L1CoreProcessorInterfaceFiles:
        """Setup the paths of the L1 Core Processor interface files

        Parameters
        ----------
        base_dir : Path
            directory where to save the L1 Core Processor files

        Returns
        -------
        L1CoreProcessorInterfaceFiles
            Struct with the paths of the files
        """
        base_name = "L1CoreProcessor"
        return cls(
            input_file=base_dir.joinpath(f"{base_name}InputFile.xml"),
            options_file=base_dir.joinpath(f"{base_name}ProcessingOptions.xml"),
            params_file=base_dir.joinpath(f"{base_name}ProcessingParameters.xml"),
            dc_poly_file=base_dir.joinpath(f"{base_name}InputDCPoly.xml"),
            ionex_files=IonexFiles.from_base_dir(base_dir),
            directory=base_dir,
        )

    def delete(self):
        """Delete files"""
        self.input_file.unlink()
        self.options_file.unlink()
        self.params_file.unlink()
        self.dc_poly_file.unlink(missing_ok=True)
        self.ionex_files.delete()
