# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import os
from glob import glob


class BIOMASSAuxPPS:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        self.name = None
        self.validity_start_time = None
        self.validity_stop_time = None


class BIOMASSAuxPPSStructure:
    """_summary_"""

    def __init__(self, product_path) -> None:
        """_summary_

        Parameters
        ----------
        product_path : _type_
            _description_
        """
        self.product_path = product_path

        self.data_subfolder = "data"
        self.schema_subfolder = "support"

        self.mph_file = None
        self.parameters_file = None
        self.schema_files = None

        self.__set_product_paths()

    def __set_product_paths(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        if os.path.exists(self.product_path):
            # Set paths of BIOMASS AUX_PPS product single files starting from an existing one
            # - MPH file
            self.mph_file = (glob(os.path.join(self.product_path, "*.xml")) or [None])[0]
            # - Parameters file
            self.parameters_file = (glob(os.path.join(self.product_path, self.data_subfolder, "*_pps.xml")) or [None])[
                0
            ]
            # - Schema files
            self.schema_files = glob(os.path.join(self.product_path, self.schema_subfolder, "*.xsd"))
        else:
            # Set paths of BIOMASS AUX_PPS product single files starting from product name
            product_name = os.path.basename(self.product_path)
            # - MPH file
            self.mph_file = os.path.join(self.product_path, product_name.lower() + ".xml")
            # - Parameters file
            self.parameters_file = (
                os.path.join(self.product_path, self.data_subfolder, product_name.lower()[:-10]) + "_pps.xml"
            )
            # - Schema files
            self.schema_files = [
                os.path.join(self.product_path, self.schema_subfolder, f)
                for f in [
                    "bio-aux-pps.xsd",
                    "bio-common-types.xsd",
                ]
            ]

        return True
