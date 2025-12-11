# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import os
from glob import glob


class BIOMASSAuxFile:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        self.name = None
        self.validity_start_time = None
        self.validity_stop_time = None


class BIOMASSAuxFileStructure:
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

        self.mph_file = None

        self.__set_product_paths()

    def __set_product_paths(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        if os.path.exists(self.product_path):
            # Set paths of BIOMASS auxiliary file single files starting from an existing one
            # - MPH file
            self.mph_file = (glob(os.path.join(self.product_path, "*.xml")) or [None])[0]
        else:
            # Set paths of BIOMASS auxiliary file single files starting from product name
            product_name = os.path.basename(self.product_path)
            # - MPH file
            self.mph_file = os.path.join(self.product_path, product_name.lower() + ".xml")

        return True
