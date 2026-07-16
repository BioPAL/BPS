# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import os
from glob import glob
from pathlib import Path

from bps.common.io.mph import get_mph_path


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
            # - MPH file
            self.mph_file = (glob(os.path.join(self.product_path, "*.xml")) or [None])[0]
        else:
            # - MPH file
            self.mph_file = str(get_mph_path(Path(self.product_path)))

        return True
