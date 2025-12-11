# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

from bps.transcoder.sarproduct.sarproduct import SARProduct


class BIOMASSL0Product(SARProduct):
    """_summary_

    Parameters
    ----------
    SARProduct : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    def __init__(self, product: SARProduct | None = None, is_monitoring: bool = False) -> None:
        """_summary_

        Parameters
        ----------
        product : SARProduct, optional
            _description_, by default None
        is_monitoring : bool, optional
            _description_, by default False
        """
        SARProduct.__init__(self)
        self.is_monitoring = is_monitoring

        if product is not None:
            self.__dict__.update(product.__dict__)

        self.mission_phase_id = None
        self.instrument_configuration_id = None
        self.datatake_id = None
        self.orbit_drift_flag = None
        self.global_coverage_id = None
        self.major_cycle_id = None
        self.repeat_cycle_id = None
        self.track_number = None
        self.slice_number = None
        self.slice_status = None
        self.baseline_id = None
        self.sensing_start_time = None
        self.sensing_stop_time = None

        self.mission_phase_id = "INTERFEROMETRIC"
        self.instrument_configuration_id = 1
        self.datatake_id = 1
        self.orbit_drift_flag = False
        self.global_coverage_id = 1
        self.major_cycle_id = 1
        self.repeat_cycle_id = 1
        self.track_number = 1
        self.slice_number = 1
        self.slice_status = "NOMINAL"
        self.baseline_id = 1
        self.sensing_start_time = self.start_time
        self.sensing_stop_time = self.stop_time

        self.file_sizes: dict[str, int] = {}
        self.bit_rate: float = 0  # Mbit/s
