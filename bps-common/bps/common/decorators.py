# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Decorators module
-----------------
"""

import functools
from datetime import datetime

from bps.common import bps_logger


def log_elapsed_time(logged_name: str):
    """Decorated functions now logs elapsed time with the bps_logger"""

    def decorator(func):
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            processing_start_time = datetime.now()
            outputs = func(*args, **kwargs)
            processing_stop_time = datetime.now()
            elapsed_time = processing_stop_time - processing_start_time
            bps_logger.info(
                "%s total processing time: %.3f s",
                logged_name,
                elapsed_time.total_seconds(),
            )
            return outputs

        return decorated_func

    return decorator
