# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Biomass L1 processor main
-------------------------
"""

import sys

from bps.common import bps_logger
from bps.l1_processor import BPS_L1_PROCESSOR_NAME
from bps.l1_processor.cli import run

SUCCESS_CODE = 0
FAILURE_CODE = 128


def main() -> int:
    """Calls command line interface and intercepts the error code"""

    try:
        # pylint: disable-next=no-value-for-parameter
        run()
    except SystemExit as exit_exc:
        if exit_exc.code == 0:
            bps_logger.info("%s correctly terminated", BPS_L1_PROCESSOR_NAME)
            return SUCCESS_CODE
        exit_exc.code = FAILURE_CODE
        bps_logger.error("Unexpected termination")
    except AssertionError:
        bps_logger.stack_trace(loglevel=bps_logger.logging.ERROR, limit=-1)
        # pylint: disable-next=broad-exception-caught
    except Exception as exc:
        bps_logger.stack_trace(loglevel=bps_logger.logging.DEBUG, limit=-1)
        bps_logger.error("%s: %s", type(exc).__name__, exc)

    bps_logger.error("%s abnormal termination", BPS_L1_PROCESSOR_NAME)
    return FAILURE_CODE


if __name__ == "__main__":
    sys.exit(main())
