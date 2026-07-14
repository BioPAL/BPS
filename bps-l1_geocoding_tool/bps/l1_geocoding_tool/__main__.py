# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Main entry point for the BPS L1 Geocoding Tool"""

from bps.l1_geocoding_tool.cli import run_cli


def main():
    """Main function to run the L1 Geocoding Tool"""
    # pylint: disable-next=no-value-for-parameter
    run_cli()


if __name__ == "__main__":
    main()
