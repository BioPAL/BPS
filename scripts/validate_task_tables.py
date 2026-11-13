# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Validate task-table and product-metadata XML files against their XSDs."""

import subprocess
import sys
from pathlib import Path


def validate_group(
    script_dir: Path,
    pattern: str,
    schema: Path,
    label: str,
) -> int:
    """Validate matching XML files and return the number of failures."""
    invalid = 0
    for xml_file in sorted(script_dir.rglob(pattern)):
        result = subprocess.run(
            ["xmllint", str(xml_file), "--noout", "--schema", str(schema)],
            check=False,
        )
        if result.returncode != 0:
            print(f"Invalid {label} file: {xml_file}", file=sys.stderr)
            invalid += 1

    if invalid:
        print(f"Found {invalid} invalid {label} file(s)", file=sys.stderr)

    return invalid


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    task_table_dir = root_dir / "bps-task-tables"
    schema_dir = root_dir / "xsd" / "icd-xsd" / "generic" / "tasktable" / "v1_4"

    invalid = validate_group(
        task_table_dir,
        "*TaskTable*.xml",
        schema_dir / "IF_PF_Proc_Task_Table.xsd",
        "TaskTable",
    )

    invalid += validate_group(
        task_table_dir,
        "*ProductMetadataMapping.xml",
        schema_dir / "IF_PF_Proc_ProductMetadataMapping.xsd",
        "ProductMetadataMapping",
    )

    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
