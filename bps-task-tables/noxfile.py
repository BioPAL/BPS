# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

"""
Automation file
---------------
"""

from pathlib import Path

import nox
import nox.command


def is_valid(session: nox.Session, xml_file: Path, schema: Path) -> bool:
    """Wether xml is valid against scheme"""
    try:
        session.run(
            *f"xmllint {xml_file} --noout --schema {schema}".split(),
            external=True,
        )
    except nox.command.CommandFailed:
        return False
    return True


@nox.session()
def validate_task_tables(session: nox.Session) -> None:
    """Validate all task tables against XSD"""
    task_tables = Path().rglob("*TaskTable*.xml")

    schema_reference = Path(
        "..",
        "xsd",
        "icd-xsd",
        "generic",
        "tasktable",
        "v1_4",
        "IF_PF_Proc_Task_Table.xsd",
    )

    invalid_task_tables = list(filter(lambda jo: not is_valid(session, jo, schema_reference), task_tables))
    if invalid_task_tables:
        for table in invalid_task_tables:
            session.warn(f"Invalid TaskTable file: {table}")
        session.error(f"Found {len(invalid_task_tables)} invalid Task Tables")


@nox.session()
def validate_product_metadata_mapping(session: nox.Session) -> None:
    """Validate all pmm against XSD"""
    pmm_files = Path().rglob("*ProductMetadataMapping.xml")

    schema_reference = Path(
        "..",
        "xsd",
        "icd-xsd",
        "generic",
        "tasktable",
        "v1_4",
        "IF_PF_Proc_ProductMetadataMapping.xsd",
    )

    invalid_pmm = list(filter(lambda jo: not is_valid(session, jo, schema_reference), pmm_files))
    if invalid_pmm:
        for pmm in invalid_pmm:
            session.warn(f"Invalid ProductMetadataMapping file: {pmm}")
        session.error(f"Found {len(invalid_pmm)} invalid ProductMetadataMapping files")


@nox.session()
def version_update(session: nox.Session) -> None:
    """rename tt and update content: nox -s version_update -- 02.21 02.22"""
    if len(session.posargs) != 2:
        raise RuntimeError(
            "Unexpected number of input arguments, usage: " + "'python -m nox -s version_update -- 02.00 02.12"
        )

    current_version, new_version = session.posargs[0:2]
    session.log(f"Bumping version from {current_version} to {new_version}")

    proc_tag_old = f"<Processor_Version>{current_version}</Processor_Version>"
    proc_tag_new = f"<Processor_Version>{new_version}</Processor_Version>"
    sw_tag_old = f"<SW_Version_Tag>{current_version}</SW_Version_Tag>"
    sw_tag_new = f"<SW_Version_Tag>{new_version}</SW_Version_Tag>"
    task_tag_old = f"<Task_Version>{current_version}</Task_Version>"
    task_tag_new = f"<Task_Version>{new_version}</Task_Version>"

    task_tables = list(Path().rglob("*TaskTable*.xml"))
    task_tables_renamed = 0
    task_tables_updated = 0
    for file in task_tables:
        text = file.read_text(encoding="utf-8")

        update_file = False
        if proc_tag_old in text:
            update_file = True
            text = text.replace(proc_tag_old, proc_tag_new)

        if sw_tag_old in text:
            update_file = True
            text = text.replace(sw_tag_old, sw_tag_new)

        if task_tag_old in text:
            update_file = True
            text = text.replace(task_tag_old, task_tag_new)

        if update_file:
            file.write_text(text, encoding="utf-8")
            session.run("git", "add", str(file), external=True)
            task_tables_updated += 1

        file_name = file.name
        new_file_name = file_name.replace(current_version, new_version)
        if new_file_name != file_name:
            new_file = file.with_name(new_file_name)
            session.run("git", "mv", "-f", str(file), str(new_file), external=True)
            task_tables_renamed += 1

    session.log(f"{task_tables_updated} task tables updates")
    session.log(f"{task_tables_renamed} task tables renamed")

    proc_tag_old = f"<ProcessorVersion>{current_version}</ProcessorVersion>"
    proc_tag_new = f"<ProcessorVersion>{new_version}</ProcessorVersion>"

    pmm_files = Path().rglob("*ProductMetadataMapping*.xml")
    pmm_renamed = 0
    pmm_updated = 0
    for file in pmm_files:
        text = file.read_text(encoding="utf-8")

        update_file = False
        if proc_tag_old in text:
            update_file = True
            text = text.replace(proc_tag_old, proc_tag_new)

        if update_file:
            file.write_text(text, encoding="utf-8")
            session.run("git", "add", str(file), external=True)
            pmm_updated += 1

        file_name = file.name
        new_file_name = file_name.replace(current_version, new_version)
        if new_file_name != file_name:
            new_file = file.with_name(new_file_name)
            session.run("git", "mv", "-f", str(file), str(new_file), external=True)
            pmm_renamed += 1

    session.log(f"{pmm_updated} product metadata mapping updates")
    session.log(f"{pmm_renamed} product metadata mapping renamed")
