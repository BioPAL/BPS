# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD models generation helper module
-----------------------------------
"""

import ast
import os
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nox

LICENSE_HEADER = """# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
# 
# SPDX-License-Identifier: MIT

"""

# NOTE: To (re)generate the BrickSAR's XSD models, uncomment relevant line
# below:
#
#    # XSD_FILES += XSD_FILES_BRICKSAR
#


@contextmanager
def working_directory(path: Path):
    """Run code in path, restore previous dir on exit

    Parameters
    ----------
    path : Path
        working directory
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@dataclass
class XSDInfo:
    """XSD source/destination information"""

    xsd_path: Path
    namespace: str
    subpackage: str
    description: str

    namespace_base_dir: Path = field(init=False)
    package: str = field(init=False)
    package_dir: Path = field(init=False)
    models_file: Path = field(init=False)
    init_file: Path = field(init=False)
    module_docstring: str = field(init=False)

    def __post_init__(self):
        self.namespace_base_dir = Path(f"bps-{self.namespace}").absolute()
        self.package_dir = self.namespace_base_dir.joinpath("bps", self.namespace, "io", self.subpackage)
        self.models_file = self.package_dir.joinpath("models.py")
        self.init_file = self.package_dir.joinpath("__init__.py")

        self.package = f"bps.{self.namespace}.io.{self.subpackage}.models"
        self.module_docstring = '"""\n' + self.description + "\n" + len(self.description) * "-" + '\n"""\n'


XSD_SOURCE_DIR = Path("xsd").absolute()
BRICKSAR_XSD = XSD_SOURCE_DIR.joinpath("bricksar-xsd")
BIOMASS_XSD = XSD_SOURCE_DIR.joinpath("biomass-xsd")
ICD_XSD = XSD_SOURCE_DIR.joinpath("icd-xsd")

XSD_FILES_BRICKSAR = [
    XSDInfo(
        xsd_path=BRICKSAR_XSD.joinpath("aresys_generic_inputfile.xsd"),
        namespace="common",
        subpackage="aresys_inputfile_models",
        description="XSD BPS Input file models",
    ),
    XSDInfo(
        xsd_path=BRICKSAR_XSD.joinpath("aresys_generic_conf.xsd"),
        namespace="common",
        subpackage="aresys_configuration_models",
        description="XSD BPS Configuration file models",
    ),
    XSDInfo(
        xsd_path=BRICKSAR_XSD.joinpath("biomassChannelImbalances.xsd"),
        namespace="l1_processor",
        subpackage="biomass_channel_imbalance",
        description="XSD BPS Channel imbalance file models",
    ),
    XSDInfo(
        xsd_path=BRICKSAR_XSD.joinpath("biomassL0ImportPreProcAnnotations.xsd"),
        namespace="transcoder",
        subpackage="biomass_l1_preproc_annotations",
        description="XSD BPS l1 preprocesso annotation xml file models",
    ),
]

XSD_FILES_ICD = [
    XSDInfo(
        xsd_path=ICD_XSD.joinpath("generic", "joborder", "v1_4", "IF_PF_Proc_Job_Order.xsd"),
        namespace="common",
        subpackage="joborder_models",
        description="XSD Job Order models",
    ),
]

XSD_BIO_COMMON_TYPES = XSDInfo(
    xsd_path=BIOMASS_XSD.joinpath("bio-common-types.xsd"),
    namespace="common",
    subpackage="common_types",
    description="XSD common types",
)

XSD_BIO_L1_ANNOTATIONS_COMMON = XSDInfo(
    xsd_path=BIOMASS_XSD.joinpath("bio-l1-annotations.xsd"),
    namespace="transcoder",
    subpackage="common_annotation_models_l1",
    description="XSD common annotation models L1",
)

XSD_BIO_L2_ANNOTATIONS_COMMON = XSDInfo(
    xsd_path=BIOMASS_XSD.joinpath("bio-l2l3-common-annotations.xsd"),
    namespace="transcoder",
    subpackage="common_annotation_models_l2",
    description="XSD common annotation models L2",
)

XSD_FILES_L1_PROCESSOR = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-pp1.xsd"),
        namespace="l1_processor",
        subpackage="aux_pp1_models",
        description="XSD PP1 models",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-parc-info.xsd"),
        namespace="l1_processor",
        subpackage="parc_info_models",
        description="XSD PARC info models",
    ),
]

XSD_FILES_L2A_PROCESSOR = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-pp2_2a.xsd"),
        namespace="l2a_processor",
        subpackage="aux_pp2_2a_models",
        description="XSD PP2 2A models",
    ),
]

XSD_FILES_L2B_FD_PROCESSOR = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-pp2_fd.xsd"),
        namespace="l2b_fd_processor",
        subpackage="aux_pp2_2b_fd_models",
        description="XSD PP2 2B FD models",
    ),
]

XSD_FILES_L2B_FH_PROCESSOR = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-pp2_fh.xsd"),
        namespace="l2b_fh_processor",
        subpackage="aux_pp2_2b_fh_models",
        description="XSD PP2 2B FH models",
    ),
]

XSD_FILES_L2B_AGB_PROCESSOR = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-pp2_ab.xsd"),
        namespace="l2b_agb_processor",
        subpackage="aux_pp2_2b_agb_models",
        description="XSD PP2 2B AGB models",
    ),
]

XSD_FILES_STACK_PROCESSOR = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-pps.xsd"),
        namespace="stack_processor",
        subpackage="aux_pps_models",
        description="XSD PPS models",
    ),
]

XSD_FILES_AUX_PRODUCTS = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-orbit.xsd"),
        namespace="transcoder",
        subpackage="aux_orb_models",
        description="XSD Orbit models",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-attitude.xsd"),
        namespace="transcoder",
        subpackage="aux_att_models",
        description="XSD Attitude models",
    ),
]

XSD_FILES_TRANSCODER_L1 = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l1ab-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l1ab",
        description="XSD Main annotation models l1ab",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l1c-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l1c",
        description="XSD Main annotation models l1c",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l1-overlay.xsd"),
        namespace="transcoder",
        subpackage="overlay",
        description="XSD Overlay models",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l1-vrt.xsd"),
        namespace="transcoder",
        subpackage="vrt",
        description="XSD VRT models",
    ),
]

XSD_FILES_TRANSCODER_L2 = [
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2a-fd-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l2a_fd",
        description="XSD Main annotation models L2a FD",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2a-fh-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l2a_fh",
        description="XSD Main annotation models L2a FH",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2a-gn-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l2a_gn",
        description="XSD Main annotation models L2a GN",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2a-tfh-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l2a_tfh",
        description="XSD Main annotation models L2a TOMO FH",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2l3-fd-proc-annotations.xsd"),
        namespace="transcoder",
        subpackage="proc_params_annotation_models_l2_fd",
        description="XSD processing parameters annotation models L2 FD",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2l3-fh-proc-annotations.xsd"),
        namespace="transcoder",
        subpackage="proc_params_annotation_models_l2_fh",
        description="XSD processing parameters annotation models L2 FH",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2l3-tfh-proc-annotations.xsd"),
        namespace="transcoder",
        subpackage="proc_params_annotation_models_l2_tomo_fh",
        description="XSD processing parameters annotation models L2 TOMO FH",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2l3-agb-proc-annotations.xsd"),
        namespace="transcoder",
        subpackage="proc_params_annotation_models_l2_agb",
        description="XSD processing parameters annotation models L2 AGB",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2b-fd-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l2b_fd",
        description="XSD Main annotation models L2b FD",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2b-fh-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l2b_fh",
        description="XSD Main annotation models L2b FH",
    ),
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-l2b-agb-main-annotation.xsd"),
        namespace="transcoder",
        subpackage="main_annotation_models_l2b_agb",
        description="XSD Main annotation models L2b AGB",
    ),
]

XSD_AUX_INS = (
    XSDInfo(
        xsd_path=BIOMASS_XSD.joinpath("bio-aux-ins.xsd"),
        namespace="l1_pre_processor",
        subpackage="aux_ins_models",
        description="XSD INS models",
    ),
)

XSDATA_CONFIG_FILE = XSD_SOURCE_DIR.joinpath(".xsdata.xml")
XSD_FILES_BIOMASS = [
    XSD_BIO_COMMON_TYPES,
    XSD_BIO_L1_ANNOTATIONS_COMMON,
    XSD_BIO_L2_ANNOTATIONS_COMMON,
]
XSD_FILES_BIOMASS += XSD_FILES_AUX_PRODUCTS
XSD_FILES_BIOMASS += XSD_FILES_L1_PROCESSOR
XSD_FILES_BIOMASS += XSD_FILES_STACK_PROCESSOR
XSD_FILES_BIOMASS += XSD_FILES_L2A_PROCESSOR
XSD_FILES_BIOMASS += XSD_FILES_L2B_FD_PROCESSOR
XSD_FILES_BIOMASS += XSD_FILES_L2B_FH_PROCESSOR
XSD_FILES_BIOMASS += XSD_FILES_L2B_AGB_PROCESSOR
XSD_FILES_BIOMASS += XSD_FILES_TRANSCODER_L1
XSD_FILES_BIOMASS += XSD_FILES_TRANSCODER_L2
XSD_FILES_BIOMASS += XSD_AUX_INS

XSD_FILES: list[XSDInfo] = []
XSD_FILES += XSD_FILES_BIOMASS
# XSD_FILES += XSD_FILES_BRICKSAR
# XSD_FILES += XSD_FILES_ICD


def drop_all_list_from_init_file(init_file: Path):
    """Drop __all__ from init file"""
    init_file.write_text(init_file.read_text().split("__all__")[0])


def prepend_license(xsd_list: Iterable[XSDInfo]):
    """Add license information on top of files"""
    for xsd_file in xsd_list:
        for file in [xsd_file.models_file, xsd_file.init_file]:
            file.write_text(
                LICENSE_HEADER + xsd_file.module_docstring + file.read_text(encoding="utf-8"),
                encoding="utf-8",
            )


def move_types_to_common_models(xsd_common: XSDInfo, dependent_files: Iterable[XSDInfo]):
    """Check if new types defined in the dependent files are actually from a common xsd.
    if so, move the definition to the common models"""
    common_types_list = get_simple_types_from_xsd(xsd_common.xsd_path)
    for file in dependent_files:
        move_definitions_to_common_file(
            xsd_common.models_file,
            xsd_common.init_file,
            file.models_file,
            common_types_list,
        )

        remove_duplicate_definitions(xsd_common.models_file, file.models_file)


def generate_xsd_models(session: nox.Session):
    """XSD Models generation core function"""
    for xsd_file in XSD_FILES:
        with working_directory(xsd_file.namespace_base_dir):
            command = (
                "xsdata generate"
                + f" --package {xsd_file.package}"
                + " --structure-style single-package"
                + f" --config {XSDATA_CONFIG_FILE}"
                + f" {xsd_file.xsd_path}"
            )

            session.run(*command.split())

        # BPS is a namespace: it should not contain __init__.py, but xsdata generates it
        extra_init_file = xsd_file.namespace_base_dir.joinpath("bps", "__init__.py")
        assert extra_init_file.read_text(encoding="utf-8") == "# nothing here\n"
        extra_init_file.unlink()

        drop_all_list_from_init_file(xsd_file.init_file)

    move_types_to_common_models(
        XSD_BIO_COMMON_TYPES,
        filter(
            lambda xsd_file: (
                xsd_file.subpackage
                not in (
                    "common_types",
                    "vrt",
                )
            ),
            XSD_FILES_BIOMASS,
        ),
    )

    move_types_to_common_models(
        XSD_BIO_L1_ANNOTATIONS_COMMON,
        XSD_FILES_TRANSCODER_L1,
    )

    move_types_to_common_models(
        XSD_BIO_L2_ANNOTATIONS_COMMON,
        XSD_FILES_TRANSCODER_L2,
    )

    prepend_license(XSD_FILES)

    files = [str(xsd_file.models_file) for xsd_file in XSD_FILES]
    files += [str(xsd_file.init_file) for xsd_file in XSD_FILES]

    session.run("python", "-m", "ruff", "format", *files)
    session.run("python", "-m", "ruff", "check", *files, "--select", "I", "--fix")


def index_from_right(input_list: list, value: Any) -> int:
    """Find last occurrence of value in input_list"""
    reverse_index = input_list[::-1].index(value)
    return len(input_list) - 1 - reverse_index


def retrieve_module_import(file: Path) -> str:
    """Retrieve module ast"""
    file_split = str(file.parent).split(os.sep)
    file_split = file_split[index_from_right(file_split, "bps") :]
    return ".".join(file_split)


def get_classes(module_ast: ast.Module) -> list[ast.ClassDef]:
    """Get all class definitions"""
    return list(filter(lambda element: isinstance(element, ast.ClassDef), module_ast.body))  # type: ignore


def get_simple_types_from_xsd(xsd_file: Path) -> list[str]:
    """Retrieve all simple types definition from an XSD file"""
    text = xsd_file.read_text(encoding="utf-8")
    list_types = []
    for line in text.splitlines():
        if "<xsd:simpleType" in line:
            list_types.append(line.split('"')[1].lower().replace("_", ""))
    return list_types


def move_definitions_to_common_file(common_module_file, common_module_init_file, module_file, class_to_move):
    """Check if classes defined in module_file are in the list class_to_move
    if so, move the definition to the common models"""
    common_ast = ast.parse(common_module_file.read_text(encoding="utf8"))
    common_classes = get_classes(common_ast)

    common_classes_names = [c.name for c in common_classes]

    module_ast = ast.parse(module_file.read_text(encoding="utf8"))
    module_classes = get_classes(module_ast)

    module_classes_to_be_moved = filter(
        lambda element: str(element.name).lower() in class_to_move and element.name not in common_classes_names,
        module_classes,
    )

    common_module_import = retrieve_module_import(common_module_file) + ".models"
    common_module_init_ast = ast.parse(common_module_init_file.read_text(encoding="utf8"))
    for class_definition in module_classes_to_be_moved:
        common_ast.body.insert(-1, class_definition)

        import_from_common_line = ast.ImportFrom(common_module_import, [ast.alias(class_definition.name)], level=0)
        common_module_init_ast.body.insert(0, import_from_common_line)

    ast.fix_missing_locations(common_ast)
    ast.fix_missing_locations(common_module_init_ast)

    modified_text = restore_original_formatting(ast.unparse(common_ast))
    Path(common_module_file).write_text(modified_text, encoding="utf-8")

    modified_text = restore_original_formatting(ast.unparse(common_module_init_ast))
    Path(common_module_init_file).write_text(modified_text, encoding="utf-8")


def remove_duplicate_definitions(common_module_file, module_file):
    """Replace duplicated class definitions with import statements"""
    common_module_import = retrieve_module_import(common_module_file)

    common_ast = ast.parse(common_module_file.read_text(encoding="utf8"))
    common_classes = get_classes(common_ast)

    remove_classes_names = [c.name for c in common_classes]

    module_ast = ast.parse(module_file.read_text(encoding="utf8"))
    module_classes = get_classes(module_ast)

    module_classes_to_be_removed = filter(lambda element: element.name in remove_classes_names, module_classes)

    for class_definition in module_classes_to_be_removed:
        module_ast.body.remove(class_definition)
        import_from_common_line = ast.ImportFrom(common_module_import, [ast.alias(class_definition.name)], level=0)
        module_ast.body.insert(0, import_from_common_line)

    ast.fix_missing_locations(module_ast)

    modified_text = restore_original_formatting(ast.unparse(module_ast))

    Path(module_file).write_text(modified_text, encoding="utf-8")


def restore_original_formatting(module: str) -> str:
    """AST module slightly change the formats, here it is restored before calling ruff formatter"""
    pattern_list = [
        "(false)|(true)",
        "UTC=.*",
        "UT1=.*",
        "TAI=.*",
        "BIOMASS",
        "OPER",
        "(bio_aux_att____.+)|(bio_.+_att)",
        "(bio_aux_orb____.+)|(bio_.+_orb)",
        "AUX_ATT___",
        "AUX_ORB___",
        "[0-9]{4}",
    ]
    pattern_replace = [("'pattern': '" + pattern + "'", "'pattern': r'" + pattern + "'") for pattern in pattern_list]

    common_part = "(((01|03|05|07|08|10|12)-(0[1-9]|[1,2][0-9]|3[0,1]))|((04|06|09|11)"
    common_part += "-(0[1-9]|[1,2][0-9]|30))|(02-(0[1-9]|[1,2][0-9])))T([0,1][0-9]|2[0-3])(:[0-5][0-9]){2}"
    regular_expression = r"'[A-Z0-9]{3}=(\\d{4}-" + common_part + r"|0000-00-00T00:00:00|9999-99-99T99:99:99)(.\\d*)?'"

    regular_expression_fixed = (
        r"r'[A-Z0-9]{3}=(\d{4}-" + common_part + r"|0000-00-00T00:00:00|9999-99-99T99:99:99)(.\d*)?'"
    )

    replace_list = [
        ("'required': True}", "'required': True,},"),
        ("'required': False}", "'required': False,},"),
        ("'type': 'Element'}", "'type': 'Element',},"),
        ("'type': 'Attribute'}", "'type': 'Attribute',},"),
        ("'min_occurs': 1}", "'min_occurs': 1,},"),
        ("'max_occurs': 5}", "'max_occurs': 5,},"),
        ("'namespace': ''}", "'namespace': '',},"),
        (regular_expression, regular_expression_fixed),
    ]
    replace_list += pattern_replace

    for replace_item in replace_list:
        module = module.replace(*replace_item)

    return module
