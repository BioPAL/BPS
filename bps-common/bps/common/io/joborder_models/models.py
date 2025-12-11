# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Job Order models
--------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from xsdata.models.datatype import XmlDateTime


@dataclass
class AoiType:
    """
    Area of interest of the processing request.

    Parameters
    ----------
    geometry
        List of coordinates defining the polygon of the requested area to be processed. The coordinates are
        specified using latitude, longitude pairs in a counter-clockwise orientation with last point equal to first
        point. E.g.: <Geometry xmlns=""> -8.015716 -63.764648 -6.809171 -63.251038 -6.967323 -62.789612 -8.176149
        -63.278503 -8.015716 -63.764648 </Geometry>
    """

    class Meta:
        name = "AOI_Type"

    geometry: Optional[str] = field(
        default=None,
        metadata={
            "name": "Geometry",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class AlternativeId:
    """Alternative identifier.

    This value shall be copied into the Alternative_ID field of the Job Order. The Task’s executable shall be
    able to recognize it.
    """

    class Meta:
        name = "Alternative_ID"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class Baseline:
    """Baseline ID (2 zero-padded digits) of the output product to be generated.

    It is set by the PF and shall be used by the processor to assign the baseline of the output product.
    """

    value: str = field(
        default="",
        metadata={
            "pattern": r"[0-9]{2}",
        },
    )


@dataclass
class CfgId:
    """Identifier of the configuration file to be passed to the Task executable via
    the Job Order.

    This ID is a helper for a Task to identify configuration files of a given category (e.g. DEM, …)
    """

    class Meta:
        name = "Cfg_ID"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class DiskSpace:
    """
    Minimum amount of disk space, in megabytes, required on the processing node in
    order to complete the processing.
    """

    class Meta:
        name = "Disk_Space"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        },
    )


@dataclass
class FileType:
    """File Class for this production request.

    It is set by the PF and shall be used by the processor to assign the file class to the output products.
    """

    class Meta:
        name = "File_Class"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class FileDir:
    """Pathname of the directory where the output products shall be written by the
    processor and from where the PF shall retrieve them.

    The path can be an absolute or a relative one. In case of a relative path it is considered relative to the
    working directory. If File_Dir is empty or not present, it is assumed that this directory corresponds to the
    working directory.
    """

    class Meta:
        name = "File_Dir"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class FileName:
    """
    Absolute path of the input file.
    """

    class Meta:
        name = "File_Name"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class FileNamePattern:
    """Expression that univocally identifies the output product (or products) among
    the files available within the working directory.

    This field is copied into the corresponding “File_Name_Pattern” field of the Job Order for information
    (Wildcards are allowed).
    """

    class Meta:
        name = "File_Name_Pattern"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class FileType:
    """
    A valid product file type (see the products naming convention of the specific
    mission).
    """

    class Meta:
        name = "File_Type"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class InputId:
    """Input identifier.

    The PF shall copy this value into the Input_ID filed of the Job Order. The Task’s executable shall be able
    to recognize it.
    """

    class Meta:
        name = "Input_ID"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class IntermediateOutputId:
    """Identifier of the intermediate output.

    The Task’s executable shall be able to recognize it.
    """

    class Meta:
        name = "Intermediate_Output_ID"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


class ListOfStderrLogLevelsStderrLogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    PROGRESS = "PROGRESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ListOfStdoutLogLevelsStdoutLogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    PROGRESS = "PROGRESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class MetadataParameterTypeToBeRemoved:
    """
    Parameters
    ----------
    name
        Name of the metadata parameter.
    value
        Value of the metadata parameter.
    """

    class Meta:
        name = "Metadata_Parameter_Type_TO-BE-REMOVED"

    name: Optional[str] = field(
        default=None,
        metadata={
            "name": "Name",
            "type": "Element",
            "required": True,
        },
    )
    value: Optional[str] = field(
        default=None,
        metadata={
            "name": "Value",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class ParameterType:
    """
    Parameters
    ----------
    name
        Name of the parameter.
    value
        Value of the parameter.
    """

    class Meta:
        name = "Parameter_Type"

    name: Optional[str] = field(
        default=None,
        metadata={
            "name": "Name",
            "type": "Element",
            "required": True,
        },
    )
    value: Optional[str] = field(
        default=None,
        metadata={
            "name": "Value",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class ProcessorName:
    """
    Name of the processor to which this Task Table refers.
    """

    class Meta:
        name = "Processor_Name"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class ProcessorVersion:
    """Version of the processor.

    Format: 4 zero-padded digits for issue and revision separated by “.” (e.g. “01.02”).
    """

    class Meta:
        name = "Processor_Version"

    value: str = field(
        default="",
        metadata={
            "pattern": r"[0-9]{2}\.[0-9]{2}",
        },
    )


@dataclass
class RamdiskType:
    """
    Parameters
    ----------
    amount
        Minimum size of the RAM disk, in megabytes, that is required by the Task executable in order to meet the
        nominal performance requirement. If the following Scalable parameter is set to “Yes”, the PF is allowed to
        increase or decrease this value applying a scaling factor and to specify, in the Job Order, the updated
        maximum amount of RAM disk that the Task is allowed to use. In such a case, the same scaling factor will be
        applied to all Tasks where the Scalable flag is set to “Yes”.
    mount_path
        Absolute mount path of the RAM disk. The value is copied by the PF into the same field of the Job Order
    """

    class Meta:
        name = "RAMDISK_Type"

    amount: Optional[int] = field(
        default=None,
        metadata={
            "name": "Amount",
            "type": "Element",
            "required": True,
        },
    )
    mount_path: Optional[str] = field(
        default=None,
        metadata={
            "name": "Mount_Path",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class RamdiskTypeScalable:
    """
    Parameters
    ----------
    amount
        Minimum size of the RAM disk, in megabytes, that is required by the Task executable in order to meet the
        nominal performance requirement. If the following Scalable parameter is set to “Yes”, the PF is allowed to
        increase or decrease this value applying a scaling factor and to specify, in the Job Order, the updated
        maximum amount of RAM disk that the Task is allowed to use. In such a case, the same scaling factor will be
        applied to all Tasks where the Scalable flag is set to “Yes”.
    scalable
        Scalability flag for the maximum amount of RAM disk. Possible values are: * “Yes”: the PF shall apply a
        scaling factor to the maximum amount of RAM disk defined by the Amount parameter and the updated value shall
        be copied into the Job Order * “No”: the scaling factor is not applied. The original value is copied by the
        PF into the Job Order
    mount_path
        Absolute mount path of the RAM disk. The value is copied by the PF into the same field of the Job Order
    """

    class Meta:
        name = "RAMDISK_Type_Scalable"

    amount: Optional[int] = field(
        default=None,
        metadata={
            "name": "Amount",
            "type": "Element",
            "required": True,
        },
    )
    scalable: Optional[str] = field(
        default=None,
        metadata={
            "name": "Scalable",
            "type": "Element",
            "required": True,
        },
    )
    mount_path: Optional[str] = field(
        default=None,
        metadata={
            "name": "Mount_Path",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class ToiType:
    """
    Time of interest of the processing request.

    Parameters
    ----------
    start
        UTC start time of the processing request. String value has the format: YYYY-MM-DDThh:mm:ss.uuuuuu
    stop
        UTC stop time of the processing request. String value has the format: YYYY-MM-DDThh:mm:ss.uuuuuu
    """

    class Meta:
        name = "TOI_Type"

    start: Optional[XmlDateTime] = field(
        default=None,
        metadata={
            "name": "Start",
            "type": "Element",
            "required": True,
        },
    )
    stop: Optional[XmlDateTime] = field(
        default=None,
        metadata={
            "name": "Stop",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class TaskName:
    """Name of the Task.

    Within a Task Table, Task_Name shall be unique. The name is copied by the PF into the Job Order and shall be
    used, by the Task executable, to identify its own section (i.e. the Task element having the same Task_Name)
    within the Job Order.
    """

    class Meta:
        name = "Task_Name"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )


@dataclass
class TaskVersion:
    """Version of the Task.

    Format: 4 zero-padded digits for issue and revision separated by “.” (e.g. “01.02”).
    It may be used by a Task executable for consistency check (by ensuring that the Task’s hardcoded version corresponds to the one executed in the Job Order).
    """

    class Meta:
        name = "Task_Version"

    value: str = field(
        default="",
        metadata={
            "pattern": r"[0-9]{2}\.[0-9]{2}",
        },
    )


@dataclass
class CfgFileType:
    """
    Parameters
    ----------
    cfg_id
    cfg_file_name
        Configuration file to be passed to the task executable via the Job Order. The path must be an absolute one,
        or relative to the root directory of the processor installation tree.
    """

    class Meta:
        name = "Cfg_File_Type"

    cfg_id: Optional[CfgId] = field(
        default=None,
        metadata={
            "name": "Cfg_ID",
            "type": "Element",
            "required": True,
        },
    )
    cfg_file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "Cfg_File_Name",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class JoIntermediateOutputType:
    """
    Parameters
    ----------
    intermediate_output_id
    intermediate_output_file
        Absolute path of the intermediate output file.
    """

    class Meta:
        name = "JO_Intermediate_Output_Type"

    intermediate_output_id: Optional[IntermediateOutputId] = field(
        default=None,
        metadata={
            "name": "Intermediate_Output_ID",
            "type": "Element",
            "required": True,
        },
    )
    intermediate_output_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "Intermediate_Output_File",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class JoOutputType:
    class Meta:
        name = "JO_Output_Type"

    file_type: Optional[FileType] = field(
        default=None,
        metadata={
            "name": "File_Type",
            "type": "Element",
            "required": True,
        },
    )
    file_name_pattern: Optional[FileNamePattern] = field(
        default=None,
        metadata={
            "name": "File_Name_Pattern",
            "type": "Element",
            "required": True,
        },
    )
    file_dir: Optional[FileDir] = field(
        default=None,
        metadata={
            "name": "File_Dir",
            "type": "Element",
        },
    )
    baseline: Optional[Baseline] = field(
        default=None,
        metadata={
            "name": "Baseline",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class ListOfMetadataParametersType:
    """
    List of product metadata parameters defining the processing request.

    Parameters
    ----------
    metadata_parameter
        Metadata parameter.
    """

    class Meta:
        name = "List_of_Metadata_Parameters_Type"

    metadata_parameter: list[ParameterType] = field(
        default_factory=list,
        metadata={
            "name": "Metadata_Parameter",
            "type": "Element",
        },
    )


@dataclass
class RequestTypeToBeRemoved:
    """
    Parameters
    ----------
    toi
        Time of interest of the processing request.
    aoi
        Area of interest of the processing request.
    list_of_metadata_parameters
        List of product metadata parameters defining the processing request.
    """

    class Meta:
        name = "Request_Type_TO-BE-REMOVED"

    toi: Optional[ToiType] = field(
        default=None,
        metadata={
            "name": "TOI",
            "type": "Element",
        },
    )
    aoi: Optional[AoiType] = field(
        default=None,
        metadata={
            "name": "AOI",
            "type": "Element",
        },
    )
    list_of_metadata_parameters: Optional["RequestTypeToBeRemoved.ListOfMetadataParameters"] = field(
        default=None,
        metadata={
            "name": "List_of_Metadata_Parameters",
            "type": "Element",
        },
    )

    @dataclass
    class ListOfMetadataParameters:
        """
        Parameters
        ----------
        metadata_parameter
            Metadata parameter.
        """

        metadata_parameter: list[ParameterType] = field(
            default_factory=list,
            metadata={
                "name": "Metadata_Parameter",
                "type": "Element",
            },
        )


@dataclass
class SelectedInputType:
    """
    Parameters
    ----------
    file_type
    list_of_file_names
        List of File_Name elements.
    """

    class Meta:
        name = "Selected_Input_Type"

    file_type: Optional[FileType] = field(
        default=None,
        metadata={
            "name": "File_Type",
            "type": "Element",
            "required": True,
        },
    )
    list_of_file_names: Optional["SelectedInputType.ListOfFileNames"] = field(
        default=None,
        metadata={
            "name": "List_of_File_Names",
            "type": "Element",
            "required": True,
        },
    )

    @dataclass
    class ListOfFileNames:
        file_name: list[FileName] = field(
            default_factory=list,
            metadata={
                "name": "File_Name",
                "type": "Element",
                "min_occurs": 1,
            },
        )


@dataclass
class JoInputType:
    class Meta:
        name = "JO_Input_Type"

    input_id: Optional[InputId] = field(
        default=None,
        metadata={
            "name": "Input_ID",
            "type": "Element",
            "required": True,
        },
    )
    alternative_id: Optional[AlternativeId] = field(
        default=None,
        metadata={
            "name": "Alternative_ID",
            "type": "Element",
            "required": True,
        },
    )
    list_of_selected_inputs: Optional["JoInputType.ListOfSelectedInputs"] = field(
        default=None,
        metadata={
            "name": "List_of_Selected_Inputs",
            "type": "Element",
            "required": True,
        },
    )

    @dataclass
    class ListOfSelectedInputs:
        selected_input: list[SelectedInputType] = field(
            default_factory=list,
            metadata={
                "name": "Selected_Input",
                "type": "Element",
                "min_occurs": 1,
            },
        )


@dataclass
class RequestType:
    class Meta:
        name = "Request_Type"

    toi: Optional[ToiType] = field(
        default=None,
        metadata={
            "name": "TOI",
            "type": "Element",
            "required": True,
        },
    )
    aoi: list[AoiType] = field(
        default_factory=list,
        metadata={
            "name": "AOI",
            "type": "Element",
            "max_occurs": 2,
        },
    )
    list_of_metadata_parameters: list[ListOfMetadataParametersType] = field(
        default_factory=list,
        metadata={
            "name": "List_of_Metadata_Parameters",
            "type": "Element",
            "max_occurs": 3,
        },
    )


@dataclass
class JoTaskType:
    """
    Parameters
    ----------
    task_name
    task_version
    number_of_cpu_cores
        Maximum Number of CPU cores that the Task executable is allowed to use for the processing. The executable
        shall ensure that the number of used cores does not exceed this limit. Depending of the PF implementation
        (e.g. use of virtual machines or containers), this value may also correspond to a physical limit set by the
        host. The nominal number of cores required by the Task is defined in the relevant CPU_Cores/Number element
        of the Task Table. In case the parameter CPU_Cores/Scalable of the Task Table is set to “Yes”, the PF can
        apply a scaling factor to that nominal value, resulting in a lower or higher number of cores that can be
        used by the Task. If the parameter CPU_Cores/Scalable is instead set to “No”, then the value of
        Number_of_CPU_Cores is copied as it is from the CPU_Cores/Number parameter of the Task Table. The value “0”
        is also allowed for this field, meaning that the amount of used CPU cores is negligible.
    amount_of_ram
        Maximum amount of RAM  in megabytes  that the Task executable is allowed to use for the processing. The
        executable shall ensure that the used amount of RAM does not exceed this limit. Depending of the PF
        implementation (e.g. use of virtual machines or containers), this value may also correspond to a physical
        limit set by the host. The nominal amount of RAM required by the Task is defined in the relevant RAM/Amount
        element of the Task Table. In case the parameter RAM/Scalable of the Task Table is set to “Yes”, the PF can
        apply a scaling factor to that nominal value, resulting in a lower or higher amount of RAM that can be used
        by the Task. If the parameter RAM/Scalable is instead set to “No”, then the value of Amount_of_RAM is copied
        as it is from the RAM/Amount parameter of the Task Table.
    list_of_ramdisks
        List of RAM-based volumes.
    number_of_gpu_cores
        Number of GPU cores in that the Task executable is allowed to use for the processing. The executable shall
        ensure that the used amount of GPUs does not exceed this limit. Depending of the PF implementation (e.g. use
        of virtual machines or containers), this value may also correspond to a physical limit set by the host. The
        nominal amount of GPUs required by the Task is defined in the relevant GPU/Cores/Number element of the Task
        Table. In case the parameter GPU/Cores/Scalable of the Task Table is set to “Yes”, the PF can apply a
        scaling factor to that nominal value, resulting in a lower or higher amount of GPU cores that can be used by
        the Task. If the parameter GPU/Cores/Scalable is instead set to “No”, then the value of Number_of_GPU is
        copied as it is from the GPU/Cores/Number parameter of the Task Table. This element shall always be defined
        if the Task_Table Task.GPU element is defined and will be set to "0" if Mandatory flsg is set to "false" and
        no GPU is assigned to the Task.
    amount_of_gpu_ram
        Maximum amount of GPU RAM in megabytes  that the Task executable is allowed to use for the processing. The
        executable shall ensure that the used amount of GPU RAM does not exceed this limit. Depending of the PF
        implementation (e.g. use of virtual machines or containers), this value may also correspond to a physical
        limit set by the host. The nominal amount of GPU RAM required by the Task is defined in the relevant
        GPU/RAM/Amount element of the Task Table. In case the parameter GPU/RAM/Scalable of the Task Table is set to
        “Yes”, the PF can apply a scaling factor to that nominal value, resulting in a lower or higher amount of GPU
        RAM that can be used by the Task. If the parameter GPU/RAM/Scalable is instead set to “No”, then the value
        of Amount_of_RAM is copied as it is from the GPU/RAM/Amount parameter of the Task Table. This element shall
        always be defined if the Task_Table Task.GPU element is defined and will be set to "0" if Mandatory flsg is
        set to "false" and no GPU is assigned to the Task.
    disk_space
    list_of_proc_parameters
        List of processing parameters copied from the Task Table or possibly overruled by the PF (e.g. from a manual
        request) depending on PF capability.
    list_of_cfg_files
        List of Cfg_File elements copied from the corresponding list of the Task Table.
    list_of_inputs
        List of Input elements.
    list_of_outputs
        List of Output elements.
    list_of_intermediate_outputs
        List of intermediate  outputs. This parameter is considered only if the Intermediate_Output_Enable flag is
        set to “true”.
    """

    class Meta:
        name = "JO_Task_Type"

    task_name: Optional[TaskName] = field(
        default=None,
        metadata={
            "name": "Task_Name",
            "type": "Element",
            "required": True,
        },
    )
    task_version: Optional[TaskVersion] = field(
        default=None,
        metadata={
            "name": "Task_Version",
            "type": "Element",
            "required": True,
        },
    )
    number_of_cpu_cores: Optional[float] = field(
        default=None,
        metadata={
            "name": "Number_of_CPU_Cores",
            "type": "Element",
            "required": True,
        },
    )
    amount_of_ram: Optional[int] = field(
        default=None,
        metadata={
            "name": "Amount_of_RAM",
            "type": "Element",
            "required": True,
        },
    )
    list_of_ramdisks: Optional["JoTaskType.ListOfRamdisks"] = field(
        default=None,
        metadata={
            "name": "List_of_RAMDISKs",
            "type": "Element",
        },
    )
    number_of_gpu_cores: Optional[int] = field(
        default=None,
        metadata={
            "name": "Number_of_GPU_Cores",
            "type": "Element",
        },
    )
    amount_of_gpu_ram: Optional[int] = field(
        default=None,
        metadata={
            "name": "Amount_of_GPU_RAM",
            "type": "Element",
        },
    )
    disk_space: Optional[DiskSpace] = field(
        default=None,
        metadata={
            "name": "Disk_Space",
            "type": "Element",
            "required": True,
        },
    )
    list_of_proc_parameters: Optional["JoTaskType.ListOfProcParameters"] = field(
        default=None,
        metadata={
            "name": "List_of_Proc_Parameters",
            "type": "Element",
            "required": True,
        },
    )
    list_of_cfg_files: Optional["JoTaskType.ListOfCfgFiles"] = field(
        default=None,
        metadata={
            "name": "List_of_Cfg_Files",
            "type": "Element",
            "required": True,
        },
    )
    list_of_inputs: Optional["JoTaskType.ListOfInputs"] = field(
        default=None,
        metadata={
            "name": "List_of_Inputs",
            "type": "Element",
            "required": True,
        },
    )
    list_of_outputs: Optional["JoTaskType.ListOfOutputs"] = field(
        default=None,
        metadata={
            "name": "List_of_Outputs",
            "type": "Element",
            "required": True,
        },
    )
    list_of_intermediate_outputs: Optional["JoTaskType.ListOfIntermediateOutputs"] = field(
        default=None,
        metadata={
            "name": "List_of_Intermediate_Outputs",
            "type": "Element",
            "required": True,
        },
    )

    @dataclass
    class ListOfRamdisks:
        """
        Parameters
        ----------
        ramdisk
            Specification of the required RAM-based volumes.
        """

        ramdisk: list[RamdiskType] = field(
            default_factory=list,
            metadata={
                "name": "RAMDISK",
                "type": "Element",
            },
        )

    @dataclass
    class ListOfProcParameters:
        """
        Parameters
        ----------
        proc_parameter
            By default, processing parameters are copied from the Task Table.
        """

        proc_parameter: list[ParameterType] = field(
            default_factory=list,
            metadata={
                "name": "Proc_Parameter",
                "type": "Element",
            },
        )

    @dataclass
    class ListOfCfgFiles:
        """
        Parameters
        ----------
        cfg_file
            Task’s configuration.
        """

        cfg_file: list[CfgFileType] = field(
            default_factory=list,
            metadata={
                "name": "Cfg_File",
                "type": "Element",
            },
        )

    @dataclass
    class ListOfInputs:
        """
        Parameters
        ----------
        input
            Description of the Task’s input data.
        """

        input: list[JoInputType] = field(
            default_factory=list,
            metadata={
                "name": "Input",
                "type": "Element",
                "min_occurs": 1,
            },
        )

    @dataclass
    class ListOfOutputs:
        """
        Parameters
        ----------
        output
            Description of the Task’s output data.
        """

        output: list[JoOutputType] = field(
            default_factory=list,
            metadata={
                "name": "Output",
                "type": "Element",
                "min_occurs": 1,
            },
        )

    @dataclass
    class ListOfIntermediateOutputs:
        """
        Parameters
        ----------
        intermediate_output
            Task’s enabled breakpoint.
        """

        intermediate_output: list[JoIntermediateOutputType] = field(
            default_factory=list,
            metadata={
                "name": "Intermediate_Output",
                "type": "Element",
            },
        )


@dataclass
class ProcessorConfigurationType:
    """
    Parameters
    ----------
    file_class
    processor_name
    processor_version
    processing_node
        Identifier of the node where the processor is running. This must be used by the processor to identify the
        node in the messages generated by the logging interface.
    list_of_stdout_log_levels
        List of log levels allowed to be emitted to the standard output stream.
    list_of_stderr_log_levels
        List of log levels allowed to be emitted to the standard error stream.
    intermediate_output_enable
        Breakpoint enabling flag: •       “true”: the breakpoints are enabled. •       “false”: the breakpoints are
        disabled. Note that this flag will enable/disable any kind of breakpoint functionality generation
        implemented by the processor (via Job Order or other processor internal mechanisms).
    processing_station
        Name or identifier of the processing facility hosting the processor.
    request
        This field specifies the request criteria. Child elements are optional but at least one of them should be
        present.
    """

    class Meta:
        name = "Processor_Configuration_Type"

    file_class: Optional[FileType] = field(
        default=None,
        metadata={
            "name": "File_Class",
            "type": "Element",
            "required": True,
        },
    )
    processor_name: Optional[ProcessorName] = field(
        default=None,
        metadata={
            "name": "Processor_Name",
            "type": "Element",
            "required": True,
        },
    )
    processor_version: Optional[ProcessorVersion] = field(
        default=None,
        metadata={
            "name": "Processor_Version",
            "type": "Element",
            "required": True,
        },
    )
    processing_node: Optional[str] = field(
        default=None,
        metadata={
            "name": "Processing_Node",
            "type": "Element",
            "required": True,
        },
    )
    list_of_stdout_log_levels: Optional["ProcessorConfigurationType.ListOfStdoutLogLevels"] = field(
        default=None,
        metadata={
            "name": "List_of_Stdout_Log_Levels",
            "type": "Element",
            "required": True,
        },
    )
    list_of_stderr_log_levels: Optional["ProcessorConfigurationType.ListOfStderrLogLevels"] = field(
        default=None,
        metadata={
            "name": "List_of_Stderr_Log_Levels",
            "type": "Element",
            "required": True,
        },
    )
    intermediate_output_enable: Optional[bool] = field(
        default=None,
        metadata={
            "name": "Intermediate_Output_Enable",
            "type": "Element",
            "required": True,
        },
    )
    processing_station: Optional[str] = field(
        default=None,
        metadata={
            "name": "Processing_Station",
            "type": "Element",
            "required": True,
        },
    )
    request: Optional[RequestType] = field(
        default=None,
        metadata={
            "name": "Request",
            "type": "Element",
            "required": True,
        },
    )

    @dataclass
    class ListOfStdoutLogLevels:
        """
        Parameters
        ----------
        stdout_log_level
            Allowed logging level for the standard output stream. Possible values are: •       “DEBUG” •
            “INFO” •       “PROGRESS” •       “WARNING” •       “ERROR” Only messages having level equal to this
            value will be printed.
        """

        stdout_log_level: list[ListOfStdoutLogLevelsStdoutLogLevel] = field(
            default_factory=list,
            metadata={
                "name": "Stdout_Log_Level",
                "type": "Element",
                "max_occurs": 5,
            },
        )

    @dataclass
    class ListOfStderrLogLevels:
        """
        Parameters
        ----------
        stderr_log_level
            Allowed logging level for the standard error stream. Possible values are: •       “DEBUG” •       “INFO”
            •       “PROGRESS” •       “WARNING” •       “ERROR” Only messages having level equal to this value will
            be printed.
        """

        stderr_log_level: list[ListOfStderrLogLevelsStderrLogLevel] = field(
            default_factory=list,
            metadata={
                "name": "Stderr_Log_Level",
                "type": "Element",
                "max_occurs": 5,
            },
        )


@dataclass
class JobOrderType:
    """
    Parameters
    ----------
    processor_configuration
        Processor configuration.
    list_of_tasks
        List of Task elements.
    schema_name
    schema_version
    schema_location
    """

    class Meta:
        name = "Job_Order_Type"

    processor_configuration: Optional[ProcessorConfigurationType] = field(
        default=None,
        metadata={
            "name": "Processor_Configuration",
            "type": "Element",
            "required": True,
        },
    )
    list_of_tasks: Optional["JobOrderType.ListOfTasks"] = field(
        default=None,
        metadata={
            "name": "List_of_Tasks",
            "type": "Element",
            "required": True,
        },
    )
    schema_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "schemaName",
            "type": "Attribute",
        },
    )
    schema_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )
    schema_location: Optional[str] = field(
        default=None,
        metadata={
            "name": "schemaLocation",
            "type": "Attribute",
        },
    )

    @dataclass
    class ListOfTasks:
        """
        Parameters
        ----------
        task
            Information for the specific Task.
        """

        task: list[JoTaskType] = field(
            default_factory=list,
            metadata={
                "name": "Task",
                "type": "Element",
                "min_occurs": 1,
            },
        )


@dataclass
class JobOrder(JobOrderType):
    """
    Root element of the "Job Order" I/F XML document.
    """

    class Meta:
        name = "Job_Order"
