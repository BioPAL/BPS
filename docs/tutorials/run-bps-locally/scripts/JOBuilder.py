# -*- coding: utf-8 -*-
"""
Copyright 2025, European Space Agency (ESA)
Licensed under ESA Software Community Licence Permissive (Type 3) - v2.4
"""

from pathlib import Path
import sys
import os
from datetime import datetime
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta
import configparser
import BiomassProduct
import re
import os
from collections import defaultdict
from datetime import datetime
from datetime import datetime, time, timedelta

# Load configuration file (INI format)
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.ini')


config = configparser.ConfigParser()
with open(config_path, "r", encoding="utf-8") as f:
    content = f.read()

config.read(config_path)

# Load template paths from config file
TEMPLATE_MAP = {
    "L1F":          config.get('TEMPLATE_JO', 'L1F'),
    "L1":           config.get('TEMPLATE_JO', 'L1'),
    "L1_chain":     config.get('TEMPLATE_JO', 'L1'), 
    "STA":          config.get('TEMPLATE_JO', 'STA'),
    "STA_chain":    config.get('TEMPLATE_JO', 'STA'),    
    "L2A":          config.get('TEMPLATE_JO', 'L2A'),
    "L2A_chain":    config.get('TEMPLATE_JO', 'L2A'),
    "L2A_FH":       config.get('TEMPLATE_JO', 'L2A_FH')
}

AUX_STATIC = {
    "DEM":          config.get('AUX_STATIC', 'DEM'),      
    "GMF":          config.get('AUX_STATIC', 'GMF'),       
    "IRI":          config.get('AUX_STATIC', 'IRI') ,  
    "FNF":          config.get('AUX_STATIC', 'FNF'),
}

AUX = {
    "AUX_DEFAULT_DIR":       config.get('AUX', 'AUX_DEFAULT_DIR'),      
    "AUX_USER_DIR":          config.get('AUX', 'AUX_USER_DIR'),       
}



def load_config(filepath):


    config.optionxform = str 
    config.read(filepath)

    aux_files_versions = {
        section.strip(): {k.strip(): v.strip() for k, v in config.items(section)}
        for section in config.sections() if section.startswith("AUX_FILES_")
    }

    return aux_files_versions

aux_versions = load_config(config_path)



# Definition of required file to processing for each level 
REQUIRED_FILES = {
    'L1F': ['RAW__0S', 'RAW__0M','AUX_ORB___'],
    'L1': ['DEM','GMF','IRI','RAW__0S', 'RAW__0M','AUX_ORB___','AUX_ATT___','AUX_TEC___','AUX_INS___', 'AUX_PP1___'],
    'STA': ['FNF', 'SCS__1S', 'AUX_PPS___']
}


def extract_start_stop_times(filename):
    """
    Extract acquisition start and stop times from the filename.

    Args:
        filename (str): Product filename containing embedded timestamps.

    Returns:
        tuple: (start_time, stop_time) as datetime objects or (None, None) on failure.
    """   
    try:
        start_str = filename[15:29]
        stop_str = filename[31:46]
        
        print(start_str,stop_str)
        start = datetime.strptime(start_str, "%Y%m%dT%H%M%S")
        stop = datetime.strptime(stop_str, "%Y%m%dT%H%M%S")
        return start, stop
    except Exception as e:
        print(f"Errore nell'estrazione di start/stop da {filename}: {e}")
        return None, None

# Trova il file RAW__0M corrispondente al RAW__0S
def find_matching_raw_0m(raw_0s_file, RAW_0M_files, folder):
    """
    Match the RAW__0M file that fully contains the time span of a given RAW__0S file.

    Args:
        raw_0s_file (str): Filename of the RAW__0S product.
        raw_0m_files (list): List of RAW__0M filenames.
        folder (str): Directory containing the RAW__0M files.

    Returns:
        str or None: Full path to the matching RAW__0M file or None.
    """  
    margin = timedelta(seconds=20) 
    start_s, stop_s = extract_start_stop_times(raw_0s_file)
    if not start_s or not stop_s:
        return None
    for raw_0m_file in RAW_0M_files:
        start_m, stop_m = extract_start_stop_times(raw_0m_file)
        # Criterio: RAW__0M completamente incluso nel RAW__0S
        if  start_s >= start_m and stop_s <= stop_m+margin:
            
            return os.path.join(folder, raw_0m_file)
    return None


def find_matching(raw_0s_file, aux_files, folder):
    """
    Find auxiliary file (e.g., AUX_ORB___) whose time span overlaps with RAW__0S.

    Args:
        raw_0s_file (str): Filename of the RAW__0S product.
        aux_files (list): List of auxiliary filenames.
        folder (str): Directory containing the auxiliary files.

    Returns:
        str or None: Full path to the matching auxiliary file or None.
        """
    
    start_s, stop_s = extract_start_stop_times(raw_0s_file)
    print(aux_files)
    for aux_file in aux_files:
        start_a, stop_a = extract_start_stop_times(aux_file)
        print(start_a, stop_a)
        # Criterio: AUX_ORB___ completamente incluso nel RAW__0S
        margin = timedelta(seconds=10)
        print(start_a.time() , stop_a.time())
        print(start_s.time() , stop_s.time())

        if (
            start_a.time() == time(0, 0, 0)
            and stop_a.time() == time(23, 59, 59)
        ):
            condition = start_s >= start_a and stop_s <= stop_a
        elif start_a.time() == time(0, 0, 0):
            condition = start_s >= start_a and stop_s <= stop_a + margin
        elif stop_a.time() == time(23, 59, 59):
            condition = start_s >= start_a - margin and stop_s <= stop_a
        else:
            condition = start_s >= start_a - margin and stop_s <= stop_a + margin
            
        

        print(condition)

        if condition:
            print(os.path.join(folder, aux_file))
            return os.path.join(folder, aux_file)

    return ""

def render_template(template_path, output_path, replacements):
    """
    Generate JobOrder XML file by replacing placeholders in the template.

    Args:
        template_path (str): Path to the template XML file.
        output_path (str): Path where the new JobOrder will be written.
        replacements (dict): Dictionary of placeholder replacements.
    """
    
    with open(template_path, 'r') as f:
        content = f.read()

    for key, value in replacements.items():
        placeholder = f"{{{{{key}}}}}"  # costruisce {{KEY}}
        content = content.replace(placeholder, str(value))

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Create JobOrder: {output_path}")

# Create JobOrder for L1F processing
def createJOL1F(processing_type,processor_version, input_folder):
    """
    Create JobOrder files for L1F processing.

    Args:
        processing_type (str): Processing level, e.g., 'L1F'.
        processor_version (str): Version of the processor (format XX.XX).
        input_folder (str): Path to the folder containing input data.

    Output:
        One JobOrder XML file for each RAW__0S file found, saved in the input_folder with a name formatted as:
        'JobOrder_<input_folder_basename>_<processing_type>_<start_time>_V<version>.xml'.

        The output directory path is also passed as a parameter into the JobOrder XML and follows the structure:
        input_folder/framers_<RAW__0S_filename>
    """ 
    print('createJOL1F')

    template_path = TEMPLATE_MAP[processing_type]
    all_inputs_folder=os.path.join(input_folder, "Inputs")
    print(all_inputs_folder)
    RAW_0S_files =      [f for f in os.listdir(all_inputs_folder) if f.startswith("BIO") and "RAW__0S" in f and "framers" not in f]
    RAW_0M_files =      [f for f in os.listdir(all_inputs_folder) if 'RAW__0M' in f]
    AUX_ORB__files =    [f for f in os.listdir(all_inputs_folder) if 'AUX_ORB___' in f]

        
    template_path = TEMPLATE_MAP[processing_type]
    for raw_file in RAW_0S_files:

        found_files = {}
        found_files['RAW__0S'] = os.path.join(all_inputs_folder, raw_file) 
        
        raw_0m_path = find_matching_raw_0m(raw_file, RAW_0M_files, all_inputs_folder)
        found_files['RAW__0M'] = raw_0m_path if raw_0m_path else ""
        found_files['AUX_ORB'] = find_matching(raw_file, AUX_ORB__files, all_inputs_folder)
        print(found_files)
        
        product_RAW__0S=    os.path.basename(found_files['RAW__0S'])
        product_RAW__0M=    os.path.basename(found_files['RAW__0M'])
        
        product_RAW__0S_inside=BiomassProduct.BiomassProductRAWS(found_files['RAW__0S'])                
        valid_start_time= datetime.strptime(product_RAW__0S_inside.valid_start_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        valid_stop_time=  datetime.strptime(product_RAW__0S_inside.valid_end_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        
        print(valid_start_time)
        print(valid_stop_time)
        #create a new JobOrder based the template
        output_folder=os.path.join(input_folder, f"frames_{product_RAW__0S}")
        print (output_folder)
        raw_prefix = product_RAW__0S[4:6]  
        derived_product= {
            "standard":        raw_prefix+"_SCS__1S",
            "monitoring":      raw_prefix+"_SCS__1M",
            "dgm":             raw_prefix+"_DGM__1S",
            "calibration_gt1": raw_prefix+"_SCS1__1S",
            "calibration_gt2": raw_prefix+"_SCS2__1S",
            "calibration_x":   raw_prefix+"_SCSX__1S",
            "calibration_y":   raw_prefix+"_SCSY__1S"
        }
        replacements = {
                "VERSION":      processor_version,
                "START_TIME":   valid_start_time.isoformat(),
                "STOP_TIME" :   valid_stop_time.isoformat(),
                "TYPE_RAW0S":   raw_prefix+"_RAW__0S",
                "PATH_RAW0S":   found_files['RAW__0S'],
                "TYPE_RAW0M":   raw_prefix+"_RAW__0M",
                "PATH_RAW0M":   found_files['RAW__0M'],
                "PATH_AUX_ORB": found_files['AUX_ORB'],
                "OUTPUT_DIR":   output_folder ,
                "BASELINE":     product_RAW__0S[71:73] 
                
                }
        start_time=  product_RAW__0S[15:29]
        print (replacements)
        folder_name = os.path.basename(os.path.normpath(input_folder))
        output_JobOrder_path=os.path.join(input_folder, f"JobOrder_{folder_name}_{processing_type}_{start_time}_V{processor_version}.xml") 
        print(output_JobOrder_path)
        render_template(template_path, output_JobOrder_path, replacements) 
            
def find_aux_files(pattern, inputs_folder):
    """
    Search for AUX files matching a pattern in the following priority order:
      1. inputs_folder  (Inputs/)
      2. AUX_USER_DIR
      3. AUX_DEFAULT_DIR

    For AUX_PP1: returns a list of ALL matching files found in the
    highest-priority folder that contains at least one match.
    For all other AUX types: returns a list with the single match
    (or raises if not found).

    Args:
        pattern (str): substring to match in filename (e.g. 'AUX_INS___')
        inputs_folder (str): path to the Inputs/ folder

    Returns:
        list: list of full paths of matching files

    Raises:
        FileNotFoundError: if no match found in any search folder
    """
    search_dirs = [
        inputs_folder,
        AUX["AUX_USER_DIR"],
        AUX["AUX_DEFAULT_DIR"],
    ]

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            print(f"  ⚠️  Skipping non-existent dir: {search_dir}")
            continue
        matches = [
            os.path.join(search_dir, f)
            for f in os.listdir(search_dir)
            if pattern in f
        ]
        if matches:
            print(f"  ✅ {pattern} found in: {search_dir} ({len(matches)} file(s))")
            return matches

    raise FileNotFoundError(
        f"❌ No file matching '{pattern}' found in any of:\n"
        + "\n".join(f"  - {d}" for d in search_dirs)
    )

def createJOL1_chain(processing_type,processor_version, input_folder):
    
    """
    Create JobOrder XML files for L1 processing using information from L1F framers.

    Args:
        processing_type (str): Type of processing (must be 'L1').
        processor_version (str): Version of the processor (e.g., '03.31').
        input_folder (str): Directory containing input products and framers.

    Output:
        One JobOrder XML per EOF frame found under each 'framer_<RAW__0S>' subfolder.
        Output files are written in input_folder and named as:
        'JobOrder_<input_folder_basename>_L1_<start_time>_V<version>_<frame_id>.xml'.
        The output directory inside the JobOrder is set to: input_folder/output
    """

   
    print("Creating JobOrder for L1 processing...")
    template_path = TEMPLATE_MAP[processing_type]
    all_inputs_folder=os.path.join(input_folder, "Inputs")
    
    print(f"Looking for input files in: {all_inputs_folder}")
    RAW_0S_files =          [f for f in os.listdir(all_inputs_folder) if f.startswith("BIO") and "RAW__0S" in f and "frames" not in f]
    RAW_0M_files =          [f for f in os.listdir(all_inputs_folder) if 'RAW__0M' in f]
    AUX_ATT__files =        [f for f in os.listdir(all_inputs_folder) if 'AUX_ATT___' in f]
    AUX_ORB__files =        [f for f in os.listdir(all_inputs_folder) if 'AUX_ORB___' in f]
    AUX_TEC__files =        [f for f in os.listdir(all_inputs_folder) if 'AUX_TEC___' in f]    
    # AUX_INS e AUX_PP1: ricerca con priorità Inputs → AUX_USER → AUX_441
    try:
        AUX_INS__files = find_aux_files('AUX_INS___', all_inputs_folder)
    except FileNotFoundError as e:
        print(e)
        return

    try:
        AUX_PP1__files = find_aux_files('AUX_PP1___', all_inputs_folder)
    except FileNotFoundError as e:
        print(e)
        return
    
    #search the template 

    print(f"Found {len(RAW_0S_files)} RAW__0S file(s): {RAW_0S_files}")
    print(f"Found {len(RAW_0M_files)} RAW__0M file(s): {RAW_0M_files}")
    print(f"Found {len(AUX_ATT__files)} AUX_ATT___ file(s): {AUX_ATT__files}")
    print(f"Found {len(AUX_ORB__files)} AUX_ORB___ file(s): {AUX_ORB__files}")
    print(f"Found {len(AUX_TEC__files)} AUX_TEC___ file(s): {AUX_TEC__files}")
    print(f"Found {len(AUX_INS__files)} AUX_INS___ file(s): {AUX_INS__files}")
    print(f"Found {len(AUX_PP1__files)} AUX_PP1___ file(s): {AUX_PP1__files}")
    #new JobOrder
    folder_name = os.path.basename(os.path.normpath(input_folder))
    AUX_INS__basenames = [os.path.basename(f) for f in AUX_INS__files]
    AUX_INS__folder    = os.path.dirname(AUX_INS__files[0])
    AUX_PP1__basenames = [os.path.basename(f) for f in AUX_PP1__files]
    AUX_PP1__folder    = os.path.dirname(AUX_PP1__files[0])
    for raw_file in RAW_0S_files:
        found_files = {}
        found_files['RAW__0S'] = os.path.join(all_inputs_folder, raw_file)
        
        raw_0m_path = find_matching_raw_0m(raw_file, RAW_0M_files, all_inputs_folder)
        found_files['RAW__0M'] = raw_0m_path if raw_0m_path else ""
        found_files['AUX_ORB'] = find_matching(raw_file, AUX_ORB__files, all_inputs_folder)
        found_files['AUX_ATT'] = find_matching(raw_file, AUX_ATT__files, all_inputs_folder)
        found_files['AUX_TEC'] = find_matching(raw_file, AUX_TEC__files, all_inputs_folder)
        found_files['AUX_INS'] = find_matching(raw_file, AUX_INS__basenames, AUX_INS__folder)
        found_files['AUX_PP1'] = find_matching(raw_file, AUX_PP1__basenames, AUX_PP1__folder)              
        found_files['L1VFRA'] =  os.path.join(input_folder, 'frames_'+raw_file)

        
        product_RAW__0S=    os.path.basename(found_files['RAW__0S'])
        start_time=         product_RAW__0S[15:29]
        
       
        
        raw_prefix = product_RAW__0S[4:6]  
        derived_product= {
            "standard":        raw_prefix+"_SCS__1S",
            "monitoring":      raw_prefix+"_SCS__1M",
            "dgm":             raw_prefix+"_DGM__1S",
            "calibration_gt1": raw_prefix+"_SCS1__1S",
            "calibration_gt2": raw_prefix+"_SCS2__1S",
            "calibration_x":   raw_prefix+"_SCSX__1S",
            "calibration_y":   raw_prefix+"_SCSY__1S"
        }
      
        if os.path.isdir(found_files['L1VFRA']):
            
            eof_files = []
            
            for f in os.listdir(found_files['L1VFRA']):
                if f.endswith(".EOF"):
                    eof_files.append(os.path.join(found_files['L1VFRA'], f))
        
        for l1vfra in eof_files:
            product_L1VFRA=BiomassProduct.BiomassProductL1VFRA(l1vfra)
            frameStartTime=product_L1VFRA.frame_start_time
            frameStopTime= product_L1VFRA.frame_stop_time
            frameFrameID=  product_L1VFRA.frame_id
            frameStatus=   product_L1VFRA.frame_status

            print("\n[INFO] Frame metadata extracted from:", l1vfra)
            print(f"  Start Time : {frameStartTime}")
            print(f"  Stop Time  : {frameStopTime}")
            print(f"  Frame ID   : {frameFrameID}")
            print(f"  Status     : {frameStatus}")


            #create a new JobOrder based the template
            replacements = {
                "VERSION":      processor_version,
                "START_TIME":   frameStartTime,
                "STOP_TIME":    frameStopTime,
                "FRAME_ID":     frameFrameID,
                "FRAME_STATUS": frameStatus,
                "PATH_DEM":     AUX_STATIC['DEM'],
                "PATH_GMF":     AUX_STATIC['GMF'],
                "PATH_IRI":     AUX_STATIC['IRI'],
                "TYPE_RAW0S":   raw_prefix+"_RAW__0S",
                "PATH_RAW0S":   found_files['RAW__0S'],
                "TYPE_RAW0M":   raw_prefix+"_RAW__0M",
                "PATH_RAW0M":   found_files['RAW__0M'],
                "PATH_AUX_ORB": found_files['AUX_ORB'],
                "PATH_AUX_ATT": found_files['AUX_ATT'],
                "PATH_AUX_TEC": found_files['AUX_TEC'],
                "PATH_AUX_INS": found_files['AUX_INS'],
                "PATH_AUX_PP1": found_files['AUX_PP1'],
                "TYPE_SCS__1S": derived_product['standard'],#<---------------------------- da pensarci bene  per le altre opzioni 
                "OUTPUT_DIR":   os.path.join(input_folder, "OUTPUT_L1") ,
                "TYPE_SCS__1M": derived_product['monitoring'],
                "BASELINE":     product_RAW__0S[71:73] ,
                "TYPE_DGM__1S": derived_product['dgm'],
                "INTERMEDIATE_DIR":os.path.join(input_folder, "intermediates_"+frameStartTime.replace("-", "").replace(":", "").replace(".", ""))
                }
            
            print("\n[INFO] Replacements for JobOrder template:")
            for key, value in replacements.items():
                print(f"  {key}: {value}")
            output_JobOrder_path=os.path.join(input_folder, f"JobOrder_{folder_name}_{processing_type}_{start_time}_V{processor_version}_{frameFrameID}.xml") 
            print(f"\n[INFO] Output JobOrder will be saved to:\n  {output_JobOrder_path}")
            render_template(template_path, output_JobOrder_path, replacements)


def extract_pp1_suffix(pp1_name):
        match = re.match(r"(BIO_AUX_PP1____\d{8}T\d{6}_\d{8}T\d{6}_\d{2}_[A-Z0-9]+)_(.*)/?", pp1_name)
        if match:
            return match.group(2).rstrip('/')
        return None



def createJOL1(processing_type,processor_version, input_folder):
    
    """
    Create JobOrder XML files for L1 processing using information from start and stop RAWS product.

    Args:
        processing_type (str): Type of processing (must be 'L1').
        processor_version (str): Version of the processor (e.g., '03.31').
        input_folder (str): Directory containing input products and framers.

    Output:
        One JobOrder XML per EOF frame found under each 'framer_<RAW__0S>' subfolder.
        Output files are written in input_folder and named as:
        'JobOrder_<input_folder_basename>_L1_<start_time>_V<version>_<frame_id>.xml'.
        The output directory inside the JobOrder is set to: input_folder/output
    """
    print("Creating JobOrder for L1 processing...")
    #FIXED_AUX_FILES=aux_versions['AUX_FILES_'+str(processor_version)]
    template_path = TEMPLATE_MAP[processing_type]
    all_inputs_folder=os.path.join(input_folder, "Inputs")
    print(all_inputs_folder)
    
    
    RAW_0S_files =          [f for f in os.listdir(all_inputs_folder) if f.startswith("BIO") and "RAW__0S" in f and "frames" not in f]
    RAW_0M_files =          [f for f in os.listdir(all_inputs_folder) if 'RAW__0M' in f]
    AUX_ATT__files =        [f for f in os.listdir(all_inputs_folder) if 'AUX_ATT___' in f]
    AUX_ORB__files =        [f for f in os.listdir(all_inputs_folder) if 'AUX_ORB___' in f]
    AUX_TEC__files =        [f for f in os.listdir(all_inputs_folder) if 'AUX_TEC___' in f] 
    # AUX_INS e AUX_PP1: ricerca con priorità Inputs → AUX_USER → AUX_441
    try:
        AUX_INS__files = find_aux_files('AUX_INS___', all_inputs_folder)
    except FileNotFoundError as e:
        print(e)
        return

    try:
        AUX_PP1__files = find_aux_files('AUX_PP1___', all_inputs_folder)
    except FileNotFoundError as e:
        print(e)
        return
        #search the template 

    print(f"Found {len(RAW_0S_files)} RAW__0S file(s): {RAW_0S_files}")
    print(f"Found {len(RAW_0M_files)} RAW__0M file(s): {RAW_0M_files}")
    print(f"Found {len(AUX_ATT__files)} AUX_ATT___ file(s): {AUX_ATT__files}")
    print(f"Found {len(AUX_ORB__files)} AUX_ORB___ file(s): {AUX_ORB__files}")
    print(f"Found {len(AUX_TEC__files)} AUX_TEC___ file(s): {AUX_TEC__files}")
    print(f"Found {len(AUX_INS__files)} AUX_INS___ file(s): {AUX_INS__files}")
    print(f"Found {len(AUX_PP1__files)} AUX_PP1___ file(s): {AUX_PP1__files}")

    AUX_INS__basenames = [os.path.basename(f) for f in AUX_INS__files]
    AUX_INS__folder    = os.path.dirname(AUX_INS__files[0])
    AUX_PP1__basenames = [os.path.basename(f) for f in AUX_PP1__files]
    AUX_PP1__folder    = os.path.dirname(AUX_PP1__files[0])
    #new JobOrder
    folder_name = os.path.basename(os.path.normpath(input_folder))    
    for raw_file in RAW_0S_files:
        found_files = {}
        found_files['RAW__0S'] = os.path.join(all_inputs_folder, raw_file)  
        raw_0m_path = find_matching_raw_0m(raw_file, RAW_0M_files, all_inputs_folder)
        found_files['RAW__0M'] = raw_0m_path if raw_0m_path else ""
        
        found_files['AUX_ORB'] = find_matching(raw_file, AUX_ORB__files, all_inputs_folder)
        found_files['AUX_ATT'] = find_matching(raw_file, AUX_ATT__files, all_inputs_folder)
        found_files['AUX_TEC'] = find_matching(raw_file, AUX_TEC__files, all_inputs_folder)
        found_files['AUX_INS'] = find_matching(raw_file, AUX_INS__basenames, AUX_INS__folder)
        found_files['AUX_PP1'] = find_matching(raw_file, AUX_PP1__files, all_inputs_folder)
        
        
        print (found_files['RAW__0M'])
        
        product_RAW__0S_basename=    os.path.basename(found_files['RAW__0S'])
        start_time=         product_RAW__0S_basename[15:29]
       
        
        raw_prefix = product_RAW__0S_basename[4:6]  
        derived_product= {
            "standard":        raw_prefix+"_SCS__1S",
            "monitoring":      raw_prefix+"_SCS__1M",
            "dgm":             raw_prefix+"_DGM__1S",
            "calibration_gt1": raw_prefix+"_SCS1__1S",
            "calibration_gt2": raw_prefix+"_SCS2__1S",
            "calibration_x":   raw_prefix+"_SCSX__1S",
            "calibration_y":   raw_prefix+"_SCSY__1S"
        }
      
        
        
        product_RAW__0S=BiomassProduct.BiomassProductRAWS(found_files['RAW__0S'])
        frameStartTime_0S_str=product_RAW__0S.start_time
        frameStartTime_0S= datetime.strptime(frameStartTime_0S_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        frameStopTime_0S_str= product_RAW__0S.end_time 
        frameStopTime_0S= datetime.strptime(frameStopTime_0S_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        dateTakeID_RAW_0S=product_RAW__0S.data_take_id
        slice_RAW_0S=product_RAW__0S_basename[66:70]
        
        
        if found_files['RAW__0M']!="":
          product_RAW__0M=BiomassProduct.BiomassProductRAWM(found_files['RAW__0M'])
          frameStartTime_0M_str=product_RAW__0M.start_time
          frameStartTime_0M= datetime.strptime(frameStartTime_0M_str, "%Y-%m-%dT%H:%M:%S.%fZ")        
          frameStopTime_0M_str= product_RAW__0M.end_time 
          frameStopTime_0M= datetime.strptime(frameStopTime_0M_str, "%Y-%m-%dT%H:%M:%S.%fZ")  
        else:
               
          frameStartTime_0M=frameStartTime_0S
          frameStopTime_0M=frameStopTime_0S
        
        print (found_files['AUX_ORB'])
        print(found_files['AUX_ATT'])
        product_ORB=BiomassProduct.auxorb(found_files['AUX_ORB'])
        frameStartTime_orb=product_ORB.start_time
        frameStopTime_orb= product_ORB.stop_time 
        
        product_ATT=BiomassProduct.auxatt(found_files['AUX_ATT'])
        frameStartTime_att=product_ATT.start_time
        frameStopTime_att= product_ATT.stop_time         
        
        
        # Lista degli start e stop time come oggetti datetime
        start_times = [
            frameStartTime_0S,
            frameStartTime_0M,
            frameStartTime_orb,
            frameStartTime_att
        ]
        
        stop_times = [
            frameStopTime_0S,
            frameStopTime_0M,
            frameStopTime_orb,
            frameStopTime_att
        ]
        
        # Calcolo: massimo degli start e minimo degli stop - 1.5s
        final_start = max(start_times)
        final_stop = min(stop_times)- timedelta(seconds=1.5)
        
        # Formatta come stringa ISO con millisecondi e 'Z'
        frameStartTime = final_start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        frameStopTime = final_stop.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" 
        
        
        for pp1_file in AUX_PP1__files:
            pp1_path = pp1_file
            suffix = extract_pp1_suffix(pp1_file) or "default"
            suffix_full = f"{dateTakeID_RAW_0S}_{slice_RAW_0S}_{frameStartTime_0S.strftime('%Y%m%dT%H%M%S')}_{suffix}_V{processor_version}"
            output_dir = os.path.join(input_folder, f"OUTPUT_L1_{suffix_full}")
            intermediate_dir = os.path.join(input_folder, f"intermediates_{suffix_full}")
            joborder_name = f"JobOrder_{folder_name}_{processing_type}_{suffix_full}.xml"
        
        
        

            #create a new JobOrder based the template
            replacements = {
                "VERSION":      processor_version,
                "START_TIME":   frameStartTime,
                "STOP_TIME":    frameStopTime,
                "FRAME_ID":     '001',
                "FRAME_STATUS": 'NOMINAL',
                "PATH_DEM":     AUX_STATIC['DEM'],
                "PATH_GMF":     AUX_STATIC['GMF'],
                "PATH_IRI":     AUX_STATIC['IRI'],
                "TYPE_RAW0S":   raw_prefix+"_RAW__0S",
                "PATH_RAW0S":   found_files['RAW__0S'],
                "TYPE_RAW0M":   raw_prefix+"_RAW__0M",
                "PATH_RAW0M":   found_files['RAW__0M'],
                "PATH_AUX_ORB": found_files['AUX_ORB'],
                "PATH_AUX_ATT": found_files['AUX_ATT'],
                "PATH_AUX_TEC": found_files['AUX_TEC'],
                "PATH_AUX_INS": found_files['AUX_INS'],
                "PATH_AUX_PP1": pp1_path,
                "TYPE_SCS__1S": derived_product['standard'],
                "OUTPUT_DIR":   output_dir ,
                "TYPE_SCS__1M": derived_product['monitoring'],
                "BASELINE":     product_RAW__0S_basename[71:73] ,
                "TYPE_DGM__1S": derived_product['dgm'],
                 "INTERMEDIATE_DIR": intermediate_dir
                }
            
            print("\n[INFO] Replacements for JobOrder template:")
            for key, value in replacements.items():
                print(f"  {key}: {value}")


            
            
            output_JobOrder_path = os.path.join(input_folder, joborder_name)
            render_template(template_path, output_JobOrder_path, replacements)
            print(f"[INFO] JobOrder created: {output_JobOrder_path}")


def createJOSTA_chain(processing_type, processor_version, input_folder, mission_phase, overlap_threshold=0.7):
    """
    Create one STA JobOrder per FRAME by grouping SCS__1S products
    having the same FRAME and repeatCycleID, and footprint overlap >= threshold.

    Args:
        processing_type (str): 'STA'
        processor_version (str): e.g. '03.31'
        input_folder (str): base processing folder
        mission_phase (str): 'interferometric' or 'tomographic'
        overlap_threshold (float): minimum footprint overlap (default 0.9)
    """

    print("\n[INFO] Creating STA JobOrders by stack (FRAME-based)")
    print(f"[INFO] Processor version : {processor_version}")
    print(f"[INFO] Mission phase     : {mission_phase}")
    print(f"[INFO] Overlap threshold : {overlap_threshold}")

    template_path = TEMPLATE_MAP[processing_type]
    input_folder_scs = os.path.join(input_folder, "OUTPUT_L1")

    # ----------------------------
    # AUX_PPS: Inputs/ → AUX_USER → AUX_DEFAULT
    # ----------------------------
    try:
        AUX_PPS__files = find_aux_files('AUX_PPS___', input_folder_scs)
        AUX_PPS__path = AUX_PPS__files[0]
        print(f"[INFO] AUX_PPS___ file path: {AUX_PPS__path}")
    except FileNotFoundError as e:
        print(e)
        return

    # ------------------------------------------------------------------
    # Step 1: Scan SCS__1S products and extract metadata
    # ------------------------------------------------------------------
    scs_files = []
    SCS_RE = re.compile(r"_SCS__1S_(\d{8}T\d{6})_(\d{8}T\d{6}).*_(F\d{3})_")
    for fname in os.listdir(input_folder_scs):
        print (fname)
        if "_SCS__1S_" not in fname:
            continue

        full_path = os.path.join(input_folder_scs, fname)
        m = SCS_RE.search(fname)
        if not m:
            print(f"[WARN] Cannot parse name, skipping: {fname}")
            continue

        start_str, stop_str, frame = m.group(1), m.group(2), m.group(3)
        print('---------------------------------------------------------')
        print(full_path)

        product = BiomassProduct.BiomassProductSCS(full_path)
        print('---------------------------------------------------------')
        scs_files.append({
            "FILE_NAME": fname,
            "FULL_PATH": full_path,
            "FRAME": frame,
            "REPEAT_CYCLE": product.repeatCycleID,  # from MPH
            "FOOTPRINT": product.footprint_polygon,
            "DATE_START": datetime.strptime(start_str, "%Y%m%dT%H%M%S"),
            "DATE_STOP": datetime.strptime(stop_str, "%Y%m%dT%H%M%S"),
        })

    print(f"[INFO] Total SCS__1S products found: {len(scs_files)}")

    if not scs_files:
        print("[ERROR] No SCS__1S products found. Aborting.")
        return

    # ------------------------------------------------------------------
    # Step 2: Group by FRAME + repeatCycleID
    # ------------------------------------------------------------------
    groups = defaultdict(list)
    for scs in scs_files:
        key = (scs["FRAME"])
        groups[key].append(scs)

    print(f"[INFO] Number of candidate stacks: {len(groups)}")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def footprint_overlap(p1, p2):
        inter = p1.intersection(p2).area
        return inter / min(p1.area, p2.area)

    def validate_overlap(group):
      """
      Check that all products in the group overlap with the reference
      footprint by at least overlap_threshold.
      """
      ref = group[0]["FOOTPRINT"]
      if ref is None:
        return False

      ref = ref.buffer(0)  # fix invalid geometry
      min_ov = 1.0

      for scs in group[1:]:
        p = scs["FOOTPRINT"]
        if p is None:
            return False

        p = p.buffer(0)

        inter_area = ref.intersection(p).area
        ov = inter_area / min(ref.area, p.area)

        min_ov = min(min_ov, ov)

        print(
            f"[DEBUG] overlap vs ref = {ov:.3f}  file={scs['FILE_NAME']}"
        )

      print(f"[DEBUG] MIN overlap in group: {min_ov:.3f}")
      return min_ov >= overlap_threshold


    # ------------------------------------------------------------------
    # Step 3: Create one JobOrder per valid stack
    # ------------------------------------------------------------------
    for (frame), group in groups.items():

        print(f"\n[INFO] Processing stack FRAME={frame}")
        print(f"[INFO] Number of products in stack: {len(group)}")

        if not validate_overlap(group):
            print(f"[WARN] Footprint overlap < {overlap_threshold*100:.0f}% → stack skipped")
            continue

        group = sorted(group, key=lambda x: x["DATE_START"])

        list_of_scs = "\n".join(
            f"        <File_Name>{item['FULL_PATH']}</File_Name>"
            for item in group
        )

        reference = group[0]
        product_name = reference["FILE_NAME"]

        raw_prefix = product_name[4:6]  # e.g. S2

        derived_product = {
            "standard": raw_prefix + "_STA__1S",
            "monitoring": raw_prefix + "_STA__1M",
        }

        joborder_name = f"JobOrder_STA_{frame}_V{processor_version}.xml"
        output_joborder = os.path.join(input_folder, joborder_name)

        replacements = {
            "VERSION": processor_version,
            "MISSION_PHASE": mission_phase,
            "SWATH": raw_prefix,
            "PATH_FNF": AUX_STATIC["FNF"],
            "TYPE_SCS1S": product_name[4:14],
            "LIST_OF_SCS_FILES": list_of_scs,
            "PATH_AUX_PPS": AUX_PPS__path,
            "TYPE_STA__1S": derived_product["standard"],
            "TYPE_STA__1M": derived_product["monitoring"],
            "OUTPUT_DIR": os.path.join(input_folder, f"OUTPUT_STA_{frame}_V{processor_version}"),
            "BASELINE": product_name[71:73],
            "INTERMEDIATE_DIR": os.path.join(Path(input_folder), f"intermediate_STA_{frame}_V{processor_version}"),
        }

        print(f"[INFO] Writing JobOrder: {output_joborder}")
        render_template(template_path, output_joborder, replacements)

    print("\n[INFO] STA JobOrder generation completed.")

               

        

def createJOSTA(processing_type,processor_version, input_folder,mission_phase) :
    print('create JobOrder STA')   
    """
    Create JobOrder for STA processing by grouping SCS__1S files from start and stop  acquisition times SCS.

    Args:
    processing_type (str): 'STA'
    processor_version (str): Version of the processor (e.g. '03.31')
    input_folder (str): Directory containing 'output_*' subfolders
    mission_phase (str): 'interferometric' or 'tomographic'
    """
    print("\n[INFO] Starting JobOrder creation for STA processing...")
    print(f"[INFO] Processing type     : {processing_type}")
    print(f"[INFO] Processor version   : {processor_version}")
    print(f"[INFO] Input folder        : {input_folder}")
    print(f"[INFO] Mission phase       : {mission_phase}")
    # Step 1: Find all SCS__1S products inside output_* folders
    #create una lista di dict in cui per ogni dict c'e  il path dell'scs__1s e lo starttime e stoptime

    template_path = TEMPLATE_MAP[processing_type]
    scs_files = []
   
    input_folder_scs = os.path.join(input_folder, "Inputs")
    #AUX_PPS__files =        [f for f in os.listdir(input_folder_scs) if 'AUX_PPS___' in f]
    # ----------------------------
    # AUX_PPS: Inputs/ → AUX_USER → AUX_DEFAULT
    # ----------------------------
    try:
        AUX_PPS__files = find_aux_files('AUX_PPS___', input_folder_scs)
        AUX_PPS__path = AUX_PPS__files[0]
        print(f"[INFO] AUX_PPS___ file path: {AUX_PPS__path}")
    except FileNotFoundError as e:
        print(e)
        return

         
    for file in os.listdir(input_folder_scs):
        scs_file=dict()
        if "_SCS__1S_" in file:
            
            print(file)
            parts = file.split("_")
            
            # STARTTIME è il 4° pezzo (index 3), STOPTIME il 5° (index 4)
            start_str = parts[5]  # '20170406T220021'
            stop_str  = parts[6]  # '20170406T220041'
            
            scs_file['DATE_STARTTIME'] = datetime.strptime(start_str, "%Y%m%dT%H%M%S")
            scs_file['DATE_STOPTIME']  = datetime.strptime(stop_str, "%Y%m%dT%H%M%S")
            

            scs_file['STARTTIME'] = scs_file['DATE_STARTTIME'].strftime("%H%M%S")
            scs_file['SCS__1S']= file
            scs_file['FULL_PATH']=os.path.join(input_folder_scs, file)

            scs_files.append(scs_file)
    print(f"[INFO] Total SCS__1S files found: {len(scs_files)}")
    list_of_scs = "\n".join([f"        <File_Name>{item['FULL_PATH']}</File_Name>" for item in scs_files])
    # Usa il primo file come riferimento
    reference = scs_files[0]

    product_SCS = reference['SCS__1S']
    folder_name = os.path.basename(os.path.dirname(input_folder))
    raw_prefix = product_SCS[4:6]

    derived_product = {
        "standard": raw_prefix + "_STA__1S",
        "monitoring": raw_prefix + "_STA__1M",
    }

    #output_folder = os.path.join(input_folder, "OUTPUT_STA")
    
    latest_scs = max(scs_files, key=lambda x: x['DATE_STARTTIME'])
    latest_start = latest_scs['DATE_STARTTIME'].strftime("%Y%m%dT%H%M%S")
    output_folder = os.path.join(input_folder, f"OUTPUT_STA_{latest_start}_V{processor_version}")
    

    replacements = {
        "VERSION": processor_version,
        "MISSION_PHASE": mission_phase,
        "SWATH": product_SCS[4:6],
        "PATH_FNF": AUX_STATIC['FNF'],
        "TYPE_SCS1S": product_SCS[4:14],
        "LIST_OF_SCS_FILES": list_of_scs,
        "PATH_AUX_PPS": AUX_PPS__path,
        "TYPE_STA__1S": derived_product['standard'],
        "OUTPUT_DIR": output_folder,
        "BASELINE": product_SCS[71:73] ,
        "TYPE_STA__1M": derived_product['monitoring'],
        "INTERMEDIATE_DIR": os.path.join(input_folder, f"intermidiate_STA_{folder_name}_{latest_start}_{processing_type}_V{processor_version}")
    }

    print("\n[INFO] Replacements for JobOrder template:")
    for key, value in replacements.items():
            print(f"  {key}: {value}")
    output_JobOrder_path=os.path.join(input_folder, f"JobOrder_STA_{folder_name}_{latest_start}_{processing_type}_V{processor_version}.xml") 
    print(f"\n[INFO] Output JobOrder will be saved to:\n  {output_JobOrder_path}")
    render_template(template_path, output_JobOrder_path, replacements)                
    
           
def createJOL2A(processing_type,processor_version, input_folder) :
    print('create JobOrder L2A')
   
    """
    Create JobOrder for L2A processing by grouping STA_1S files from start and stop  acquisition times STA.

    Args:
    processing_type (str): 'L2A'
    processor_version (str): Version of the processor (e.g. '03.31')
    input_folder (str): Directory containing 'OUTPUT_*' subfolders

    """
    print("\n[INFO] Starting JobOrder creation for STA processing...")
    print(f"[INFO] Processing type     : {processing_type}")
    print(f"[INFO] Processor version   : {processor_version}")
    print(f"[INFO] Input folder        : {input_folder}")
    
    # Step 1: Find all STA__1S products inside output_* folders
    #create una lista di dict in cui per ogni dict c'e  il path dell'scs__1s e lo starttime e stoptime

    template_path = TEMPLATE_MAP[processing_type]
    sta_files = []
   
    input_folder_sta = os.path.join(input_folder, "Inputs")
    #AUX_PPS__files =        [f for f in os.listdir(input_folder_scs) if 'AUX_PPS___' in f]
    try:
        AUX_PP2_2A_files = find_aux_files('AUX_PP2_2A_', input_folder_sta)
        AUX_PP2_2A_path = AUX_PP2_2A_files[0]
        print(f"[INFO] AUX_PP2_2A_ file path: {AUX_PP2_2A_path}")
    except FileNotFoundError as e:
        print(e)
        return

    print(f"[INFO] _AUX_PP2_2A_ file path: {AUX_PP2_2A_path}")
     
    
    for file in os.listdir(input_folder_sta):
        sta_file=dict()
        if "_STA__1S_" in file:
            
            print(file)
            parts = file.split("_")
            
            # STARTTIME è il 4° pezzo (index 3), STOPTIME il 5° (index 4)
            start_str = parts[5]  # '20170406T220021'
            stop_str  = parts[6]  # '20170406T220041'
            
            sta_file['DATE_STARTTIME'] = datetime.strptime(start_str, "%Y%m%dT%H%M%S")
            sta_file['DATE_STOPTIME']  = datetime.strptime(stop_str, "%Y%m%dT%H%M%S")
            

            sta_file['STARTTIME'] = sta_file['DATE_STARTTIME'].strftime("%H%M%S")
            sta_file['STA__1S']= file
            sta_file['FULL_PATH']=os.path.join(input_folder_sta, file)

            sta_files.append(sta_file)
    print(f"[INFO] Total STA__1S files found: {len(sta_files)}")
    list_of_sta = "\n".join([f"        <File_Name>{item['FULL_PATH']}</File_Name>" for item in sta_files])
    # Usa il primo file come riferimento
    reference = sta_files[0]

    product_STA = reference['STA__1S']
    folder_name = os.path.basename(os.path.dirname(input_folder))
    raw_prefix = product_STA[4:6]

    derived_product = {
        "standard": raw_prefix + "_STA__1S",
        "monitoring": raw_prefix + "_STA__1M",
    }

    output_folder = os.path.join(input_folder, "OUTPUT_L2A")

    replacements = {
        "VERSION": processor_version,
        "SWATH": product_STA[4:6],
        "PATH_FNF": AUX_STATIC['FNF'],
        "TYPE_STA1S": product_STA[4:14],
        "LIST_OF_STA_FILES": list_of_sta,
        "PATH_AUX_PP2": AUX_PP2_2A_path,
        "TYPE_STA__1S": derived_product['standard'],
        "OUTPUT_DIR": output_folder,
        "BASELINE": product_STA[71:73] 
        # "INTERMEDIATE_DIR": os.path.join(input_folder, f"intermidiate_STA_{folder_name}_{processing_type}_V{processor_version}")
    }

    print("\n[INFO] Replacements for JobOrder template:")
    for key, value in replacements.items():
            print(f"  {key}: {value}")
    output_JobOrder_path=os.path.join(input_folder, f"JobOrder_L2A_{folder_name}_{processing_type}_V{processor_version}.xml") 
    print(f"\n[INFO] Output JobOrder will be saved to:\n  {output_JobOrder_path}")
    render_template(template_path, output_JobOrder_path, replacements) 
    



def createJOL2A_chain(processing_type, processor_version, input_root):

    print("\n[INFO] Creating JobOrders L2A (grouped by FRAME)")
    print(f"[INFO] Root folder: {input_root}")

    # --------------------------------------------------
    # 1) Find PP2: input_root → AUX_USER → AUX_DEFAULT
    # --------------------------------------------------
    try:
        AUX_PP2_2A_files = find_aux_files('AUX_PP2_2A_', input_root)
        AUX_PP2_2A_path = AUX_PP2_2A_files[0]
        print(f"[INFO] PP2 file: {AUX_PP2_2A_path}")
    except FileNotFoundError as e:
        print(e)
        return

    # --------------------------------------------------
    # 2) Locate OUTPUT_STA folders
    # --------------------------------------------------
    sta_folders = sorted(
        f for f in os.listdir(input_root)
        if f.startswith("OUTPUT_STA")
    )

    template_path = TEMPLATE_MAP[processing_type]

    # --------------------------------------------------
    # 3) Loop over OUTPUT_STA folders
    # --------------------------------------------------
    for sta_folder in sta_folders:

        print("\n--------------------------------------------------")
        print(f"[INFO] Processing folder: {sta_folder}")

        sta_path = os.path.join(input_root, sta_folder)

        # --------------------------------------------------
        # 4) Group STA__1S products by FRAME
        # --------------------------------------------------
        groups = defaultdict(list)

        for entry in os.listdir(sta_path):

            if "_STA__1S_" not in entry:
                continue

            parts = entry.split("_")

            try:
                frame = parts[12]          # Fxxx
                start_str = parts[5]
                stop_str  = parts[6]
            except IndexError:
                print(f"[WARN] Unexpected naming format: {entry}")
                continue

            date_start = datetime.strptime(start_str, "%Y%m%dT%H%M%S")
            date_stop  = datetime.strptime(stop_str,  "%Y%m%dT%H%M%S")

            groups[frame].append({
                "DATE_STARTTIME": date_start,
                "DATE_STOPTIME":  date_stop,
                "STA_NAME": entry,
                "FULL_PATH": os.path.join(sta_path, entry)
            })

        if not groups:
            print(f"[WARN] No STA__1S products found in {sta_folder}")
            continue

        # --------------------------------------------------
        # 5) Create one JobOrder per FRAME
        # --------------------------------------------------
        for frame, sta_list in groups.items():

            print(f"\n[INFO] FRAME: {frame}")
            print(f"[INFO] Number of STA__1S in stack: {len(sta_list)}")

            # Sort temporally (recommended)
            sta_list = sorted(sta_list,
                              key=lambda x: x["DATE_STARTTIME"])

            list_of_sta = "\n".join(
                f"        <File_Name>{item['FULL_PATH']}</File_Name>"
                for item in sta_list
            )

            reference = sta_list[0]["STA_NAME"]

            raw_prefix = reference[4:6]

            derived_product = {
                "standard": raw_prefix + "_STA__1S",
                "monitoring": raw_prefix + "_STA__1M",
            }

            output_folder = os.path.join(
                input_root,
                f"OUTPUT_L2A_{frame}_V{processor_version}"
            )

            replacements = {
                "VERSION": processor_version,
                "SWATH": raw_prefix,
                "PATH_FNF": AUX_STATIC["FNF"],
                "TYPE_STA1S": reference[4:14],
                "LIST_OF_STA_FILES": list_of_sta,
                "PATH_AUX_PP2": AUX_PP2_2A_path,
                "TYPE_STA__1S": derived_product["standard"],
                "OUTPUT_DIR": output_folder,
                "BASELINE": reference[71:73],
            }

            output_joborder = os.path.join(
                input_root,
                f"JobOrder_L2A_{frame}_V{processor_version}.xml"
            )

            print(f"[INFO] Writing JobOrder: {output_joborder}")

            render_template(template_path,
                            output_joborder,
                            replacements)

    print("\n[OK] All L2A JobOrders created.")



def createJOL2B(processing_type,processor_version, input_folder) :
    print('create JobOrder L2B')
    
    

# Entry point of the script
def main(processing_type,processor_version, input_folder, mission_phase=None):
    
    #based the processing type you want create a JobOrder
    if processing_type=='L1F':
        createJOL1F(processing_type,processor_version, input_folder)
    elif processing_type=='L1_chain':
        createJOL1_chain(processing_type,processor_version, input_folder)
        
    elif processing_type=='STA':
            createJOSTA(processing_type,processor_version, input_folder, mission_phase)       
        
    elif processing_type=='STA_chain':
        createJOSTA_chain(processing_type,processor_version, input_folder, mission_phase)   

    elif processing_type=='L1':
        createJOL1(processing_type,processor_version, input_folder)       
        
    elif processing_type=='L2A':
    
        createJOL2A(processing_type,processor_version, input_folder)
    elif processing_type=='L2A_chain':
    
        createJOL2A_chain(processing_type,processor_version, input_folder)
    else:
        print(f"Unsupported processing type: {processing_type}")
                 

def print_help():
    """
    Print command-line usage instructions for JOBuilder, including:
    - expected input folder structure per processing type
    - expected input files
    - output folders generated
    """

    help_message = r"""
===============================================================================
 JOBuilder.py – BIOMASS JobOrder Generator
===============================================================================

USAGE
-----
python JOBuilder.py <processing_type> <processor_version> <input_folder> [mission_phase]

REQUIRED ARGUMENTS
------------------
<processing_type>    One of: L1F, L1, L1_chain, STA, STA_chain, L2A, L2A_chain
<processor_version>  Format: XX.XX  (e.g. 04.31)
<input_folder>       Root processing folder
[mission_phase]      Required only for STA / STA_chain:
                     TOMOGRAPHIC or INTERFEROMETRIC

COMMON REQUIREMENTS
-------------------
- config.ini must be located in the same directory as JOBuilder.py
- Template paths and static AUX paths are read from config.ini

===============================================================================
SUPPORTED VALUES – EXPECTED INPUTS & OUTPUTS
===============================================================================

1) L1F
------
Goal:
  Create 1 JobOrder per RAW__0S (L1F framing)

Expected input folder structure:
  <input_folder>/
    Inputs/
      BIO_*RAW__0S*       (one or more)
      BIO_*RAW__0M*       (optional but expected)
      BIO_*AUX_ORB___*    (one or more)

Expected matching logic:
  - RAW__0M selected such that it fully covers RAW__0S time span
  - AUX_ORB___ selected by time-coverage rule (with margins)

Outputs created:
  - JobOrders written in <input_folder>:
      JobOrder_<folder>_L1F_<startTime>_V<version>.xml
  - Output folder path injected inside JobOrder:
      <input_folder>/frames_<RAW__0S_filename>
    (NOTE: this is the OUTPUT_DIR parameter written in the JobOrder)


2) L1
-----
Goal:
  Create JobOrders for L1 processing (RAW + AUX + PP1 → SCS/DGM)

Expected input folder structure:
  <input_folder>/
    Inputs/
      BIO_*RAW__0S*
      BIO_*RAW__0M*              (optional but expected)
      BIO_*AUX_ORB___*
      BIO_*AUX_ATT___*
      BIO_*AUX_TEC___*
      BIO_*AUX_INS___*
      BIO_*AUX_PP1___*           (one or more, looped)
    (No frames_* folders required for this mode)

Expected matching logic:
  - RAW__0M matched by time span
  - AUX_* matched by time-coverage rule (with margins)

Outputs created:
  - JobOrders written in <input_folder>:
      JobOrder_<folder>_L1_<suffix_full>.xml
    where suffix_full = <datatake>_<slice>_<rawStart>_<pp1Suffix>_V<version>
  - Output folders injected inside each JobOrder (one per PP1):
      <input_folder>/OUTPUT_L1_<suffix_full>
  - Intermediate folders injected:
      <input_folder>/intermediates_<suffix_full>


3) L1_chain
-----------
Goal:
  Create 1 JobOrder per EOF frame by using L1F framers (frames_* folders)

Expected input folder structure:
  <input_folder>/
    Inputs/
      BIO_*RAW__0S*
      BIO_*RAW__0M*
      BIO_*AUX_ORB___*
      BIO_*AUX_ATT___*
      BIO_*AUX_TEC___*
      BIO_*AUX_INS___*
      BIO_*AUX_PP1___*
    frames_<RAW__0S_filename>/
      *.EOF                (one or more frame files)

Expected logic:
  - For each RAW__0S, read all EOF files under frames_<RAW__0S>
  - Create one JobOrder per EOF (frame_id)

Outputs created:
  - JobOrders written in <input_folder>:
      JobOrder_<folder>_L1_<rawStart>_V<version>_<frameID>.xml
  - Output folder injected:
      <input_folder>/OUTPUT_L1
  - Intermediate folder injected (per frame start time):
      <input_folder>/intermediates_<frameStartTime_compact>


4) STA
------
Goal:
  Create 1 STA JobOrder from SCS__1S files (single joborder)

Required extra argument:
  mission_phase = TOMOGRAPHIC or INTERFEROMETRIC

Expected input folder structure (as implemented in createJOSTA):
  <input_folder>/
    Inputs/
      BIO_*SCS__1S*         (one or more)
      BIO_*AUX_PPS___*      (exactly one expected)
    (Note: here your code reads SCS__1S from Inputs/)

Outputs created:
  - JobOrder written in <input_folder>:
      JobOrder_STA_<folder>_<latestStart>_STA_V<version>.xml
  - Output folder injected:
      <input_folder>/OUTPUT_STA_<latestStart>_V<version>
  - Intermediate folder injected:
      <input_folder>/intermidiate_STA_<folder>_<latestStart>_STA_V<version>


5) STA_chain
------------
Goal:
  Create STA JobOrders grouped by FRAME (stack-based) using SCS__1S products

Required extra argument:
  mission_phase = TOMOGRAPHIC or INTERFEROMETRIC

Expected input folder structure (as implemented in createJOSTA_chain):
  <input_folder>/
    OUTPUT_L1/
      BIO_*SCS__1S*         (one or more)
      BIO_*AUX_PPS___*      (exactly one expected in OUTPUT_L1/)
    (Note: here your code reads SCS__1S from OUTPUT_L1/)

Additional expected metadata:
  - footprint_polygon and repeatCycleID read from SCS MPH via BiomassProductSCS
  - overlap check uses polygon overlap >= overlap_threshold

Outputs created:
  - JobOrders written in <input_folder>:
      JobOrder_STA_<Fxxx>_V<version>.xml
  - Output folders injected (one per FRAME):
      <input_folder>/OUTPUT_STA_<Fxxx>_V<version>
  - Intermediate folders injected:
      <input_folder>/intermediate_STA_<Fxxx>_V<version>


6) L2A
------
Goal:
  Create 1 L2A JobOrder from STA__1S files

Expected input folder structure (as implemented in createJOL2A):
  <input_folder>/
    Inputs/
      BIO_*STA__1S*           (one or more)
      BIO_AUX_PP2_2A_*        (exactly one expected)

Outputs created:
  - JobOrder written in <input_folder>:
      JobOrder_L2A_<folder>_L2A_V<version>.xml
  - Output folder injected:
      <input_folder>/OUTPUT_L2A


7) L2A_chain
------------
Goal:
  Create 1 L2A JobOrder per FRAME by scanning OUTPUT_STA* folders

Expected input folder structure (as implemented in createJOL2A_chain):
  <input_root>/
    BIO_AUX_PP2_2A_*          (exactly one expected at root)
    OUTPUT_STA* /             (one or more folders)
      BIO_*STA__1S*           (one or more inside each STA folder)

Logic:
  - For each OUTPUT_STA* folder:
      group STA__1S products by FRAME (parts[12] expected to be Fxxx)
      create one L2A JobOrder per FRAME

Outputs created:
  - JobOrders written in <input_root>:
      JobOrder_L2A_<Fxxx>_V<version>.xml
  - Output folders injected:
      <input_root>/OUTPUT_L2A_<Fxxx>_V<version>


===============================================================================
EXAMPLES
===============================================================================

python JOBuilder.py L1F       04.31 /data/run_001
python JOBuilder.py L1        04.31 /data/run_001
python JOBuilder.py L1_chain  04.31 /data/run_001

python JOBuilder.py STA       04.31 /data/run_001 TOMOGRAPHIC
python JOBuilder.py STA_chain 04.31 /data/run_001 INTERFEROMETRIC

python JOBuilder.py L2A       04.31 /data/run_001
python JOBuilder.py L2A_chain 04.31 /data/run_001

===============================================================================
"""
    print(help_message)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print_help()
    else:
        processing_type = sys.argv[1]
        processor_version = sys.argv[2]
        input_folder = sys.argv[3]

        if processing_type == "STA_chain" or processing_type == "STA" :
            if len(sys.argv) != 5:
                print("Error: 'STA' processing requires a mission_phase argument (interferometric or tomographic).")
                print_help()
                sys.exit(1)
            else:
                mission_phase = sys.argv[4]
                main(processing_type, processor_version, input_folder, mission_phase)
        else:
            main(processing_type, processor_version, input_folder)
        
        
        
        
        