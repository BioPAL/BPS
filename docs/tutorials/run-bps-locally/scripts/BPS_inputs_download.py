# BPS_inputs_download is a set of functions that allows users to download automatically all the necessary input files to be used within the ESA BPS, given one single Biomass L0S (RAW_0S) product id.
# DISCLAIMER: this is a first version with limited functionalities. Further improvements and other features will be implemented.

# For any information_______Joseph Melizza (joseph.melizza@serco.com)

# Version 0.1.1

# Fixes: 
# 1. output products to be searched also considering the RAW_0S input baseline (productVersion)
# 2. metadata values are now extracted from the ogc_17 file (metadata_ogc_17_003r2) instead of the ogc_10 file (metadata_ogc_10_157r4)
# 3. output searching criteria to be displayed after the script is launched (track number, frame number, global coverage, repeat cycle, major cycle and baseline)



from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
from pystac_client import Client
from tqdm import tqdm
import urllib.request
import numpy as np
import requests
import pathlib
import fsspec
import json
import os
import io



def product_search(catalog, collections, prod_type, datetime, baseline):
    '''
    Searches a STAC catalog product matching a given product type, collection and datetime.

    Args:
        catalog (Client): pystac_client Client instance connected to the catalog.
        collections (list[str]): list of collection names to search in.
        prod_type (str): product type filter string (e.g. 'AUX_ATT___').
        datetime (list[str]): a [start_datetime, end_datetime] pair in ISO 8601 format.

    Returns:
        list[Item]: a list containing at most one matching STAC Item.
    '''

    search = catalog.search(
    collections = collections,
    filter = f"product:type='{prod_type}' and\
                version='{baseline}'",
    datetime = datetime,
    method = "GET",
    max_items = 1,  
    )

    items = list(search.items())
    return items


def product_search_majorcycle(catalog, collections, prod_type, global_id, major_cycle, track, frame, baseline):
    """
    Searches a STAC catalog product matching a given product type within a specific major cycle, track, and frame.

    Args:
        catalog (Client): a pystac_client Client instance connected to the catalog.
        collections (list[str]): list of collection names to search in.
        prod_type (str): product type filter string (e.g. 'AUX_ATT___').
        global_id (int): global coverage ID to filter by (eofeos:global_coverage_id).
        major_cycle (int): major cycle ID to filter by (eofeos:major_cycle_id).
        track (int): track number to filter by.
        frame (int): frame number to filter by.
        baseline (str): baseline number to filter by.

    Returns:
        list[Item]: a list of matching STAC Items, up to 10.
    """

    search = catalog.search(
    collections = collections,
    filter = f"product:type='{prod_type}' and\
            eofeos:global_coverage_id={global_id} and\
            eofeos:major_cycle_id={major_cycle} and\
            track={track} and\
            frame={frame} and\
            version='{baseline}'",
        
    method = "GET",
    max_items = 10,  
    )

    items = list(search.items())
    return items


def product_search_majorcycle_0M(catalog, collections, prod_type, global_id, major_cycle, track, baseline):
    """
    Searches a STAC catalog for 0M-type product matching a given product type within a specific major cycle and track (0M products do not have frame).

    Args:
        catalog (Client): a pystac_client Client instance connected to the catalog.
        collections (list[str]): list of collection names to search in.
        prod_type (str): product type filter string (e.g. 'AUX_ATT___').
        global_id (int): global coverage ID to filter by (eofeos:global_coverage_id).
        major_cycle (int): major cycle ID to filter by (eofeos:major_cycle_id).
        track (int): track number to filter by.
        baseline (str): baseline number to filter by

    Returns:
        list[Item]: a list of matching STAC Items, up to 10.
    """

    search = catalog.search(
    collections = collections,
    filter = f"product:type='{prod_type}' and\
            eofeos:global_coverage_id={global_id} and\
            eofeos:major_cycle_id={major_cycle} and\
            track={track} and\
            version='{baseline}'",
        
    method = "GET",
    max_items = 10,  
    )

    items = list(search.items())
    return items


def download_product(urls, output_path, disable_bar=False):
    """
    Downloads and unzips a list of products from the given URLs into the specified output path. 
    After the product is unzipped, the .zip file is deleted to save memory.

    The function authenticates using the MAAP_CATALOGUE_ACCESS_TOKEN environment variable.
    Each product is downloaded as a .zip file and then extracted into a subdirectory
    called 'Inputs' within output_path. Both directories are created if they do not exist.

    Args:
        urls (list[str]): list of URLs pointing to the products to download.
        output_path (str): local directory path where downloaded .zip files will be saved and extracted.
        disable_bar (bool): optional, if True, suppresses the tqdm progress bar and prints a completion message instead. Default is False.

    Raises:
        OSError: if the output directories cannot be created or if the deletion of the .zip file cannot be performed.
        requests.exceptions.RequestException: if the HTTP request fails (e.g. authentication error, connection timeout).
        Exception: if the unzipping step fails.
    """

    try:

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    except OSError as e:
        print(f"Error creating output directories: {e}")

    
    for url in urls:

        try:

            filename = url.rsplit('/', 1)[-1] + '.zip'
            output_filename = os.path.join(output_path, filename)
            unzip_path = os.path.join(output_path, 'Inputs')
            token = os.environ["MAAP_CATALOGUE_ACCESS_TOKEN"]
    
    
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
    
            file_size = int(response.headers.get('content-length', 0))
    
            chunk_size = 8 * 1024 * 1024 
            with open(output_filename, "wb") as f, tqdm(
                desc = url.rsplit('/', 1)[-1] + '.zip',
                total = file_size,
                unit = 'iB',
                unit_scale = True,
                unit_divisor = 1024,
                disable = disable_bar,
              ) as bar:
              for chunk in response.iter_content(chunk_size = chunk_size):
                read_size = f.write(chunk)
                bar.update(read_size)
    
            if (disable_bar): 
              print(f"File downloaded successfully to {output_filename}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")

        
        try:
            print(f"Unzipping {url.rsplit('/', 1)[-1]} into {unzip_path}")
            import subprocess
            subprocess.run(["unzip", "-q", "-o", output_filename, "-d", unzip_path], check=True)

        except Exception as e:
            print(f"Error unzipping file: {e}")

        
        try:
            print(f'Removing {url.rsplit('/', 1)[-1] + '.zip'} to save memory')
            os.remove(output_filename)
            print(f"Deleted {output_filename}")
            
        except OSError as e:
            print(f"Error deleting file: {e}")

            

def extract_xml_fields(url, keys=None):
    """
    Fetches an OWC/GeoJSON metadata document from a URL and extracts field values
    by recursively searching all nested dicts and lists under 'properties'.

    Args:
        url (str): URL of the OWC JSON document to fetch and parse.
        keys (list[str]): List of field names to extract. Each key is searched
                          recursively across the entire properties tree.
                          First match wins (depth-first).

    Returns:
        dict: Mapping each requested key to its extracted value (None if not found).
    """
    
    keys    = keys or []
    results = {k: None for k in keys}

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))

    props = data.get("properties", {})

    def _find(obj, key):
        """Recursively search dicts and lists for a given key."""
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = _find(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find(item, key)
                if found is not None:
                    return found
        return None

    for key in keys:
        results[key] = _find(props, key)

    return results



def get_raw_repeat_cycle(ID, output_path):
    """
    Downloads all raw input products needed for a single repeat cycle.

    Given a L0S product ID, this function retrieves and downloads the corresponding
    0S and 0M products along with the required auxiliary files (AUX_ATT, AUX_ORB, AUX_TEC)
    covering the same datetime interval. All products are downloaded and extracted into output_path.

    Args:
        ID (str): STAC item ID of the reference L0S product (e.g. 'BIO_S3_RAW__0S_...').
        output_path (str): local directory path where all products will be downloaded and extracted.

    Raises:
        IndexError: if no item matching the given ID is found in the catalog.
        requests.exceptions.RequestException: if any product download fails.
    """

    catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
    catalog = Client.open(catalog_url)

    search = catalog.search(
    collections = ["BiomassLevel0","BiomassLevel0IOC"],
    ids=ID,
    method = "GET",
    max_items = 1,  
    )

    items = list(search.items())
    L0_S = items

    start_date = L0_S[0].properties.get("start_datetime")
    end_date = L0_S[0].properties.get("end_datetime")
    prod_type_0S = L0_S[0].properties.get("product:type")
    prod_type_0M = prod_type_0S[:-1] + "M"
    metadata_url = L0_S[0].assets['metadata_ogc_17_003r2'].href

    datetime = [start_date, end_date]

    # Get the baseline from the product metadata file
    values = extract_xml_fields(
    url = metadata_url,
    keys = [
        "productVersion",
    ]
    )

    baseline = values['productVersion']

    
    # Get the 0M product
    collections = ["BiomassLevel0","BiomassLevel0IOC"]
    L0_M = product_search(catalog, collections, prod_type_0M, datetime, baseline)

    # Get the AUX_ATT
    collections = ["BiomassAuxIOC","BiomassAux","BiomassAuxRest"]
    BIO_AUX_ATT = product_search(catalog, collections, 'AUX_ATT___', datetime, baseline)

    # Get the AUX_ORB
    BIO_AUX_ORB = product_search(catalog, collections, 'AUX_ORB___', datetime, baseline)

    # Get the AUX_TEC
    BIO_AUX_TEC = product_search(catalog, collections, 'AUX_TEC___', datetime, baseline)


    total_prod = L0_S + L0_M + BIO_AUX_ATT + BIO_AUX_ORB + BIO_AUX_TEC

    
    # Print a table resuming the matching values
    labels={
    "productVersion":   "Baseline",
    }
    lines  = [f"{labels.get(k, k)}: {v}" for k, v in values.items()]
    max_len = max(len(line) for line in lines)
    width   = max_len + 4

    top    = "┌" + "─" * width + "┐"
    bottom = "└" + "─" * width + "┘"
    sep    = "├" + "─" * width + "┤"

    print(f'Matching parameters for {ID}')

    print(top)
    for i, line in enumerate(lines):
        print(f"│  {line:<{max_len}}  │")
        if i < len(lines) - 1:
            print(sep)
    print(bottom)

    print('\n')
    print(f'Products to be downloaded in {output_path} ({len(total_prod)}):')

    for item in total_prod:
        print(f"- {item.id}")
    print('\n')


    # Get the urls for the download
    L0_S_url = L0_S[0].assets['product'].href
    L0_M_url = L0_M[0].assets['product'].href
    AUX_ATT_url = BIO_AUX_ATT[0].assets['product'].href
    AUX_ORB_url = BIO_AUX_ORB[0].assets['product'].href
    AUX_TEC_url = BIO_AUX_TEC[0].assets['product'].href

    urls = [L0_S_url, L0_M_url, AUX_ATT_url, AUX_ORB_url, AUX_TEC_url]

    # Download and unzip products in the specified path
    download_product(urls, output_path, disable_bar=False)

    return L0_S, L0_M, BIO_AUX_ATT, BIO_AUX_ORB, BIO_AUX_TEC


def get_raws_major_cycle(ID, output_path):
    """
    Downloads all raw input products needed for a full major cycle.

    Given a L0S product ID, this function extracts the global coverage ID, major cycle ID,
    track, and frame from its metadata, then retrieves all 0S products sharing the same
    major cycle, track, and frame. For each 0S product, the corresponding auxiliary files
    (AUX_ATT, AUX_ORB, AUX_TEC) are fetched. All matching 0M products for the same major cycle
    and track are also retrieved. All products are downloaded and extracted into output_path.

    Args:
        ID (str): STAC item ID of the reference L0S product (e.g. 'BIO_S3_RAW__0S_...').
        output_path (str): local directory path where all products will be downloaded and extracted.

    Raises:
        IndexError: if no item matching the given ID is found in the catalog.
        KeyError: if required metadata fields are missing from the product XML.
        requests.exceptions.RequestException: if any product download fails.
    """

    catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
    catalog = Client.open(catalog_url)

    search = catalog.search(
    collections = ["BiomassLevel0","BiomassLevel0IOC"],
    ids=ID,
    method = "GET",
    max_items = 1,  
    )

    items = list(search.items())
    L0_S = items
    prod_type_0S = L0_S[0].properties.get("product:type")
    prod_type_0M = prod_type_0S[:-1] + "M"

    
    # Get the remaining 0S products
    collections = ["BiomassLevel0","BiomassLevel0IOC"]
    metadata_url = L0_S[0].assets['metadata_ogc_17_003r2'].href

    values = extract_xml_fields(
    url = metadata_url,
    keys = [
        "wrsLongitudeGrid",
        "wrsLatitudeGrid",
        "globalCoverageID",
        "repeatCycleID",
        "majorCycleID",
        "productVersion",
    ]
    )

    global_id = int(values['globalCoverageID'])
    major_cycle = int(values['majorCycleID'])
    track = int(values['wrsLongitudeGrid'])
    frame = int(values['wrsLatitudeGrid'])
    baseline = values['productVersion']
    
    L0S_items = product_search_majorcycle(catalog, collections, prod_type_0S, global_id, major_cycle, track, frame, baseline)

    # Get the 0M products
    L0M_items = product_search_majorcycle_0M(catalog, collections, prod_type_0M, global_id, major_cycle, track, baseline)

    # Get the AUX products
    collections = ["BiomassAuxIOC","BiomassAux","BiomassAuxRest"]

    ATT_items = []
    ORB_items = []
    TEC_items = []
    for item in L0S_items:
        start_date = item.properties.get("start_datetime")
        end_date = item.properties.get("end_datetime")
    
        datetime = [start_date, end_date]
    
        ATT_items.extend(product_search(catalog, collections, 'AUX_ATT___', datetime, baseline))
        ORB_items.extend(product_search(catalog, collections, 'AUX_ORB___', datetime, baseline))
        TEC_items.extend(product_search(catalog, collections, 'AUX_TEC___', datetime, baseline))
    

    total_prod = L0S_items + L0M_items + ATT_items + ORB_items + TEC_items
    
    # Print a table resuming the matching values
    labels={
    "globalCoverageID": "Global Coverage",
    "majorCycleID":     "Major Cycle",
    "repeatCycleID":    "Repeat Cycle",
    "wrsLongitudeGrid": "Track",
    "wrsLatitudeGrid":  "Frame", 
    "productVersion":   "Baseline",
    }

    lines  = [f"{labels.get(k, k)}: {v}" for k, v in values.items()]
    max_len = max(len(line) for line in lines)
    width   = max_len + 4

    top    = "┌" + "─" * width + "┐"
    bottom = "└" + "─" * width + "┘"
    sep    = "├" + "─" * width + "┤"

    print(f'Matching parameters for {ID}')

    print(top)
    for i, line in enumerate(lines):
        print(f"│  {line:<{max_len}}  │")
        if i < len(lines) - 1:
            print(sep)
    print(bottom)

    print('\n')
    print(f'Products to be downloaded in {output_path} ({len(total_prod)}):')

    for item in total_prod:
        print(f"- {item.id}")
    print('\n')
    
    
    # Getting the urls of the products
    L0S_urls = [url.assets["product"].href for url in L0S_items]
    L0M_urls = [url.assets["product"].href for url in L0M_items]
    ATT_urls = [url.assets["product"].href for url in ATT_items]
    ORB_urls = [url.assets["product"].href for url in ORB_items]
    TEC_urls = [url.assets["product"].href for url in TEC_items]

    # Download and extract the products
    download_product(L0S_urls, output_path, disable_bar=False)
    download_product(L0M_urls, output_path, disable_bar=False)
    download_product(ATT_urls, output_path, disable_bar=False)
    download_product(ORB_urls, output_path, disable_bar=False)
    download_product(TEC_urls, output_path, disable_bar=False)
