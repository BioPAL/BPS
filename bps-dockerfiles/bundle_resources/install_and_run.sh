# Create a conda environment
# - it must have gdal installed
# - python 3.9 is the suggested version
# - it is ok if 'arepytools' package is already installed
conda create -n dem_index_env python=3.9
conda activate dem_index_env
conda install -c conda-forge GDAL=3.8
conda install conda-build

# Install the generate-index script
mkdir -p dem_ch/noarch
cp -r ./bundle/copernicus-dem-index-generator/* dem_ch/noarch 
conda index --verbose ${PWD}/dem_ch
conda config --prepend channels ${PWD}/dem_ch

conda install arepyextras-copernicus_dem_extractor

# Run the generate-index script from the DEM folder
cd folder/to/COP-DEM_GLO-90-DGED
copernicus_dem_extractor generate-index -p . -i demIndex.xml
