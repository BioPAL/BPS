compute_bps_tag() {
   local version="$1"
   local major
   local minor
   local patch

   major=$(echo "$version" | awk -F"." '{print $1}')
   if [ ${#major} -lt 2 ]
   then
      major="0${major}"
   fi
   minor=$(echo "$version" | awk -F"." '{print $2}')
   patch=$(echo "$version" | awk -F"." '{print $3}')
   if [ ${#patch} -gt 1 ]
   then
      patch=${patch:0:1}
   fi

   if [ -n "$CI_COMMIT_TAG" ]; then
      echo "${major}.${minor}${patch}"
   elif [ "$BPS_WEEKLY" == "true" ]; then
      echo "${major}.${minor}${patch}-weekly-${CI_COMMIT_SHORT_SHA}"
   else
      echo "${major}.${minor}${patch}-dev-${CI_COMMIT_SHORT_SHA}"
   fi
}

export BPS_VERSION=$(grep  -m 1 "__version__" $PWD/bps-common/bps/common/__init__.py | awk -F '"' '{print $2}')
export BPS_TAG=$(compute_bps_tag "$BPS_VERSION")

export AREPYTOOLS_VERSION="1.8.1"
export AREPYTOOLS_URL="${INTERNAL_CONDA_CHANNEL_URL}/${INTERNAL_CONDA_CHANNEL_NAME}/noarch/arepytools-${AREPYTOOLS_VERSION}-py_0.conda"
export AREPYEXTRAS_RUNNER_URL="${INTERNAL_CONDA_CHANNEL_URL}/${INTERNAL_CONDA_CHANNEL_NAME}/noarch/arepyextras-runner-1.0.2-py_0.tar.bz2"

# Additional tools
export AREPYAPPS_COPERNICUS_DEM_URL="${INTERNAL_CONDA_CHANNEL_URL}/${INTERNAL_CONDA_CHANNEL_NAME}/noarch/arepyextras-copernicus_dem_extractor-3.1.4-py_0.tar.bz2"
export AREPYAPPS_QUALITY_URL="${INTERNAL_CONDA_CHANNEL_URL}/${INTERNAL_CONDA_CHANNEL_NAME}/noarch/arepyextras-quality-1.2.8-py_0.tar.bz2"
