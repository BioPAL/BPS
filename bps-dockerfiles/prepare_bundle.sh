PREPARE_BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${PREPARE_BUNDLE_DIR}/.." && pwd)

function download(){
	local folder="$1"
	shift 1
	local condapkgs=("$@")

	cd $folder

	if [ $folder == "l1_framing_processor" ]
	then
		echo "Downloading eocfi libs for $folder"
		mkdir -p lib
		cd lib
		curl $EOCFI_LIBS_LNX --show-error --fail -q -o BPS-EOCFI-4.20-CPPLIB-LINUX64_LEGACY.tar.gz
		tar -xvzf BPS-EOCFI-4.20-CPPLIB-LINUX64_LEGACY.tar.gz
		rm BPS-EOCFI-4.20-CPPLIB-LINUX64_LEGACY.tar.gz
		cd ..
	fi

	mkdir pkgs
	cd pkgs
	mkdir noarch
	mkdir linux-64

	for i in "${condapkgs[@]}"; do
		if [[ $i == *"http"* ]] 
		then
			echo "Downloading $i for $folder"
			cd noarch && curl --show-error --fail -q -O ${i} && cd -
		else
			for file in "$REPO_ROOT/$i/dist"/*; do
				echo "Copying $file for $folder"
				if [[ $file == *"py_0"* ]]
				then
					cp $file noarch
				else
					cp $file linux-64
				fi
			done
		fi
	done

	cd ../..
	rm -f $folder/.gitignore $folder/.dockerignore
	cp -r $folder bundle/
	rm -rf $folder/pkgs
	rm -rf $folder/bin
	rm -rf bundle/$folder/Dockerfile
}

mkdir bundle
download prereq "bps-common" "bps-transcoder" "$AREPYTOOLS_URL" "$AREPYEXTRAS_RUNNER_URL"
download l1_framing_processor "bps-l1_framing_processor"
download l1_processor "bps-l1_binaries" "bps-l1_core_processor" "bps-l1_processor" "bps-l1_pre_processor"
download l2a_processor "bps-l2a_processor"
download l2b_fd_processor "bps-l2b_fd_processor"
download l2b_fh_processor "bps-l2b_fh_processor"
download l2b_agb_processor "bps-l2b_agb_processor"
download stack_processor "bps-stack_binaries" "bps-stack_cal_processor" "bps-stack_coreg_processor" "bps-stack_pre_processor" "bps-stack_processor"

cp bundle/prereq/pkgs/noarch/* bundle/l1_framing_processor/pkgs/noarch
cp bundle/prereq/pkgs/noarch/* bundle/l1_processor/pkgs/noarch
cp bundle/prereq/pkgs/noarch/* bundle/l2a_processor/pkgs/noarch
cp bundle/prereq/pkgs/noarch/* bundle/l2b_fd_processor/pkgs/noarch
cp bundle/prereq/pkgs/noarch/* bundle/l2b_fh_processor/pkgs/noarch
cp bundle/prereq/pkgs/noarch/* bundle/l2b_agb_processor/pkgs/noarch
cp bundle/prereq/pkgs/noarch/* bundle/stack_processor/pkgs/noarch

mkdir bundle/copernicus-dem-index-generator
cd bundle/copernicus-dem-index-generator
curl --show-error --fail -q -O ${AREPYTOOLS_URL} 
curl --show-error --fail -q -O ${AREPYAPPS_COPERNICUS_DEM_URL}
cp ../../bundle_resources/install_and_run.sh .
cp ../../bundle_resources/DemIndex.xsd .
cd ../..

mkdir bundle/quality-tools
cd bundle/quality-tools
curl --show-error --fail -q -O ${AREPYTOOLS_URL}
curl --show-error --fail -q -O ${AREPYAPPS_QUALITY_URL}
cd ../..

cp ../CREDITS.md bundle/
cp ../LICENSE.md bundle/
