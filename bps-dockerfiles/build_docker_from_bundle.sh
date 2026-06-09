print_usage() {
    echo "Tool to build the bps docker images from the bundle archive"
    echo "  build_docker_from_bundle.sh --bundle=..."
    echo ""
    echo "Options:"
    echo "  -h --help                    print this help message"
    echo ""
    echo "  --bundle=...                 path to bundle tar.gz archive"
    echo "  --test                       optional - if set, avoids saving every docker image as a file"
    echo ""
}

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

# Parse command line arguments
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

while (($#)); do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        --test)
            TESTING=true
            ;;
        --bundle=?*)
            BUNDLE="${1#*=}"
            ;;
        -*)
            die "unknown flag $1"
            ;;
        *)
            break
            ;;
    esac
    shift
done

if [ -z "$BUNDLE" ]
then
      die "Bundle file not specified"
fi

if [ -n "$TESTING" ]; then
    echo "Testing mode"
fi
echo "Bundle file: $BUNDLE"

DOCKERFILES="$PWD"
BPS_VERSION=$(basename ${BUNDLE} .tar.gz | awk -F"bps-bundle-v" '{print $2}')
MAJOR=$(echo $BPS_VERSION | awk -F"." '{print $1}')
if [ ${#MAJOR} -lt 2 ]
then
    MAJOR="0${MAJOR}"
fi
MINOR=$(echo $BPS_VERSION | awk -F"." '{print $2}')
PATCH=$(echo $BPS_VERSION | awk -F"." '{print $3}')
if [ ${#PATCH} -gt 1 ] 
then
    PATCH=${PATCH:0:1}
fi
IMAGETAG="${MAJOR}.${MINOR}${PATCH}"

tar xf $BUNDLE
cd bundle

cp ${DOCKERFILES}/prereq/Dockerfile prereq/
cp ${DOCKERFILES}/l1_framing_processor/Dockerfile l1_framing_processor/
cp ${DOCKERFILES}/l1_processor/Dockerfile l1_processor/
cp ${DOCKERFILES}/stack_processor/Dockerfile stack_processor/
cp ${DOCKERFILES}/l2a_processor/Dockerfile l2a_processor/
cp ${DOCKERFILES}/l2b_fd_processor/Dockerfile l2b_fd_processor/
cp ${DOCKERFILES}/l2b_fh_processor/Dockerfile l2b_fh_processor/
cp ${DOCKERFILES}/l2b_agb_processor/Dockerfile l2b_agb_processor/

cd prereq
IMAGENAME="bps-prereq"
echo "=== Building image: ${IMAGENAME} ==="
docker build -q --no-cache --pull -t ${IMAGENAME}:latest -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null

cd l1_framing_processor
IMAGENAME="bps-l1-framing-processor"
echo "=== Building image: ${IMAGENAME} ==="
docker build -q -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null

cd l1_processor
IMAGENAME="bps-l1-processor"
echo "=== Building image: ${IMAGENAME} ==="
docker build -q -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null


cd stack_processor
IMAGENAME=""bps-stack-processor""
echo "=== Building image: ${IMAGENAME} ==="
docker build -q -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null


cd l2a_processor
IMAGENAME="bps-l2a-processor"
echo "=== Building image: ${IMAGENAME} ==="
docker build -q -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null

cd l2b_fd_processor
IMAGENAME="bps-l2b-fd-processor"
echo "=== Building image: ${IMAGENAME} ==="
docker build -q -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null

cd l2b_fh_processor
IMAGENAME="bps-l2b-fh-processor"
echo "=== Building image: ${IMAGENAME} ==="
docker build -q -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null

cd l2b_agb_processor
IMAGENAME="bps-l2b-agb-processor"
echo "=== Building image: ${IMAGENAME} ==="
docker build -q -t ${IMAGENAME}:${IMAGETAG} . || exit 1
if [ -z "$TESTING" ]; then
    docker save ${IMAGENAME}:${IMAGETAG} | xz -f0 > ${IMAGENAME}-${IMAGETAG}.tar.xz
fi
echo "=== Docker image building completed: ${IMAGENAME} ==="
cd - >/dev/null

cd ..
