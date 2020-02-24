#!/bin/bash
#
# Script to preprocess the Unimib 2016 dataset.
#
# Usage:
#   bash ./convert_unimib2016.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_unimib2016_data.py
#     - convert_unimib2016.sh
#     - remove_uni_colormap.py
#     + unimib2016
#       + UM
#         + UM2016
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./unimib2016"
#mkdir -p "${WORK_DIR}"
#cd "${WORK_DIR}"

# Helper function to download and unpack VOC 2012 dataset.
# download_and_uncompress() {
#   local BASE_URL=${1}
#   local FILENAME=${2}

#   if [ ! -f "${FILENAME}" ]; then
#     echo "Downloading ${FILENAME} to ${WORK_DIR}"
#     wget -nd -c "${BASE_URL}/${FILENAME}"
#   fi
#   echo "Uncompressing ${FILENAME}"
#   tar -xf "${FILENAME}"
# }

# Download the images.
# BASE_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
# FILENAME="VOCtrainval_11-May-2012.tar"

# download_and_uncompress "${BASE_URL}" "${FILENAME}"

cd "${CURRENT_DIR}"

# Root path for PASCAL VOC 2012 dataset.
UNI_ROOT="${WORK_DIR}/UM/UM2016"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${UNI_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${UNI_ROOT}/SegmentationClassRaw"

echo "Removing the color map in ground truth annotations..."
python ./remove_uni_colormap.py \
  --original_uni_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${UNI_ROOT}/JPEGImages"
LIST_FOLDER="${UNI_ROOT}/ImageSets/Segmentation"

echo "Converting UNIMIB 2016 dataset..."
python ./build_unimib2016_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
