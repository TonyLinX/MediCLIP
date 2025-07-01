#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing now..."
    pip install gdown || { echo "Failed to install gdown"; exit 1; }
else
    echo "gdown is already installed."
fi

echo "Downloading files..."

gdown --id 1PyvMXdNEVY86BY1PV8yKhPVS30TAmS6X -O file1.zip
unzip file1.zip

gdown --id 1kldE-5_wXaN-JR_8Y_mRCKQ6VZiyv3km -O file2.zip
unzip file2.zip

gdown --id 1pVYRipGC2VqjYP-wHdDFR-lLf7itLiUi -O file3.zip
unzip file3.zip
