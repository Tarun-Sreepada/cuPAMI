#!/bin/bash

# Define the output file name
OUTPUT_FILE="../results/topology.svg"

# Check if lstopo is installed
if ! command -v lstopo &> /dev/null
then
    echo "Error: lstopo is not installed. Install hwloc to use this script."
    exit 1
fi

if [ -f $OUTPUT_FILE ]; then
    echo "Warning: $OUTPUT_FILE already exists. Overwriting it."
    rm $OUTPUT_FILE
fi

# Generate the topology SVG with a transparent background
lstopo --output-format svg $OUTPUT_FILE --no-legend -v


# Notify user of success
if [ $? -eq 0 ]; then
    echo "Topology saved as $OUTPUT_FILE with a transparent background."
else
    echo "Failed to generate topology diagram."
fi
