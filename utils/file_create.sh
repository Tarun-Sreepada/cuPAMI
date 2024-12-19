#!/bin/bash

# Compile the C++ program
g++ -O3 -o file_generator file_generator.cpp


save_dir="../datasets/synthetic/transactional"
# Create the synthetic directory if it does not exist
mkdir -p $save_dir



# Check if compilation was successful
if [[ $? -ne 0 ]]; then
    echo "Compilation failed."
    exit 1
fi

# File sizes and shapes
sizes=("1000M" "1100M" "1200M" "1300M")
shapes=("square" "triangle")
delimiter=","

# Generate files
for size in "${sizes[@]}"; do
    for shape in "${shapes[@]}"; do
        fileName="${shape}_${size}.csv"

        # Generate the file
        fileName="${save_dir}/${fileName}"

        echo "Generating ${fileName}..."
        if [[ "$shape" == "square" ]]; then
            echo "$fileName" | ./file_generator $fileName $size $delimiter 1 > /dev/null
        elif [[ "$shape" == "triangle" ]]; then
            echo "$fileName" | ./file_generator $fileName $size $delimiter 2 > /dev/null
        fi
    done
done

echo "File generation complete."
