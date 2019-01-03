#!/bin/bash

# Exit on error
set -o errexit

execute_time=10000000

# Run all files in order
# Notebook 1 - Apply the Ras and TP53 classifiers to the input PDX data
jupyter nbconvert --to=html \
        --FilesWriter.build_directory=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 1.apply-classifier.ipynb

# Notebook 2 - Evaluate the prediction performance
jupyter nbconvert --to=html \
        --FilesWriter.build_directory=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 2.evaluate-classifier.ipynb

# Notebook 3 - Explore classifier score assignments
jupyter nbconvert --to=html \
        --FilesWriter.build_directory=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 3.explore-variants.ipynb

# Convert all notebooks to python scripts
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts *.ipynb
