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
        --execute 1.apply-classifier-pdx.ipynb

# Notebook 2 - Evaluate the prediction performance
jupyter nbconvert --to=html \
        --FilesWriter.build_directory=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=$execute_time \
        --execute 2.evaluate-classifier-pdx.ipynb

# Convert all notebooks to python scripts
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts *.ipynb
