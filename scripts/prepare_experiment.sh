#!/bin/bash
# Run basic experiment

EXPERIMENT_DIR=$2
TRANSCRIPT=$1

# Clearn directory
echo "Cleaning directory '$EXPERIMENT_DIR'"
rm -r $EXPERIMENT_DIR
mkdir $EXPERIMENT_DIR

# Prepare transcript for segmentation using wordseg
echo "Preparing transcript for segmentation in '$EXPERIMENT_DIR'"
wordseg-prep -P -u phone --gold $EXPERIMENT_DIR/gold.txt $TRANSCRIPT > $EXPERIMENT_DIR/prepared.txt
