#!/bin/bash
# Run basic experiment

EXPERIMENT_DIR=$1
TRANSCRIPT=$2

# Clearn directory
echo "Cleaning directory '$EXPERIMENT_DIR'"
rm -r $EXPERIMENT_DIR
mkdir $EXPERIMENT_DIR

if (( $# > 2 ))
then
    echo "Copying stress file to '$EXPERIMENT_DIR/stress.txt'"
    cp $3 $EXPERIMENT_DIR/stress.txt
fi

# Prepare transcript for segmentation using wordseg
echo "Preparing transcript for segmentation in '$EXPERIMENT_DIR'"
wordseg-prep -P -u phone --gold $EXPERIMENT_DIR/gold.txt $TRANSCRIPT > $EXPERIMENT_DIR/prepared.txt
