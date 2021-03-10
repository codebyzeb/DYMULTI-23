#!/bin/bash
# Run basic experiment

EXPERIMENT_DIR="experiments/experiment"
TRANSCRIPT="/Users/zebulon/Documents/UniDocs/Year4/Project/Repository/data/br-phonemes.txt"
CLEAN=TRUE
SEGMENTER=${1:-baseline}

if $CLEAN
then
    # Clearn directory
    echo "Cleaning directory '$EXPERIMENT_DIR'"
    rm -r $EXPERIMENT_DIR
    mkdir $EXPERIMENT_DIR

    # Prepare transcript for segmentation using wordseg
    echo "Preparing transcript for segmentation in '$EXPERIMENT_DIR'"
    wordseg-prep -P -u phone --gold $EXPERIMENT_DIR/gold.txt $TRANSCRIPT > $EXPERIMENT_DIR/prepared.txt
fi

# Run segmenter
if (( $# < 1 ))
then
	echo "Running baseline segmenter with no additional arguments"
elif (( $# < 2 ))
then
	echo "Running '$SEGMENTER' segmenter with no additional arguments"
else
	echo "Running '$SEGMENTER' segmenter with additional arguments '${@:2}'"
fi
python -m segmenter.$SEGMENTER -o $EXPERIMENT_DIR/segmented.txt ${@:2} $EXPERIMENT_DIR/prepared.txt

# Evaluate segmentation using wordseg
echo "Calculating statistics"
wordseg-eval $EXPERIMENT_DIR/segmented.txt $EXPERIMENT_DIR/gold.txt > $EXPERIMENT_DIR/eval.txt
less $EXPERIMENT_DIR/eval.txt | column -t
