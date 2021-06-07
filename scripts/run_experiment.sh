#!/bin/bash
# Run basic experiment

EXPERIMENT_DIR=$2
SEGMENTER=$1
LAST_1000=FALSE

# Run segmenter
if (( $# < 3 ))
then
	echo "Running '$SEGMENTER' segmenter with no additional arguments"
else
	echo "Running '$SEGMENTER' segmenter with additional arguments '${@:3}'"
fi
python -m segmenter.$SEGMENTER -o $EXPERIMENT_DIR/segmented.txt ${@:3} $EXPERIMENT_DIR/prepared.txt

# Evaluate excluding first 1000 lines
if $LAST_1000
then
	echo "Removing first 1000 lines for evaluation"
	sed -ie 1,1000d $EXPERIMENT_DIR/gold.txt
	sed -ie 1,1000d $EXPERIMENT_DIR/prepared.txt
	sed -ie 1,1000d $EXPERIMENT_DIR/segmented.txt
fi

# Evaluate segmentation using wordseg
echo "Calculating statistics"
wordseg-eval -r $EXPERIMENT_DIR/prepared.txt -s $EXPERIMENT_DIR/seg_errors.json $EXPERIMENT_DIR/segmented.txt $EXPERIMENT_DIR/gold.txt > $EXPERIMENT_DIR/eval.txt
python scripts/calculate_errors.py $EXPERIMENT_DIR/segmented.txt $EXPERIMENT_DIR/gold.txt $EXPERIMENT_DIR/prepared.txt >> $EXPERIMENT_DIR/eval.txt
less $EXPERIMENT_DIR/eval.txt | column -t
python scripts/print_table_scores.py $EXPERIMENT_DIR/eval.txt
echo "Written error summary to $EXPERIMENT_DIR/seg_errors.json"
