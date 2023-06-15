# Word Segmentation using Multiple Cues

This repository contains the code for *"Word Segmentation and Lexicon Learning from Child-Directed Speech Using Multiple Cues"* (pending acceptance into the Journal of Child Language). 

## Prerequisites

This project depends on the [wordseg](https://github.com/bootphon/wordseg) library and uses [conda](https://docs.conda.io/) to manage the virtual environment. After cloning or downloading this repository, follow the instructions for installing *wordseg* in a virtual conda environment named `wordseg`. The relevant packages for this project can then be installed as follows:

    cd REPOSITORY_DIRECTORY/
    pip install -r requirements.txt

## Data

In `data/` I redistribute the modified BR corpus used in the studies of Çöltekin, accessed from his [repository](https://bitbucket.org/coltekin/seg/) on 29/01/2021. The folder contains the following files:
* `br-phonemes.txt` containing the word-separated phonemes ready to be processed by `wordseg`.
* `br-stress.txt` containing stress alignment for the BR corpus.
* `br-text.txt` containing an orthographic transcription of the corpus.

## Preparing the data

To prepare the corpus for segmentation, run the following:

    sh scripts/prepare_experiment.sh experiment data/br-phonemes.txt data/br-stress.txt

This will create a new folder `experiment/` containing the unsegmented phonemes in `prepared.txt`, the correct segmentation in `gold.txt` and the stress alignment in `stress.txt`.

## Running a segmentation model

The following sections give example commands for how to run various segmentation models provided by this project. These all use the `run_experiment.sh` script, resulting in the files `segmented.txt` and `eval.txt` in the experiment directory containing the segmented output and relevant evaluation metrics, respectively.

To get more information on which arguments can be passed to each model, run the model with `-h`, such as:

    python -m segmenter.baseline -h

### Baseline model

The baseline model of [Brent (1999)](https://link.springer.com/article/10.1023/A:1007541817488), called BASELINE in this study, can be run as follows:

    sh scripts/run_experiment.sh baseline experiment -v -P 0.5

### MULTICUE models

The multiple-cue model of [Çöltekin and Nerbonne (2014)](https://www.aclweb.org/anthology/W14-0505.pdf), called MULTICUE-14 in this study, can be run as follows:

    sh scripts/run_experiment.sh multicue experiment -v -n 3,1 -d both -P ent,mi,bp -L both -X experiment/stress.txt

The multiple-cue model of [Çöltekin (2017)](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12454), called MULTICUE-17 in this study, can be run as follows:

    sh scripts/run_experiment.sh multicue experiment -v -n 4,3,2,1 -d both -P sv

The new MULTICUE-21 model presented in this study, can be run as follows:

    sh scripts/run_experiment.sh multicue experiment -v -n 4,3,2,1 -d both -P sv,bp -L both

Any of these models can be run with the new weighted majority-vote algorithm variants presented in this study using the `-W` flag as follows:

* `-W precision` for precision weights
* `-W recall` for recall weights
* `-W f1` for F-score weights

### PHOCUS models

The model of [Venkataraman (2001)](https://direct.mit.edu/coli/article/27/3/351/1717/A-Statistical-Model-for-Word-Discovery-in), called PHOCUS-1 in this study, can be run as follows:

    sh scripts/run_experiment.sh probabilistic experiment -v -n 0 -m venk

The PHOCUS-1S model of [Blanchard et al. (2010)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.180.2891&rep=rep1&type=pdf) can be run as follows:

    sh scripts/run_experiment.sh probabilistic experiment -v -n 0 -m blanch

### DYMULTI models

The DYMULTI model presented in this study can be run with any of the cues of the MULTICUE models. For instance, DYMULTI-23 can be run as follows:

    sh scripts/run_experiment.sh dynamicmulticue experiment -v -n 4,3,2,1 -d both -P sv,bp -L both -a 0

To add the lexical recognition process, simply change the value after the `-a` flag to a different value for alpha.

## Running several shuffles

In my results, I often report scores averaged over 10 shuffles of the input corpus. This was achieved by running `python scripts/run_shuffles.py`. This script is not as polished as the others, but in its current form it will run DYMULTI-23 with alpha=0 for ten shuffles of whatever corpus is prepared in the directory `experiment/`. It can be manually adjusted to run other models in other directories, if need be.

## Analysis 

The python notebooks in `analysis/` contain the code used to generate the figures used in the study, as well as other calculations. 
