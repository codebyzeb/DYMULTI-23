""" Run tests for the utility methods """

from segmenter import utils 

def test_segmented_utterance_to_boundaries():

    segmented = ["a" , " ", "b", "c", " "]

    boundaries = utils.segmented_utterance_to_boundaries(segmented)

    assert(boundaries == [True, False, True])

def test_boundaries_to_segmented_utterance():

    utterance = "a b c"
    boundaries = [True, False, True]

    segmented = utils.boundaries_to_segmented_utterance(utterance, boundaries)

    assert(segmented == ["a" , " ", "b", "c", " "])

def test_split_segmented_utterance():

    segmented = [" ", "a", "b", " ", "c", "d", " "]

    words = utils.split_segmented_utterance(segmented)

    assert(words == [["a", "b"], ["c", "d"]])