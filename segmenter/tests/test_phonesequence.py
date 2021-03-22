""" Run tests for the PhoneSequence class """

from segmenter.phonesequence import PhoneSequence 

def test_init():
    phones = ['a', 'b']
    boundaries = [False, True]

    seq = PhoneSequence(phones)

    assert(seq.phones == phones)
    assert(seq.boundaries == boundaries)

def test_get_words():
    phones = ['a', 'b', 'c', 'd']
    boundaries = [False, True, True, True]

    seq = PhoneSequence(phones)
    seq.boundaries = boundaries
    words = seq.get_words()

    assert(words == [['a', 'b'], ['c'], ['d']])

def test_str():
    phones = ['a', 'b', 'c', 'd']
    boundaries = [False, True, True, True]

    seq = PhoneSequence(phones)
    seq.boundaries = boundaries

    assert(str(seq) == "ab c d")

def test_len():
    phones = ['a', 'b', 'c', 'd']

    seq = PhoneSequence(phones)

    assert(len(seq) == 4)

"""
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
    """