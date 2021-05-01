""" Run tests for the PhoneStats class """

import pytest

from segmenter.peakmodels import StressModel
from segmenter.phonestats import PhoneStats
from segmenter.phonesequence import PhoneSequence

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_stressstats_sets_correct_properties():

    model = StressModel(ngram_length=2, increase=False, right=True, stressstats=None)

    assert(model.ngram_length == 2)
    assert(model.increase == False)
    assert(model.right == True)
    assert(model._updatestress)
    assert(not model._stressstats is None)
    assert(model._stressstats.max_ngram == 3)

def test_init_with_stressstats_saves_reference():

    stressstats = PhoneStats(2)

    model = StressModel(stressstats=stressstats)

    assert(not model._updatestress)
    assert(model._stressstats == stressstats)

def test_init_with_ngram_less_than_one_raises_value_error():

    with pytest.raises(ValueError, match=".*non-positive n-gram.*"):
        StressModel(ngram_length=0)

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = StressModel(ngram_length=2, increase=False, right=True, stressstats=None)

    s = str(model)

    assert(s == "StressModel(N: 2,Decrease of Right Stress)")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = StressModel()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b c", "b a c b"]
    stress = ["0 0 0 0", "1 1 1 1"]
    model = StressModel()

    list(model.segment(text, update_model=False, stress_lines=stress))

    assert(model._stressstats.ntokens[1] == 0)

def test_segment_updates_phonestats_when_not_provided():
    """ If not provided a PhoneStats object, a local one is created an updated at each utterance """

    text = ["a b c d", "e f g h"]
    stress = ["0 0 0 0", "1 1 1 1"]
    model = StressModel()

    list(model.segment(text, update_model=True, stress_lines=stress))

    assert(model._stressstats.ntokens[1] == 12)
    assert(model._stressstats.ntokens[2] == 10)
    assert(model._stressstats.ngrams['00'] == 3)

def test_segment_does_not_update_phonestats_when_provided():
    """ If provided a PhoneStats object, the model should not update it """

    text = ["a b c d", "e f g h"]
    stress = ["0 0 0 0", "1 1 1 1"]
    stressstats = PhoneStats(max_ngram=2)
    model = StressModel(stressstats=stressstats)

    list(model.segment(text, update_model=True, stress_lines=stress))

    assert(model._stressstats.ntokens[1] == 0)
    assert(model._stressstats.ngrams['0'] == 0)

def test_segmented_utterance_has_correct_number_of_boundaries():
    
    model = StressModel()
    utterance = PhoneSequence("a b c d".split(' '), stress=['0', '1', '2', '1'])

    segmented = model.segment_utterance(utterance, update_model=False)

    assert(len(segmented.boundaries) == len(utterance.boundaries))

# TODO: Write tests to ensure the model segments when stress boundary probability increases or decreases
