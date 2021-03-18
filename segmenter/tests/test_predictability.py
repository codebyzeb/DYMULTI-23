""" Run tests for the PhoneStats class """

import pytest

from segmenter.predictability import PredictabilityModel
from segmenter.phonestats import PhoneStats

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_phonestats_sets_correct_properties():

    model = PredictabilityModel(ngram_length=2, increase=False, measure="ent", reverse=True, phonestats=None)

    assert(model.ngram_length == 2)
    assert(model.increase == False)
    assert(model.measure == "ent")
    assert(model.reverse == True)
    assert(model._updatephonestats)
    assert(not model._phonestats is None)
    assert(model._phonestats.max_ngram == 3)

def test_init_with_phonestats_saves_reference():

    phonestats = PhoneStats(2)

    model = PredictabilityModel(phonestats=phonestats)

    assert(not model._updatephonestats)
    assert(model._phonestats == phonestats)

def test_init_with_ngram_less_than_one_raises_value_error():

    with pytest.raises(ValueError, match=".*non-positive n-gram.*"):
        PredictabilityModel(ngram_length=0)

def test_init_with_invalid_measure_raises_value_error():

    with pytest.raises(ValueError, match=".*unknown predictability measure.*"):
        PredictabilityModel(measure="fake")

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = model = PredictabilityModel(ngram_length=2, increase=False, measure="ent", reverse=True, phonestats=None)

    s = str(model)

    assert(s == "Predictability(N: 2,Decrease of Reverse Boundary Entropy)")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = PredictabilityModel()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b c", "b a c b"]
    model = PredictabilityModel()

    list(model.segment(text, update_model=False))

    assert(model._phonestats.ntokens[1] == 0)

def test_segment_update_model_true_updates_model():

    text = ["a b b c", "b a c b"]
    model = PredictabilityModel()

    list(model.segment(text, update_model=True))

    assert(model._phonestats.ntokens[1] == 8)

def test_segment_updates_phonestats_when_not_provided():
    """ If not provided a PhoneStats object, a local one is created an updated at each utterance """

    text = ["a b c d", "e f g h"]
    model = PredictabilityModel()

    list(model.segment(text, update_model=True))

    assert(model._phonestats.ntokens[1] == 8)
    assert(model._phonestats.ntokens[2] == 6)
    assert(model._phonestats.ngrams[2][('a', 'b')] == 1)

def test_segment_does_not_update_phonestats_when_provided():
    """ If provided a PhoneStats object, the model should not update it """

    text = ["a b c d", "e f g h"]
    phonestats = PhoneStats()
    model = PredictabilityModel(phonestats=phonestats)

    list(model.segment(text, update_model=True))

    assert(model._phonestats.ntokens[1] == 0)
    assert(model._phonestats.ngrams[1][('a',)] == 0)

# TODO: Write tests to ensure the model segments when unpredictability increases or decreases