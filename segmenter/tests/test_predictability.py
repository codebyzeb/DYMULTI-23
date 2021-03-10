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

# Could make the following tests more general to capture different ngram lengths, measures and setting 
# reverse=True to capture all possible settings...

def test_increasing_model_segments_at_increase_of_unpredictability_unigrams():

    text = ["a b b c", "b a c b"]
    model = PredictabilityModel(ngram_length=1, increase=True, measure="ent", reverse=False)

    # Train model
    list(model.segment(text, update_model=True))

    # Test model
    segmented = list(model.segment(text, update_model=False))

    # The model only segments where boundary entropy has increased before a position
    assert(segmented[0] == "ab bc")
    assert(model._phonestats._boundary_entropy(['b']) > model._phonestats._boundary_entropy(['a']))

    assert(segmented[1] == "bac b")
    assert(not model._phonestats._boundary_entropy(['a']) > model._phonestats._boundary_entropy(['b']))
    assert(model._phonestats._boundary_entropy(['c']) > model._phonestats._boundary_entropy(['a']))

def test_increasing_model_segments_at_increase_of_unpredictability_bigrams():

    text = ["a b b b c", "c c a b b"]
    model = PredictabilityModel(ngram_length=2, increase=True, measure="ent", reverse=False)

    # Train model
    list(model.segment(text, update_model=True))

    # Test model
    segmented = list(model.segment(text, update_model=False))

    # The model only segments where boundary entropy has increased before a position
    assert(segmented[0] == "abb bc")
    assert(model._phonestats._boundary_entropy(['b','b']) > model._phonestats._boundary_entropy(['a','b']))

    assert(segmented[1] == "ccab b")
    assert(not model._phonestats._boundary_entropy(['c', 'a']) > model._phonestats._boundary_entropy(['c','c']))
    assert(model._phonestats._boundary_entropy(['a','b']) > model._phonestats._boundary_entropy(['c','a']))

def test_decreasing_model_segments_at_decrease_of_unpredictability():

    text = ["a b b c", "b a c b"]
    model = PredictabilityModel(ngram_length=1, increase=False, measure="ent", reverse=False)

    # Train model
    list(model.segment(text, update_model=True))

    # Test model
    segmented = list(model.segment(text, update_model=False))

    # The model only segments where boundary entropy has decreased after a position
    assert(segmented[0] == "abb c")
    assert(not model._phonestats._boundary_entropy(['b']) < model._phonestats._boundary_entropy(['a']))
    assert(not model._phonestats._boundary_entropy(['b']) < model._phonestats._boundary_entropy(['b']))
    assert(model._phonestats._boundary_entropy(['c']) < model._phonestats._boundary_entropy(['b']))

    assert(segmented[1] == "b acb")
    assert(model._phonestats._boundary_entropy(['a']) < model._phonestats._boundary_entropy(['b']))
    assert(not model._phonestats._boundary_entropy(['c']) < model._phonestats._boundary_entropy(['a']))
    assert(not model._phonestats._boundary_entropy(['b']) < model._phonestats._boundary_entropy(['c']))
