""" Run tests for the PhoneStats class """

import pytest
import numpy as np

from segmenter.predictability import MultiPredictabilityModel

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_sets_correct_properties():

    model = MultiPredictabilityModel(max_ngram=2, measure="ent", direction="forwards")

    assert(model.max_ngram == 2)
    assert(model.measure == "ent")
    assert(model._phonestats.max_ngram == 3)

def test_init_with_forward_and_unigrams_creates_two_models():

    model = MultiPredictabilityModel(max_ngram=1, measure="ent", direction="forwards")
    
    assert(len(model.models) == 2)
    assert(model.models[0].ngram_length == 1 and model.models[0].increase == True and model.models[0].reverse==False)
    assert(model.models[1].ngram_length == 1 and model.models[1].increase == False and model.models[1].reverse==False)

def test_init_with_both_directions_and_unigrams_creates_four_models():

    model = MultiPredictabilityModel(max_ngram=1, measure="ent", direction="both")
    
    assert(len(model.models) == 4)
    assert(model.models[0].ngram_length == 1 and model.models[0].increase == True and model.models[0].reverse==False)
    assert(model.models[1].ngram_length == 1 and model.models[1].increase == False and model.models[1].reverse==False)
    assert(model.models[2].ngram_length == 1 and model.models[2].increase == True and model.models[2].reverse==True)
    assert(model.models[3].ngram_length == 1 and model.models[3].increase == False and model.models[3].reverse==True)

def test_init_with_ngram_less_than_one_raises_value_error():

    with pytest.raises(ValueError, match=".*non-positive n-gram.*"):
        MultiPredictabilityModel(max_ngram=0)

def test_init_with_invalid_measure_raises_value_error():

    with pytest.raises(ValueError, match=".*unknown predictability measure.*"):
        MultiPredictabilityModel(measure="fake")

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = MultiPredictabilityModel(max_ngram=1, measure="ent", direction="forwards")

    s = str(model)

    assert(s == "MultiPredictability(Predictability(N: 1,Increase of Boundary Entropy), Predictability(N: 1,Decrease of Boundary Entropy))")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = MultiPredictabilityModel()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b b c", "b a c b b"]
    model = MultiPredictabilityModel()

    list(model.segment(text, update_model=False))

    assert(model._phonestats.ntokens[1] == 0)
    assert((model.weights == np.ones(model.num_models)).all())
    assert((model.errors == np.zeros(model.num_models)).all())
    assert(model.num_boundaries == 0)

def test_segment_update_model_true_updates_model():

    text = ["a b b b c", "b a c b b"]
    model = MultiPredictabilityModel()

    list(model.segment(text, update_model=True))

    assert(model._phonestats.ntokens[1] == 14)
    assert((model.weights != np.ones(model.num_models)).any())
    assert((model.errors != np.zeros(model.num_models)).any())
    assert(model.num_boundaries == 8)

def test_segment_updates_phonestats_of_submodels():

    text = ["a b b b c", "b a c b b"]
    model = MultiPredictabilityModel()

    list(model.segment(text, update_model=True))

    for submodel in model.models:
        assert(submodel._phonestats.ntokens[1] == 14)
        assert(submodel._phonestats.ntokens[2] == 12)