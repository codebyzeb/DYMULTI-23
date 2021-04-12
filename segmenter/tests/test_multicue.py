""" Run tests for the MultieCueModel class """

from segmenter.phonesequence import PhoneSequence
import pytest
import numpy as np

from segmenter.baseline import BaselineModel
from segmenter.multicue import MultiCueModel
from segmenter.phonestats import PhoneStats
from segmenter.lexicon import Lexicon
from segmenter.predictability import PredictabilityModel

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_model_raises_value_error():

    with pytest.raises(ValueError, match=".*Cannot initialise MultiCueModel without any sub models.*"):
        MultiCueModel()

def test_init_with_not_a_model_raises_value_error():

    with pytest.raises(ValueError, match=".*not an instance.*"):
        MultiCueModel(models=[5])

def test_init_assigns_correct_initial_values():

    model = MultiCueModel(models=[BaselineModel(), BaselineModel()])

    assert(not model.precision_weights)
    assert(model.num_models == 2)
    assert((model.weights == np.ones(model.num_models)).all())
    assert((model.errors == np.zeros(model.num_models)).all())
    assert(model.num_boundaries == 0)

def init_with_precision_weights_assigns_correct_initial_values():

    model = MultiCueModel(models=[BaselineModel(), BaselineModel()], precision_weights=True)

    assert(model.precision_weights)
    assert(model.num_models == 2)
    assert((model.weights_positive == np.ones(model.num_models)).all())
    assert((model.weights_negative == np.ones(model.num_models)).all())
    assert((model.errors_positive == np.zeros(model.num_models)).all())
    assert((model.errors_negative == np.zeros(model.num_models)).all())
    assert(model.num_boundaries_not_placed == 0)
    assert(model.num_boundaries_placed == 0)

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = MultiCueModel(models=[BaselineModel(0), BaselineModel(1)])

    s = str(model)

    assert(s == "MultiCue(Baseline(P=0), Baseline(P=1))")

# TODO: Add tests for _make_boundary_decision()

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

# TODO: Add tests for when precision_weights is True

def test_segment_empty_text():

    model = MultiCueModel(models=[BaselineModel()])

    segmented = [utt for utt in model.segment("")]

    assert(len(segmented) == 0)

def test_segmented_utterance_has_correct_number_of_boundaries():
    
    model = MultiCueModel(models=[BaselineModel(1), BaselineModel(0)])
    utterance = PhoneSequence("a b c d".split(' '))

    segmented = model.segment_utterance(utterance, update_model=False)

    assert(len(segmented.boundaries) == len(utterance.boundaries))

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b b c", "b a c b b"]
    corpus_phonestats = PhoneStats(max_ngram=2)
    lexicon = Lexicon()
    lexicon_phonestats = PhoneStats(max_ngram=2)
    models = [PredictabilityModel(ngram_length=1, measure="bp", phonestats=corpus_phonestats),
            PredictabilityModel(ngram_length=1, measure="sv", phonestats=corpus_phonestats)]
    model = MultiCueModel(models=models, corpus_phonestats=corpus_phonestats, lexicon=lexicon, lexicon_phonestats=lexicon_phonestats)

    list(model.segment(text, update_model=False))

    assert((model.weights == np.ones(model.num_models)).all())
    assert((model.errors == np.zeros(model.num_models)).all())
    assert(model.num_boundaries == 0)
    assert(len(lexicon) == 0)
    assert(corpus_phonestats.ntokens[1] == 0)
    assert(lexicon_phonestats.ntokens[1] == 0)

def test_segment_update_model_true_updates_model():

    text = ["a b b b c", "b a c b b"]
    corpus_phonestats = PhoneStats(max_ngram=2)
    lexicon = Lexicon()
    lexicon_phonestats = PhoneStats(max_ngram=2)
    models = [PredictabilityModel(ngram_length=1, measure="bp", phonestats=corpus_phonestats),
            PredictabilityModel(ngram_length=1, measure="sv", phonestats=corpus_phonestats)]
    model = MultiCueModel(models=models, corpus_phonestats=corpus_phonestats, lexicon=lexicon, lexicon_phonestats=lexicon_phonestats)

    list(model.segment(text, update_model=True))

    assert((model.weights != np.ones(model.num_models)).any())
    assert((model.errors != np.zeros(model.num_models)).any())
    assert(model.num_boundaries == 8)
    assert(len(lexicon) == 3)
    assert(corpus_phonestats.ntokens[1] == 14)
    assert(lexicon_phonestats.ntokens[1] == 16)
