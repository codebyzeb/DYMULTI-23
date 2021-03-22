""" Run tests for the MultieCueModel class """

from segmenter.phonesequence import PhoneSequence
import pytest
import numpy as np

from segmenter.baseline import BaselineModel
from segmenter.multicue import MultiCueModel

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_model_uses_baseline():

    model = MultiCueModel()

    assert(model.num_models == 1)
    assert(isinstance(model.models[0], BaselineModel))

def test_init_with_not_a_model_raises_value_error():

    with pytest.raises(ValueError, match=".*not an instance.*"):
        MultiCueModel(models=[5])

def test_init_assigns_correct_initial_values():

    model = MultiCueModel(models=[BaselineModel(), BaselineModel()])

    assert(model.num_models == 2)
    assert((model.weights == np.ones(model.num_models)).all())
    assert((model.errors == np.zeros(model.num_models)).all())
    assert(model.num_boundaries == 0)

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

def test_segment_empty_text():

    model = MultiCueModel(models=[BaselineModel()])

    segmented = [utt for utt in model.segment("")]

    assert(len(segmented) == 0)

def test_segment_update_model_true_updates_model():
    """
    Ensure that when update_model is True, that the weights, errors and number of boundaries seen update.
    """
    model = MultiCueModel(models=[BaselineModel(1), BaselineModel(0)])
    utterance = PhoneSequence("a b c d".split(' '))

    model.segment_utterance(utterance, update_model=True)

    assert((model.weights != np.ones(model.num_models)).any())
    assert((model.errors != np.zeros(model.num_models)).any())
    assert(model.num_boundaries == 3)

def test_segment_update_model_false_does_not_update_model():
    """
    Ensure that when update_model is False, that the weights and errors do not update
    """

    model = MultiCueModel(models=[BaselineModel(1), BaselineModel(0)])
    utterance = PhoneSequence("a b c d".split(' '))

    model.segment_utterance(utterance, update_model=False)

    assert((model.weights == np.ones(model.num_models)).all())
    assert((model.errors == np.zeros(model.num_models)).all())
    assert(model.num_boundaries == 0)

def test_segmented_utterance_has_correct_number_of_boundaries():
    
    model = MultiCueModel(models=[BaselineModel(1), BaselineModel(0)])
    utterance = PhoneSequence("a b c d".split(' '))

    segmented = model.segment_utterance(utterance, update_model=False)

    assert(len(segmented.boundaries) == len(utterance.boundaries))

