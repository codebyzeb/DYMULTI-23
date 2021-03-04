""" Run tests for the MultieCueModel class """

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

# TODO: Add tests for _make_boundary_decision(), _segmented_utterance_to_boundaries and _boundaries_to_segmented_utterance

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
    text = "a b c d"
    update = True

    model.segment_utterance(text, update_model=update)

    assert((model.weights != np.ones(model.num_models)).any())
    assert((model.errors != np.zeros(model.num_models)).any())
    assert(model.num_boundaries == 3)

def test_segment_update_model_false_does_not_update_model():
    """
    Ensure that when update_model is False, that the weights and errors do not update
    """

    model = MultiCueModel(models=[BaselineModel(1), BaselineModel(0)])
    text = "a b c d"
    update = False

    model.segment_utterance(text, update_model=update)

    assert((model.weights == np.ones(model.num_models)).all())
    assert((model.errors == np.zeros(model.num_models)).all())
    assert(model.num_boundaries == 0)
