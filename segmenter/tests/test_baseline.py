""" Run tests for the BaselineModel class """

import random
import pytest

from segmenter.baseline import BaselineModel

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_probability_bigger_than_one_raises_value_error():
    with pytest.raises(ValueError, match=".*not a valid probability.*"):
        BaselineModel(1.2)

def test_probability_smaller_than_zero_raises_value_error():
    with pytest.raises(ValueError, match=".*not a valid probability.*"):
        BaselineModel(-0.1)

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = BaselineModel(0.5)

    s = str(model)

    assert(s == "Baseline(P=0.5)")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = BaselineModel()

    segmented = [utt for utt in model.segment("")]

    assert(len(segmented) == 0)

def test_zero_probability_segments_nothing():

    text = ["a b c d", "e f g h"]
    model = BaselineModel(0)

    segmented = [utt for utt in model.segment(text)]

    assert(len(segmented) == 2)
    assert(segmented[0] == "abcd")
    assert(segmented[1] == "efgh")

def test_one_probability_segments_all():

    text = ["a b c d", "e f g h"]
    model = BaselineModel(1)

    segmented = [utt for utt in model.segment(text)]

    assert(len(segmented) == 2)
    assert(segmented[0] == "a b c d")
    assert(segmented[1] == "e f g h")

def test_half_probability_segments_some():

    text = ["a b c d", "e f g h"]
    random.seed(0)
    model = BaselineModel(0.5)

    segmented = [utt for utt in model.segment(text)]

    assert(len(segmented) == 2)
    assert(segmented[0] == "abc d")
    assert(segmented[1] == "ef gh")
