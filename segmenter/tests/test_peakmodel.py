""" Run tests for the PeakModel abstract class """

import pytest

from segmenter.peakmodel import PeakModel

# Create subclass with implementation of score() for testing
class ExamplePeakModel(PeakModel):

    def score(self, utterance, position):
        if position < 0:
            return None
        if utterance[position] == "a":
            return 1
        if utterance[position] == "b":
            return 2
        return 0

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_sets_correct_properties():

    model = ExamplePeakModel(increase=True)

    assert(model.increase == True)

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = ExamplePeakModel(increase=False)

    s = str(model)

    assert(s == "PeakModel(Decrease)")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = ExamplePeakModel()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_at_increase():

    # scores are Xa1b2b2c0a1c0, so the increase model will segment between "bb" and between "ac"
    utt = ["a b b c a c"]
    model = ExamplePeakModel(increase=True)

    segmentation = list(model.segment(utt))[0]

    assert(segmentation == "ab bca c")

def test_segment_at_decrease():

    # scores are Xa1b2b2c0a1c0, so the decrease model will segment between "bc" and "ac"
    utt = ["a b b c a c"]
    model = ExamplePeakModel(increase=False)

    segmentation = list(model.segment(utt))[0]

    assert(segmentation == "abb ca c")
