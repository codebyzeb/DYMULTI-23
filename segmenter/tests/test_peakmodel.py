""" Run tests for the PeakModel abstract class """

from segmenter.model import PeakModel
from segmenter.phonesequence import PhoneSequence

# Create subclass with implementation of score() for testing
class ExamplePeakModel(PeakModel):

    def __init__(self, increase=True):
        self.a = 1
        super().__init__(increase)

    def score(self, utterance, position):
        if position == len(utterance):
            return None
        if utterance.phones[position] == "a":
            return 1
        if utterance.phones[position] == "b":
            return 2
        return 0

    def update(self, segmented):
        self.a = 2

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

    # scores are 1a2b2b0c1a0cX, so the increase model will segment between "ab" and between "ca"
    utt = ["a b b c a c"]
    model = ExamplePeakModel(increase=True)

    segmentation = list(model.segment(utt))[0]

    assert(segmentation == "a bbc ac")

def test_segment_at_decrease():

    # scores are 1a2b2b0c1a0cX, so the decrease model will segment between "bb" and "acac"
    utt = ["a b b c a c"]
    model = ExamplePeakModel(increase=False)

    segmentation = list(model.segment(utt))[0]

    assert(segmentation == "ab bc ac")

def test_segmented_utterance_has_correct_number_of_boundaries():
    
    model = ExamplePeakModel()
    utterance = PhoneSequence("a b c d".split(' '))

    segmented = model.segment_utterance(utterance, update_model=False)

    assert(len(segmented.boundaries) == len(utterance.boundaries))

"""
----------------------------------------------
            UPDATE TESTS
----------------------------------------------
"""

def test_update_true_update_model():

    utt = ["a b b c a c"]
    model = ExamplePeakModel()

    list(model.segment(utt, update_model=True))

    assert(model.a == 2)

def test_update_false_does_not_update_model():

    utt = ["a b b c a c"]
    model = ExamplePeakModel()

    list(model.segment(utt, update_model=False))

    assert(model.a == 1)

    