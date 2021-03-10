""" Run tests for the Model class """

from segmenter.model import Model

class SimpleModel(Model):

    """ To test segment() method from Model class, need to implement segment_utterance() in a subclass """

    def segment_utterance(self, utterance, update_model=True):
        return ''.join(utterance.strip().split(' '))

def test_model_segments_each_utterance():

    text = ["a b c d", "e f g h"]
    model = SimpleModel()

    segmented = list(model.segment(text))

    assert(len(segmented) == 2)
    assert(segmented[0] == "abcd")
    assert(segmented[1] == "efgh")

def test_model_to_string():
    
    model = SimpleModel()
    assert(str(model) == "Abstract")