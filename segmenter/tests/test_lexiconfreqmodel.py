""" Run tests for the LexiconFrequencyModel class """

from segmenter.phonesequence import PhoneSequence
from segmenter.lexicon import Lexicon
from segmenter.lexicon import LexiconFrequencyModel

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_lexicon_sets_correct_properties():
    
    model = LexiconFrequencyModel(increase=False, use_presence=True, lexicon=None)

    assert(model.increase == False)
    assert(model.use_presence == True)
    assert(model._updatelexicon == True)
    assert(model._lexicon == Lexicon())

def test_init_with_lexicon_sets_correct_properties():
    
    lexicon = Lexicon()
    lexicon.increase_count("word", 10)
    model = LexiconFrequencyModel(increase=False, use_presence=True, lexicon=lexicon)

    assert(model.increase == False)
    assert(model.use_presence == True)
    assert(model._updatelexicon == False)
    assert(model._lexicon == lexicon)

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = LexiconFrequencyModel(increase=False, use_presence=True, lexicon=None)

    s = str(model)

    assert(s == "LexiconFreqModel(Decrease,Type Frequency)")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = LexiconFrequencyModel()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b c", "b a c b"]
    model = LexiconFrequencyModel()

    list(model.segment(text, update_model=False))

    assert(model._lexicon["abbc"] == 0)
    assert(model._lexicon["bacb"] == 0)

def test_segment_update_model_true_updates_model():

    text = ["a b b c", "b a c b"]
    model = LexiconFrequencyModel()

    list(model.segment(text, update_model=True))

    assert(model._lexicon["abbc"] == 1)
    assert(model._lexicon["bacb"] == 1)

def test_segment_does_not_update_lexicon_when_provided():
    """ If provided a Lexicon object, the model should not update it """

    text = ["a b b c", "b a c b"]
    lexicon = Lexicon()
    model = LexiconFrequencyModel(lexicon=lexicon)

    list(model.segment(text, update_model=True))

    assert(model._lexicon["abbc"] == 0)
    assert(model._lexicon["bacb"] == 0)

def test_previously_seen_utterances_used_as_words_increase_true():
    """ Previously seen utterances are saved as words and later used to segment """

    text = ["c a r", "p a r k", "c a r p o o l"]
    model = LexiconFrequencyModel(increase=True)

    segmentations = list(model.segment(text, update_model=True))

    assert(segmentations[0] == "car")
    assert(segmentations[1] == "park")
    assert(segmentations[2] ==  "car pool")
    assert(model._lexicon["car"] == 2)
    assert(model._lexicon["park"] == 1)
    assert(model._lexicon["pool"] == 1)
    assert(list(model._lexicon) == ["car", "park", "pool"])

def test_previously_seen_utterances_used_as_words_increase_false():
    """ Previously seen utterances are saved as words and later used to segment """

    text = ["c a r", "p a r k", "c a r p o o l"]
    model = LexiconFrequencyModel(increase=False)

    segmentations = list(model.segment(text, update_model=True))

    assert(segmentations[0] == "car")
    assert(segmentations[1] == "park")
    assert(segmentations[2] ==  "car pool")
    assert(model._lexicon["car"] == 2)
    assert(model._lexicon["park"] == 1)
    assert(model._lexicon["pool"] == 1)
    assert(list(model._lexicon) == ["car", "park", "pool"])

def test_segmented_utterance_has_correct_number_of_boundaries():
    
    model = LexiconFrequencyModel(increase=False)
    utterance = PhoneSequence("a b c d".split(' '))

    segmented = model.segment_utterance(utterance, update_model=False)

    assert(len(segmented.boundaries) == len(utterance.boundaries))


"""
----------------------------------------------
            SCORE TESTS
----------------------------------------------
"""

def test_score():

    utterance = PhoneSequence("i t s a b a b y".split(" "))
    lexicon = Lexicon({"it" : 1, "i" : 1, "its" : 1, "a" : 1, "baby": 1, "by" : 1})
    model = LexiconFrequencyModel(lexicon=lexicon)

    assert(model.score(utterance, -1) == 3) # | i, it, its
    assert(model.score(utterance, 0) == 1) # i |
    assert(model.score(utterance, 1) == 1) # it |
    assert(model.score(utterance, 2) == 2) # its | a
    assert(model.score(utterance, 3) == 2) # a | baby
    assert(model.score(utterance, 4) == 1) # | a
    assert(model.score(utterance, 5) == 2) # a | by
    assert(model.score(utterance, 6) == 0)
    assert(model.score(utterance, 7) == 2) # baby, by |
