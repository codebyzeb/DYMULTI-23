""" Run tests for the LexiconFrequencyIndicator class """

from segmenter.phonesequence import PhoneSequence
from segmenter.lexicon import Lexicon
from segmenter.partialpeakindicators import LexiconFrequencyIndicator

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_lexicon_sets_correct_properties():
    
    model = LexiconFrequencyIndicator(increase=False, right=True, use_presence=True, lexicon=None)

    assert(model.increase == False)
    assert(model.right == True)
    assert(model.use_presence == True)
    assert(model._updatelexicon == True)
    assert(model._lexicon == Lexicon())

def test_init_with_lexicon_sets_correct_properties():
    
    lexicon = Lexicon()
    lexicon.increase_count("word", 10)
    model = LexiconFrequencyIndicator(increase=False, right=True, use_presence=True, lexicon=lexicon)

    assert(model.increase == False)
    assert(model.right == True)
    assert(model.use_presence == True)
    assert(model._updatelexicon == False)
    assert(model._lexicon == lexicon)

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = LexiconFrequencyIndicator(increase=False, right=True, use_presence=True, lexicon=None)

    s = str(model)

    assert(s == "LexiconFreqIndicator(Decrease,Type Frequency,Right Context)")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = LexiconFrequencyIndicator()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b c", "b a c b"]
    model = LexiconFrequencyIndicator()

    list(model.segment(text, update_model=False))

    assert(model._lexicon["abbc"] == 0)
    assert(model._lexicon["bacb"] == 0)

def test_segment_update_model_true_updates_model():

    text = ["a b b c", "b a c b"]
    model = LexiconFrequencyIndicator()

    list(model.segment(text, update_model=True))

    assert(model._lexicon["abbc"] == 1)
    assert(model._lexicon["bacb"] == 1)

def test_segment_does_not_update_lexicon_when_provided():
    """ If provided a Lexicon object, the model should not update it """

    text = ["a b b c", "b a c b"]
    lexicon = Lexicon()
    model = LexiconFrequencyIndicator(lexicon=lexicon)

    list(model.segment(text, update_model=True))

    assert(model._lexicon["abbc"] == 0)
    assert(model._lexicon["bacb"] == 0)

def test_previously_seen_utterances_used_as_words_increase_true():
    """ Previously seen utterances are saved as words and later used to segment """

    text = ["c a r", "p a r k", "c a r p o o l"]
    model = LexiconFrequencyIndicator(increase=True, right=False)

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
    model = LexiconFrequencyIndicator(increase=False, right=False)

    segmentations = list(model.segment(text, update_model=True))

    assert(segmentations[0] == "car")
    assert(segmentations[1] == "park")
    assert(segmentations[2] ==  "car pool")
    assert(model._lexicon["car"] == 2)
    assert(model._lexicon["park"] == 1)
    assert(model._lexicon["pool"] == 1)
    assert(list(model._lexicon) == ["car", "park", "pool"])

def test_segmented_utterance_has_correct_number_of_boundaries():
    
    model = LexiconFrequencyIndicator(increase=False)
    utterance = PhoneSequence("a b c d".split(' '))

    segmented = model.segment_utterance(utterance, update_model=False)

    assert(len(segmented.boundaries) == len(utterance.boundaries))


"""
----------------------------------------------
            SCORE TESTS
----------------------------------------------
"""

def test_score_right_true():

    utterance = PhoneSequence("i t s a b a b y".split(" "))
    lexicon = Lexicon({"it" : 1, "i" : 1, "its" : 1, "a" : 1, "baby": 1, "by" : 1})
    model = LexiconFrequencyIndicator(lexicon=lexicon, right=True)

    assert(model.score(utterance, 0) == 3) # | i, it, its
    assert(model.score(utterance, 1) == 0)
    assert(model.score(utterance, 2) == 0)
    assert(model.score(utterance, 3) == 1) # | a
    assert(model.score(utterance, 4) == 1) # | baby
    assert(model.score(utterance, 5) == 1) # | a
    assert(model.score(utterance, 6) == 1) # | by
    assert(model.score(utterance, 7) == 0)
    assert(model.score(utterance, 8) == 0) 

def test_score_right_false():

    utterance = PhoneSequence("i t s a b a b y".split(" "))
    lexicon = Lexicon({"it" : 1, "i" : 1, "its" : 1, "a" : 1, "baby": 1, "by" : 1})
    model = LexiconFrequencyIndicator(lexicon=lexicon, right=False)

    assert(model.score(utterance, 0) == 0)
    assert(model.score(utterance, 1) == 1) # i |
    assert(model.score(utterance, 2) == 1) # it |
    assert(model.score(utterance, 3) == 1) # its |
    assert(model.score(utterance, 4) == 1) # a |
    assert(model.score(utterance, 5) == 0)
    assert(model.score(utterance, 6) == 1) # a |
    assert(model.score(utterance, 7) == 0)
    assert(model.score(utterance, 8) == 2) # baby, by |
