""" Run tests for the Lexicon Boundary Model class """

from segmenter.phonesequence import PhoneSequence
from segmenter.phonestats import PhoneStats
from segmenter.lexicon import Lexicon
from segmenter.lexicon import LexiconBoundaryModel

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_lexicon_or_phonestats_sets_correct_properties():
    
    model = LexiconBoundaryModel(ngram_length=3, increase=False, right=True, lexicon=None, phonestats=None)

    assert(model.increase == False)
    assert(model.right == True)
    assert(model.ngram_length == 3)
    assert(model._updatelexicon == True)
    assert(model._lexicon == Lexicon())
    assert(model._updatephonestats == True)
    assert(not model._phonestats is None)
    assert(model._phonestats.max_ngram == 4)

def test_init_with_lexicon_and_phonestats_sets_correct_properties():
    
    lexicon = Lexicon()
    lexicon.increase_count("word", 10)
    phonestats = PhoneStats(3, use_boundary_tokens=True)

    model = LexiconBoundaryModel(ngram_length=3, increase=False, right=True, lexicon=lexicon, phonestats=phonestats)

    assert(model.increase == False)
    assert(model.right == True)
    assert(model.ngram_length == 3)
    assert(model._updatelexicon == False)
    assert(model._lexicon == lexicon)
    assert(model._updatephonestats == False)
    assert(model._phonestats == phonestats)

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = LexiconBoundaryModel(ngram_length=3, increase=False, right=True, lexicon=None, phonestats=None)

    s = str(model)

    assert(s == "LexiconBoundaryModel(N: 3,Decrease,Right Context)")

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = LexiconBoundaryModel()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b c", "b a c b"]
    model = LexiconBoundaryModel()

    list(model.segment(text, update_model=False))

    assert(model._lexicon["abbc"] == 0)
    assert(model._lexicon["bacb"] == 0)
    assert(model._phonestats.ntokens[1] == 0)

def test_segment_update_model_true_updates_model():

    text = ["a b c d", "e f g h"]
    model = LexiconBoundaryModel()

    list(model.segment(text, update_model=True))

    assert(model._lexicon["abcd"] == 1)
    assert(model._lexicon["efgh"] == 1)
    assert(model._phonestats.ntokens[1] == 12)

def test_segment_does_not_update_lexicon_or_phonestats_when_provided():
    """ If provided a Lexicon object and a Phonestats object, the model should not update it """

    text = ["a b b c", "b a c b"]
    lexicon = Lexicon()
    phonestats = PhoneStats(max_ngram=2, use_boundary_tokens=True)
    model = LexiconBoundaryModel(phonestats=phonestats, lexicon=lexicon)

    list(model.segment(text, update_model=True))

    assert(len(model._lexicon) == 0)
    assert(model._phonestats.ntokens[1] == 0)

def test_previously_seen_utterances_used_as_words():
    """ Previously seen utterances are saved as words and later used to segment """

    text = ["c a r", "p o o l", "c a r p o o l"]

    for increase in [True, False]:
        for right in [True, False]:

            model = LexiconBoundaryModel(increase=increase, right=right)

            segmentations = list(model.segment(text, update_model=True))

            assert(segmentations[0] == "car")
            assert(segmentations[1] == "pool")
            assert(segmentations[2] ==  "car pool")
            assert(model._lexicon["car"] == 2)
            assert(model._lexicon["pool"] == 2)
            assert(list(model._lexicon) == ["car", "pool"])
            assert(model._phonestats.ntokens[1] == 22)

def test_segmented_utterance_has_correct_number_of_boundaries():
    
    model = LexiconBoundaryModel(increase=True, right=False)
    utterance = PhoneSequence("a b c d".split(' '))

    segmented = model.segment_utterance(utterance, update_model=False)

    assert(len(segmented.boundaries) == len(utterance.boundaries))


"""
----------------------------------------------
            SCORE TESTS
----------------------------------------------
"""

def test_score_right_context():

    utterance = PhoneSequence("c a r p o o l".split(' '))
    phonestats = PhoneStats(2, use_boundary_tokens=True)
    phonestats.add_phones(["c", "a", "r"])
    phonestats.add_phones(["p", "a", "r", "k"])
    lexicon = Lexicon({"car" : 1, "park" : 1})

    model = LexiconBoundaryModel(phonestats=phonestats, lexicon=lexicon, right=True)

    # Right context used for scoring
    assert(model.score(utterance, -1) == 1) # P(boundary | 'c' == 1)
    assert(model.score(utterance, 0) == 0) # P(boundary | 'a' == 0)
    assert(model.score(utterance, 1) == 0) # P(boundary | 'r' == 0)
    assert(model.score(utterance, 2) == 1) # P(boundary | 'p' == 1)
    assert(model.score(utterance, 3) == 0) # P(boundary | 'o' == 0)
    assert(model.score(utterance, 4) == 0) # P(boundary | 'o' == 0)
    assert(model.score(utterance, 5) == 0) # P(boundary | 'l' == 0)
    assert(model.score(utterance, 6) == 0) # P(boundary | boundary == 0)

def test_score_left_context():

    utterance = PhoneSequence("c a r p o o l".split(' '))
    phonestats = PhoneStats(2, use_boundary_tokens=True)
    phonestats.add_phones(["c", "a", "r"])
    phonestats.add_phones(["p", "a", "r", "k"])
    lexicon = Lexicon({"car" : 1, "park" : 1})

    model = LexiconBoundaryModel(phonestats=phonestats, lexicon=lexicon, right=False)

    # Left context used for scoring
    assert(model.score(utterance, -1) == 0) # P(boundary | boundary == 0) -> problem?
    assert(model.score(utterance, 0) == 0) # P(boundary | 'c' == 0)
    assert(model.score(utterance, 1) == 0) # P(boundary | 'a' == 0)
    assert(model.score(utterance, 2) == 0.5) # P(boundary | 'r' == 0.5)
    assert(model.score(utterance, 3) == 0) # P(boundary | 'p' == 0)
    assert(model.score(utterance, 4) == 0) # P(boundary | 'o' == 0)
    assert(model.score(utterance, 5) == 0) # P(boundary | 'o' == 0)
    assert(model.score(utterance, 6) == 0) # P(boundary | 'l' == 0)
