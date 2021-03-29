""" Run tests for the Probabilistic Model class """

import pytest

from numpy import log2

from segmenter.phonestats import PhoneStats
from segmenter.lexicon import Lexicon
from segmenter.probabilistic import ProbabilisticModel

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_without_lexicon_or_phonestats_sets_correct_properties():
    
    model = ProbabilisticModel(alpha=0, ngram_length=2, model_type="venk")

    assert(model.alpha == 0)
    assert(model.ngram_length == 2)
    assert(model._updatelexicon == True)
    assert(model.type == "venk")
    assert(model._lexicon == Lexicon())
    assert(model._updatephonestats == True)
    assert(not model._phonestats is None)
    assert(model._phonestats.max_ngram == 3)

def test_init_with_lexicon_and_phonestats_sets_correct_properties():
    
    lexicon = Lexicon()
    lexicon.increase_count("word", 10)
    phonestats = PhoneStats(3, use_boundary_tokens=True)

    model = ProbabilisticModel(alpha=0, ngram_length=2, model_type="venk", lexicon=lexicon, phonestats=phonestats)

    assert(model.alpha == 0)
    assert(model.ngram_length == 2)
    assert(model.type == "venk")
    assert(model._updatelexicon == False)
    assert(model._lexicon == lexicon)
    assert(model._updatephonestats == False)
    assert(model._phonestats == phonestats)

def test_init_with_invalid_model_type_raises_value_error():

    with pytest.raises(ValueError, match=".*with unknown model type.*"):
        ProbabilisticModel(model_type="blah")

"""
----------------------------------------------
            UTILITY TESTS
----------------------------------------------
"""

def test_to_string():

    model = ProbabilisticModel(alpha=0, ngram_length=2, model_type="venk")

    s = str(model)

    assert(s == "ProbabilisticModel(N: 2,alpha=0,type=venk)")

"""
----------------------------------------------
            WORD SCORE TESTS
----------------------------------------------
"""

def test_word_score_lm_unseen_words():

    phonestats = PhoneStats(max_ngram=2, smoothing=0, use_boundary_tokens=True)
    phones = ['a', 'a', 'b', 'c']
    alpha, ngram_length = 0.2, 1
    word = ['a', 'b']
    model = ProbabilisticModel(alpha=alpha, ngram_length=ngram_length, model_type="lm", phonestats=phonestats)

    phonestats.add_phones(phones)
    prob = model.word_score(word)

    # For unseen words, P(word) is given by phonestats and multiplied by alpha
    assert(prob == -log2(alpha) + phonestats.get_log_word_probability(word, ngram_length=ngram_length))

def test_word_score_lm_seen_words():

    lexicon = Lexicon({"ab" : 2, "bc" : 3})
    alpha, ngram_length = 0.2, 1
    word = ['a', 'b']
    model = ProbabilisticModel(alpha=alpha, ngram_length=ngram_length, model_type="lm", lexicon=lexicon)

    prob = model.word_score(word)

    # For seen words, P(word) is the negative log probability of the relative frequency
    # of the word in the lexicon, multiplied by (1-alpha)
    assert(prob == -log2(1 - alpha) -log2(lexicon.relative_freq(''.join(word))))

def test_word_score_venk_unseen_words():

    phonestats = PhoneStats(max_ngram=2, smoothing=0, use_boundary_tokens=True)
    lexicon = Lexicon({"dc" : 1, "ba" : 1, "ce" : 1})
    phones = ['d', 'c', 'b', 'a', 'c', 'e']
    ngram_length = 1
    word = ['a', 'b']
    model = ProbabilisticModel(ngram_length=ngram_length, model_type="venk", phonestats=phonestats, lexicon=lexicon)

    phonestats.add_phones(phones)
    prob = model.word_score(word)

    # For unseen words, P(word) is given by phonestats and multiplied by the estimate unseen word probability and the boundary token probability
    bound_prob = 3/8 # seen three words out of 8 phonemes
    unseen_prob = lexicon.unseen_freq()
    assert(prob == -log2(bound_prob / (1 - bound_prob)) -log2(unseen_prob) + phonestats.get_log_word_probability(word, ngram_length=ngram_length))

def test_word_score_venk_seen_words():

    lexicon = Lexicon({"ab" : 2, "bc" : 3})
    ngram_length = 1
    word = ['a', 'b']
    model = ProbabilisticModel(ngram_length=ngram_length, model_type="venk", lexicon=lexicon)

    prob = model.word_score(word)

    # For seen words, P(word) is the negative log probability of the relative frequency
    # of the word in the lexicon, smoothing for unseen words
    assert(prob == -log2(lexicon.relative_freq(''.join(word), consider_unseen=True)))

"""
----------------------------------------------
            SEGMENTATION TESTS
----------------------------------------------
"""

def test_segment_empty_text():

    model = ProbabilisticModel()

    segmented = list(model.segment(""))

    assert(len(segmented) == 0)

def test_segment_update_model_false_does_not_update_model():

    text = ["a b b c", "b a c b"]
    model = ProbabilisticModel()

    list(model.segment(text, update_model=False))

    assert(model._lexicon["abbc"] == 0)
    assert(model._lexicon["bacb"] == 0)
    assert(model._phonestats.ntokens[1] == 0)

def test_segment_update_model_true_updates_model():

    text = ["a b c d", "e f g h"]
    model = ProbabilisticModel()

    list(model.segment(text, update_model=True))

    assert(model._lexicon["abcd"] == 1)
    assert(model._lexicon["efgh"] == 1)
    assert(model._phonestats.ntokens[1] == 12)

def test_segment_does_not_update_lexicon_or_phonestats_when_provided():
    """ If provided a Lexicon object and a Phonestats object, the model should not update it """

    text = ["a b b c", "b a c b"]
    lexicon = Lexicon()
    phonestats = PhoneStats(max_ngram=2, use_boundary_tokens=True)
    model = ProbabilisticModel(phonestats=phonestats, lexicon=lexicon)

    list(model.segment(text, update_model=True))

    assert(len(model._lexicon) == 0)
    assert(model._phonestats.ntokens[1] == 0)

def test_segment_utterance_with_seen_word_segments_at_seen_word():

    lexicon = Lexicon({"abc" : 1, "ca" : 1})
    phonestats = PhoneStats(max_ngram=1, use_boundary_tokens=True)
    phonestats.add_phones(['a', 'b', 'c', 'd', 'e'])
    text = ['a b c b', 'c a b a']
    model = ProbabilisticModel(alpha=0.5, ngram_length=0, phonestats=phonestats, lexicon=lexicon)

    segmented = list(model.segment(text))

    assert(segmented[0] == "abc b")
    assert(segmented[1] == "ca ba")

def test_segment_utterance_without_seen_word_segments_entire_utterance():

    lexicon = Lexicon({"abc" : 1, "cba" : 1})
    phonestats = PhoneStats(max_ngram=1, use_boundary_tokens=True)
    phonestats.add_phones(['a', 'b', 'c', 'd', 'e'])
    text = ['b c a d', 'c a b a']
    model = ProbabilisticModel(alpha=0.5, ngram_length=0, phonestats=phonestats, lexicon=lexicon)

    segmented = list(model.segment(text))

    assert(segmented[0] == "bcad")
    assert(segmented[1] == "caba")

