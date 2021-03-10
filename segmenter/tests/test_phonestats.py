""" Run tests for the PhoneStats class """

import collections
import pytest

from segmenter.phonestats import PhoneStats

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_max_ng_less_than_one_raises_value_error():
    with pytest.raises(ValueError, match=".*not a valid ngram length.*"):
        PhoneStats(0)

def test_initialise_correct_ngram_counters_and_ntoken_counts():
    
    phonestats = PhoneStats(3)
    empty_counter = collections.Counter()

    assert(phonestats.max_ngram == 3)
    assert(list(phonestats.ngrams.keys()) == [1,2,3])
    for key in phonestats.ngrams:
        assert(phonestats.ngrams[key] == empty_counter)
    assert(list(phonestats.ntokens.keys()) == [1,2,3])
    for key in phonestats.ntokens:
        assert(phonestats.ntokens[key] == 0)

"""
----------------------------------------------
            UPDATING NGRAM COUNTS TESTS
----------------------------------------------
"""

def test_add_utterance_updates_ngram_counts():

    phonestats = PhoneStats(3)
    utterance = ['a', 'b', 'b', 'b', 'c']
    unigram_counter = collections.Counter({('a',) : 1, ('b',) : 3, ('c',) : 1})
    bigram_counter = collections.Counter({('a','b') : 1, ('b','b') : 2, ('b','c') : 1})
    trigram_counter = collections.Counter({('a', 'b', 'b') : 1, ('b','b','b') : 1, ('b', 'b', 'c') : 1})

    phonestats.add_utterance(utterance)

    assert(phonestats.ngrams[1] == unigram_counter)
    assert(phonestats.ngrams[2] == bigram_counter)
    assert(phonestats.ngrams[3] == trigram_counter)

def test_add_utterance_updates_ntokens_counts():

    phonestats = PhoneStats(3)
    utterance = ['a', 'b', 'b', 'b', 'c']

    phonestats.add_utterance(utterance)

    # The utterance contains five unigrams, four bigrams, three trigrams
    assert(phonestats.ntokens[1] == 5)
    assert(phonestats.ntokens[2] == 4)
    assert(phonestats.ntokens[3] == 3)

"""
----------------------------------------------
            PROBABILITY TESTS
----------------------------------------------
"""

def test_probability_ngram_too_large_raises_value_error():

    phonestats = PhoneStats(3)
    fourgram = ['a', 'b', 'b', 'c']

    with pytest.raises(ValueError, match=".*not stored in this PhoneStats object.*"):
        phonestats.probability(fourgram)

def test_probability_of_unseen_unigram():
    """ Test that the probabilty returns a valid value if the unigram hasn't been seen """
    phonestats = PhoneStats(1, smoothing=False)

    probability = phonestats.probability(['a'])

    assert(probability == 0)

def test_probability_of_unseen_unigram_smoothed():
    """ Test that the probabilty returns a valid value if the unigram hasn't been seen """
    phonestats = PhoneStats(1, smoothing=True)

    probability = phonestats.probability(['a'])

    # Using add1 smoothing, so (0+1)/(0+1) = 1
    assert(probability == 1)

def test_probability_of_seen_unigram():

    phonestats = PhoneStats(1, smoothing=False)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats.probability(['a'])

    assert(probability == 1/5)

def test_probability_of_seen_unigram_smoothed():

    phonestats = PhoneStats(1, smoothing=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats.probability(['a'])

    # Using add1 smoothing, so P('a') == 2 / 6
    assert(probability == 2/6)

def test_probability_of_bigram_ending_in_unigram():
    
    phonestats = PhoneStats(2, smoothing=False)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._probability_ngram_end(['b'], 2)

    # Out of the 4 bigrams, 3 end in 'b',  P(end in b) == 3/4
    assert(probability == 3/4)

def test_probability_of_bigram_ending_in_unigram_smoothed():
    
    phonestats = PhoneStats(2, smoothing=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._probability_ngram_end(['b'], 2)

    # Out of the 4 bigrams, 3 end in 'b', so add1 smoothing gives P(end in b) == 4/5
    assert(probability == 4/5)

def test_probability_of_bigram_starting_in_unigram():
    
    phonestats = PhoneStats(2, smoothing=False)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._probability_ngram_start(['a'], 2)

    # Out of the 4 bigrams, 1 starts in 'a', P(end in a) == 1/4
    assert(probability == 1/4)

def test_probability_of_bigram_starting_in_unigram_smoothed():
    
    phonestats = PhoneStats(2, smoothing=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._probability_ngram_start(['a'], 2)

    # Out of the 4 bigrams, 1 starts in 'a', so add1 smoothing gives P(end in a) == 2/5
    assert(probability == 2/5)

"""
----------------------------------------------
            CONDITIONAL PROBABILITY TESTS
----------------------------------------------
"""

def test_conditional_probability_ngram_too_large_raises_value_error():

    phonestats = PhoneStats(3)
    threegram = ['a', 'b', 'b']
    unigram = ['a']

    with pytest.raises(ValueError, match=".*not stored in this PhoneStats object.*"):
        phonestats._conditional_probability(threegram, unigram)

def test_reverse_conditional_probability_ngram_too_large_raises_value_error():

    phonestats = PhoneStats(3)
    threegram = ['a', 'b', 'b']
    unigram = ['a']

    with pytest.raises(ValueError, match=".*not stored in this PhoneStats object.*"):
        phonestats._conditional_probability_reverse(threegram, unigram)

def test_conditional_probability_fast_of_unigram():
    """ Test unsmoothed conditional probability using the lower-ngram assumption """

    phonestats = PhoneStats(2, smoothing=False)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._conditional_probability(['a'], ['b'])

    # P('b') == 3/5, P('ab') == 1/4 so P('a'|'b') is 5/12
    # Note that this method uses the assumption that P('b') is a good estimate for P('Xb'),
    # the probability that a bigram ends in 'b'.
    assert(probability == (1/4)/(3/5))

def test_conditional_probability_correct_of_unigram():
    """ Test unsmoothed conditional probability using the correct method """

    phonestats = PhoneStats(2, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._conditional_probability(['a'], ['b'])

    # P('xb') == 3/4, P('ab') == 1/4 so P('a'|'b') is 1/3
    # Note that this is the correct probability as it considers all bigrams that end
    # in 'b', rather than the probability that a unigram is 'b'.
    assert(probability == (1/4)/(3/4))

def test_reverse_conditional_probability_fast_of_unigram():
    """ Test unsmoothed reverse conditional probability using the lower-ngram assumption """

    phonestats = PhoneStats(2, smoothing=False)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._conditional_probability_reverse(['a'], ['b'])

    # P('a') == 1/5, P('ab') == 1/4 so P('a'|'b') is 5/4
    # Note that this method uses the assumption that P('a') is a good estimate for P('aX'),
    # the probability that a bigram starts in 'a'. As a result, we get a probability bigger than 1.
    assert(probability == (1/4)/(1/5))

def test_reverse_conditional_probability_correct_of_unigram():
    """ Test unsmoothed reverse conditional probability using the correct method """

    phonestats = PhoneStats(2, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    probability = phonestats._conditional_probability_reverse(['a'], ['b'])

    # P('ax') == 1/4, P('ab') == 1/4 so P('a'|'b') is 1
    # Note that this is the correct probability as it considers all bigrams that start
    # in 'a', rather than the probability that a unigram is 'a', and so we correctly get 1.
    assert(probability == 1)

"""
----------------------------------------------
            PREDICTABILITY TESTS
----------------------------------------------
"""

def test_boundary_entropy():

    phonestats = PhoneStats(3, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    entropy = phonestats._boundary_entropy(['a', 'b'])

    # The boundary entropy of "ab" is - SUM(P(x|ab)*log2(P(x|ab))) over all unigrams x
    # where P(x|ab) is the conditional probability of x FOLLOWING ab.
    # We have P(abx) == 1/3 (since out of three
    # trigrams, only one starts with 'ab'), so P(a|ab) == 0, P(b|ab) == 1, P(c|ab) == 0
    # using the correct conditional probability calculation.
    # This gives a boundary entropy of 0 (which makes sense, since from our utterance "ab" fully
    # predicts what comes next, so there is no uncertainty).

    assert(entropy == 0)

def test_boundary_entropy_smoothed():

    phonestats = PhoneStats(3, smoothing=True, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    entropy = phonestats._boundary_entropy(['a', 'b'])

    # The boundary entropy of "ab" is - SUM(P(x|ab)*log2(P(x|ab))) over all unigrams x
    # where P(x|ab) is the conditional probability of x FOLLOWING ab.
    # We have P(abx) == 2/4 using the smoothed probability calculation,
    # and P(aba) == 1/4, P(abb) = 2/4, P(abc) = 1/4
    # so P(a|ab) == 1/2, P(b|ab) == 1, P(c|ab) == 1/2 using the correct conditional probability.
    # This gives a boundary entropy of 1, much higher than the true value of 0 that we get
    # from the unsmoothed calculation.

    assert(entropy == 1)

def test_reverse_boundary_entropy():

    phonestats = PhoneStats(3, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    entropy = phonestats._boundary_entropy_reverse(['b', 'b'])

    # The reverse boundary entropy of "bb" is - SUM(P(x|bb)*log2(P(x|bb))) over all unigrams x
    # where P(x|bb) is the conditional probability of x PRECEDING ab.
    # We have P(xbb) == 2/3 (since out of three trigrams, two end with 'bb'),
    # so P(a|bb) == 1/2, P(b|bb) == 1/2, P(c|bb) == 0 using the correct conditional probability calculation.
    # This gives a boundary entropy of 1 (which makes sense, since from our utterance, 'bb' can be
    # preceded by either 'a' or 'b'.

    assert(entropy == 1)

def test_reverse_boundary_entropy_smoothed():

    phonestats = PhoneStats(3, smoothing=True, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    entropy = phonestats._boundary_entropy_reverse(['b', 'b'])

    # The reverse boundary entropy of "bb" is - SUM(P(x|bb)*log2(P(x|bb))) over all unigrams x
    # where P(x|bb) is the conditional probability of x PRECEDING ab.
    # We have P(xbb) == 3/5 using the smoothed probability calculation,
    # and P(abb) == 2/5, P(bbb) = 2/5, P(cbb) = 1/5
    # so P(a|bb) == 2/3, P(b|bb) == 2/3, P(c|bb) == 1/3 using the correct conditional probability.
    # This gives a boundary entropy of 1.3, higher than the true value of 1 that we get
    # from the unsmoothed calculation.

    assert(round(entropy, 2) == 1.31)

"""
----------------------------------------------
            UTTERANCE PREDICTABILITY TESTS
----------------------------------------------
"""

def test_get_boundary_entropy_in_utterance_uses_correct_ngram():
    """
    For utterance "abbbc" checks if finding the ngram entropy at position n-1
    returns the entropy of the correct ngram. For the bigram case, this is "ab" at position 1.
    """

    phonestats = PhoneStats(5, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    measure = "ent" # getting boundary entropy
    reverse = False # forward measure

    for ngram_length in range(1, 5):
        position = ngram_length - 1
        unpredictability = phonestats.get_unpredictability(utterance, position, measure, reverse, ngram_length)
        ngram_entropy = phonestats._boundary_entropy(utterance[:ngram_length])

        assert(unpredictability == ngram_entropy)

def test_get_boundary_entropy_at_bad_position_returns_none():
    """
    For utterance "abbbc" checks if finding the ngram entropy at position n-2
    returns None (since there is no ngram before that position).
    """

    phonestats = PhoneStats(5, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    measure = "ent" # getting boundary entropy
    reverse = False # forward measure

    for ngram_length in range(1, 5):
        position = ngram_length - 2
        unpredictability = phonestats.get_unpredictability(utterance, position, measure, reverse, ngram_length)

        assert(unpredictability == None)

def test_get_reverse_boundary_entropy_in_utterance_uses_correct_ngram():
    """
    For utterance "abbbc" checks if finding the reverse ngram entropy at position 5-n-1
    returns the entropy of the correct ngram. For the bigram case, this is "bc" at position 2.
    """

    phonestats = PhoneStats(5, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    measure = "ent" # getting boundary entropy
    reverse = True # forward measure

    for ngram_length in range(1, 5):
        position = 5 - ngram_length - 1
        unpredictability = phonestats.get_unpredictability(utterance, position, measure, reverse, ngram_length)
        ngram_entropy = phonestats._boundary_entropy_reverse(utterance[5-ngram_length:])

        assert(unpredictability == ngram_entropy)

def test_get_reverse_boundary_entropy_at_late_position_returns_none():
    """
    For utterance "abbbc" checks if finding the ngram entropy at position 5-n
    returns None (since there is no ngram after that position).
    """

    phonestats = PhoneStats(5, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    measure = "ent" # getting boundary entropy
    reverse = True # reverse measure

    for ngram_length in range(1, 5):
        position = 5 - ngram_length
        unpredictability = phonestats.get_unpredictability(utterance, position, measure, reverse, ngram_length)

        assert(unpredictability == None)

def test_get_transitional_probability_computes_conditional_entropy():
    """
    For utterance "abbbc" checks if finding the ngram transitional probability at position n-1
    returns the entropy of the correct ngram. For the bigram case, this is "ab" at position 1.
    """

    phonestats = PhoneStats(5, smoothing=False, correct_conditional=True)
    utterance = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_utterance(utterance)

    measure = "tp" # getting boundary entropy
    reverse = False # forward measure

    for ngram_length in range(1, 5):
        position = ngram_length - 1
        unpredictability = phonestats.get_unpredictability(utterance, position, measure, reverse, ngram_length)
        ngram_cond_prob = phonestats._conditional_probability_reverse(utterance[:ngram_length], [utterance[ngram_length]])

        assert(unpredictability == ngram_cond_prob)

