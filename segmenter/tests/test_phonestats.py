""" Run tests for the PhoneStats class """

import collections
from segmenter.phonesequence import PhoneSequence
import pytest
import numpy as np

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
    assert(phonestats.ngrams == empty_counter)
    assert(list(phonestats.ntokens.keys()) == [1,2,3])
    for key in phonestats.ntokens:
        assert(phonestats.ntokens[key] == 0)

"""
----------------------------------------------
            UPDATING NGRAM COUNTS TESTS
----------------------------------------------
"""

def test_add_utterance_updates_ngram_counts():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    ngram_counter = collections.Counter({'a' : 1, 'b' : 3, 'c' : 1,
                                        'ab' : 1, 'bb' : 2, 'bc' : 1,
                                        'abb' : 1, 'bbb' : 1, 'bbc' : 1})

    phonestats.add_phones(phones)

    assert(phonestats.ngrams == ngram_counter)

def test_add_phones_updates_ntokens_counts():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']

    phonestats.add_phones(phones)

    # The phones contains five unigrams, four bigrams, three trigrams
    assert(phonestats.ntokens[1] == 5)
    assert(phonestats.ntokens[2] == 4)
    assert(phonestats.ntokens[3] == 3)

def test_add_phones_updates_ntokens_counts_using_boundary_tokens():

    phonestats = PhoneStats(3, use_boundary_tokens=True)
    phones = ['a', 'b', 'b', 'b', 'c']

    phonestats.add_phones(phones)

    # The phones contains 9 unigrams, eight bigrams, seven trigrams
    assert(phonestats.ntokens[1] == 9)
    assert(phonestats.ntokens[2] == 8)
    assert(phonestats.ntokens[3] == 7)

def test_get_types():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']

    phonestats.add_phones(phones)

    assert(phonestats._types(1) == [['a'], ['b'], ['c']])
    assert(phonestats._types(2) == [['a', 'b'], ['b', 'b'], ['b', 'c']])
    assert(phonestats._types(3) == [['a', 'b', 'b'], ['b', 'b', 'b'], ['b', 'b', 'c']])

def test_add_invalid_phoness():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    phones = []

    phonestats.add_phones(phones)

    assert(phonestats.ntokens[1] == 0)
    assert(phonestats.ntokens[2] == 0)
    assert(phonestats.ntokens[3] == 0)

"""
----------------------------------------------
            PROBABILITY TESTS
----------------------------------------------
"""

def test_probability_ngram_too_large_raises_value_error():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    fourgram = ['a', 'b', 'b', 'c']

    with pytest.raises(ValueError, match=".*not stored in this PhoneStats object.*"):
        phonestats.probability(fourgram)

def test_probability_of_unseen_ngram():
    """ Test that the probabilty returns a valid value if the ngram hasn't been seen """
    phonestats = PhoneStats(2, smoothing=0, use_boundary_tokens=False)

    probability_uni = phonestats.probability(['a'])
    probability_bi = phonestats.probability(['a', 'b'])

    assert(probability_uni == 0)
    assert(probability_bi == 0)

def test_probability_of_unseen_ngram_smoothed():
    """ Test that the probabilty returns a valid value if the ngram hasn't been seen """
    phonestats = PhoneStats(2, smoothing=1, use_boundary_tokens=False)

    probability_uni = phonestats.probability(['a'])
    probability_bi = phonestats.probability(['a', 'b'])

    # Using add1 smoothing, so (0+1)/(0+1) = 1
    assert(probability_uni == 1)
    assert(probability_bi == 1)

def test_probability_of_seen_unigram():

    phonestats = PhoneStats(1, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    probability = phonestats.probability(['a'])

    assert(probability == 1/5)

def test_probability_of_seen_unigram_smoothed():

    phonestats = PhoneStats(1, smoothing=1, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    probability = phonestats.probability(['a'])

    # Using add1 smoothing, so P('a') == 2 / 6
    assert(probability == 2/6)

def test_probability_of_seen_bigram():

    phonestats = PhoneStats(2, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    probability = phonestats.probability(['a', 'b'])

    assert(probability == 1/4)

def test_probability_of_seen_bigram_smoothed():

    phonestats = PhoneStats(2, smoothing=1, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    probability = phonestats.probability(['a', 'b'])

    # Using add1 smoothing, so P('ab') == 2 / 5
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

def test_conditional_probability_of_ngram_fast():
    """ Test unsmoothed conditional probability """

    phonestats = PhoneStats(3, smoothing=0)
    phonestats.add_phones(['a', 'b', 'b', 'b'])
    phonestats.add_phones(['b', 'b', 'c', 'a'])

    probability_uni = phonestats._conditional_probability(['a'], ['b'])
    probability_bi = phonestats._conditional_probability(['a'], ['b', 'b'])

    # freq('b') == 5, freq('ab') == 1 so P('a'|'b') is 1/5
    assert(probability_uni == (1/5))
    # freq('bb') == 3, freq('abb') == 1 so P('a'|'bb') is 1/3
    assert(probability_bi == (1/3))

def test_conditional_probability_of_ngram_smoothed():
    """ Test unsmoothed conditional probability """

    phonestats = PhoneStats(3, smoothing=1)
    phonestats.add_phones(['a', 'b', 'b', 'b'])
    phonestats.add_phones(['b', 'b', 'c', 'a'])

    probability_uni = phonestats._conditional_probability(['a'], ['b'])
    probability_bi = phonestats._conditional_probability(['a'], ['b', 'b'])

    # freq('b') == 5, freq('ab') == 1 so P('a'|'b') is (1+1)/(5+1)
    assert(probability_uni == (2/6))
    # freq('bb') == 3, freq('abb') == 1 so P('a'|'bb') is (1+1)/(3+1)
    assert(probability_bi == (2/4))

def test_reverse_conditional_probability_of_ngram_unsmoothed():
    """ Test unsmoothed reverse conditional probability """

    phonestats = PhoneStats(3, smoothing=0)
    phonestats.add_phones(['a', 'b', 'b', 'b'])
    phonestats.add_phones(['b', 'b', 'c', 'a'])

    probability_uni = phonestats._conditional_probability_reverse(['a'], ['b'])
    probability_bi = phonestats._conditional_probability_reverse(['a', 'b'], ['b'])

    # freq('a') == 2, freq('ab') == 1 so P('b'|'a') is 1/2
    assert(probability_uni == 1/2)
    # freq('ab') == 1, freq('abb') == 1 so P('b'|'ab') is 1
    assert(probability_bi == 1)

def test_reverse_conditional_probability_of_ngram_smoothed():
    """ Test smoothed reverse conditional probability """

    phonestats = PhoneStats(3, smoothing=1)
    phonestats.add_phones(['a', 'b', 'b', 'b'])
    phonestats.add_phones(['b', 'b', 'c', 'a'])

    probability_uni = phonestats._conditional_probability_reverse(['a'], ['b'])
    probability_bi = phonestats._conditional_probability_reverse(['a', 'b'], ['b'])

    # freq('a') == 2, freq('ab') == 1 so P('b'|'a') is (1+1)/(2+1)
    assert(probability_uni == 2/3)
    # freq('ab') == 1, freq('abb') == 1 so P('b'|'ab') is (1+1)/(1+1)
    assert(probability_bi == 1)

"""
----------------------------------------------
            ENTROPY TESTS
----------------------------------------------
"""

def test_boundary_entropy():

    phonestats = PhoneStats(3, smoothing=0, use_boundary_tokens=False)
    phonestats.add_phones(['a', 'b', 'b', 'b', 'c'])

    entropy = phonestats._boundary_entropy(['a', 'b'])

    # The boundary entropy of "ab" is - SUM(P(x|ab)*log2(P(x|ab))) over all unigrams x
    # where P(x|ab) is the conditional probability of x FOLLOWING ab.
    # We have freq(ab) == 1, so P(a|ab) == 0, P(b|ab) == 1, P(c|ab) == 0
    # using the correct conditional probability calculation.
    # This gives a boundary entropy of 0 (which makes sense, since from our phones "ab" fully
    # predicts what comes next, so there is no uncertainty).

    assert(entropy == 0)

def test_boundary_entropy_smoothed():

    phonestats = PhoneStats(3, smoothing=1, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    entropy = phonestats._boundary_entropy(['a', 'b'])

    # The boundary entropy of "ab" is - SUM(P(x|ab)*log2(P(x|ab))) over all unigrams x
    # where P(x|ab) is the conditional probability of x FOLLOWING ab.
    # We have freq(ab) == 1
    # so P(a|ab) == 1/2, P(b|ab) == 1, P(c|ab) == 1/2 using the smoothed conditional probability.
    # This gives a boundary entropy of 1, much higher than the true value of 0 that we get
    # from the unsmoothed calculation.

    assert(entropy == 1)

def test_reverse_boundary_entropy():

    phonestats = PhoneStats(3, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    entropy = phonestats._boundary_entropy_reverse(['b', 'b'])

    # The reverse boundary entropy of "bb" is - SUM(P(x|bb)*log2(P(x|bb))) over all unigrams x
    # where P(x|bb) is the conditional probability of x PRECEDING ab.
    # We have freq(bb) == 2
    # so P(a|bb) == 1/2, P(b|bb) == 1/2, P(c|bb) == 0 using the conditional probability calculation.
    # This gives a boundary entropy of 1 (which makes sense, since from our phones, 'bb' can be
    # preceded by either 'a' or 'b'.

    assert(entropy == 1)

def test_reverse_boundary_entropy_smoothed():

    phonestats = PhoneStats(3, smoothing=1, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    entropy = phonestats._boundary_entropy_reverse(['b', 'b'])

    # The reverse boundary entropy of "bb" is - SUM(P(x|bb)*log2(P(x|bb))) over all unigrams x
    # where P(x|bb) is the conditional probability of x PRECEDING ab.
    # We have freq(bb) == 2 
    # so P(a|bb) == 2/3, P(b|bb) == 2/3, P(c|bb) == 1/3 using the smoothed conditional probability.
    # This gives a boundary entropy of 1.3, higher than the true value of 1 that we get
    # from the unsmoothed calculation.

    assert(round(entropy, 2) == 1.31)

"""
----------------------------------------------
            SUCCESSOR VARIETY TESTS
----------------------------------------------
"""

def test_successor_variety_ngram_too_large_raises_value_error():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    threegram = ['a', 'b', 'b']

    with pytest.raises(ValueError, match=".*not stored in this PhoneStats object.*"):
        phonestats._successor_variety(threegram)

def test_successor_variety():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    phonestats.add_phones(['a', 'b', 'b', 'c'])
    phonestats.add_phones(['b', 'a', 'b'])

    sv_a = phonestats._successor_variety(['a'])
    sv_b = phonestats._successor_variety(['b'])
    sv_c = phonestats._successor_variety(['c'])
    sv_ab = phonestats._successor_variety(['a', 'b'])

    assert(sv_a == 1)
    assert(sv_b == 3)
    assert(sv_c == 0)
    assert(sv_ab == 1)

def test_reverse_successor_variety():

    phonestats = PhoneStats(3, use_boundary_tokens=False)
    phonestats.add_phones(['a', 'b', 'b', 'c'])
    phonestats.add_phones(['b', 'a', 'b'])

    sv_a = phonestats._successor_variety_reverse(['a'])
    sv_b = phonestats._successor_variety_reverse(['b'])
    sv_c = phonestats._successor_variety_reverse(['c'])
    sv_ab = phonestats._successor_variety_reverse(['a', 'b'])

    assert(sv_a == 1)
    assert(sv_b == 2)
    assert(sv_c == 1)
    assert(sv_ab == 1)

"""
----------------------------------------------
            MUTUAL INFORMATION TESTS
----------------------------------------------
"""

def test_mutual_information():

    phonestats = PhoneStats(3, smoothing=0, use_boundary_tokens=False)
    phonestats.add_phones(['a', 'b', 'b', 'b', 'c'])

    mutual_information_uni = phonestats._mutual_information(['a'], ['b'])
    mutual_information_bi = phonestats._mutual_information(['a', 'b'], ['b'])

    # The mutual information of "ab" is log2(P(ab)/(P(a)P(b)))
    # We have P(ab) == 1/4, P(a) == 1/5, P(b) == 3/5
    # so mutual information is log2((1/4)/(3/25)
    assert(mutual_information_uni == np.log2((1/4)/(3/25)))
    # The mutual information of "abb" is log2(P(abb)/(P(ab)P(b)))
    # We have P(abb) == 1/3, P(ab) == 1/4, P(b) == 3/5
    # so mutual information is log2((1/3)/(3/20)
    assert(mutual_information_bi == np.log2((1/3)/(3/20)))

def test_mutual_information_smoothed():

    phonestats = PhoneStats(3, smoothing=1, use_boundary_tokens=False)
    phonestats.add_phones(['a', 'b', 'b', 'b', 'c'])

    mutual_information_uni = phonestats._mutual_information(['a'], ['b'])
    mutual_information_bi = phonestats._mutual_information(['a', 'b'], ['b'])

    # The mutual information of "ab" is log2(P(ab)/(P(a)P(b)))
    # We have P(ab) == 2/5, P(a) == 2/6, P(b) == 4/6
    # so mutual information is log2((2/5)/(2/9)
    assert(mutual_information_uni == np.log2((2/5)/(2/9)))
    # The mutual information of "abb" is log2(P(abb)/(P(ab)P(b)))
    # We have P(abb) == 2/4, P(ab) == 2/5, P(b) == 4/6
    # so mutual information is log2((2/4)/(4/15)
    assert(mutual_information_bi == np.log2((2/4)/(4/15)))


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

    phonestats = PhoneStats(5, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    measure = "ent" # getting boundary entropy
    reverse = False # forward measure

    for ngram_length in range(1, 5):
        position = ngram_length
        unpredictability = phonestats.get_unpredictability(phones, position, measure, reverse, ngram_length)
        ngram_entropy = phonestats._boundary_entropy(phones[:ngram_length])

        assert(unpredictability == ngram_entropy)

def test_get_boundary_entropy_at_bad_position_returns_none():
    """
    For utterance "abbbc" checks if finding the ngram entropy at position n-2
    returns None (since there is no ngram before that position).
    """

    phonestats = PhoneStats(5, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    measure = "ent" # getting boundary entropy
    reverse = False # forward measure

    for ngram_length in range(1, 5):
        position = ngram_length - 1
        unpredictability = phonestats.get_unpredictability(phones, position, measure, reverse, ngram_length)

        assert(unpredictability == None)

def test_get_reverse_boundary_entropy_in_utterance_uses_correct_ngram():
    """
    For utterance "abbbc" checks if finding the reverse ngram entropy at position 5-n-1
    returns the entropy of the correct ngram. For the bigram case, this is "bc" at position 2.
    """

    phonestats = PhoneStats(5, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    measure = "ent" # getting boundary entropy
    reverse = True # forward measure

    for ngram_length in range(1, 5):
        position = 5 - ngram_length
        unpredictability = phonestats.get_unpredictability(phones, position, measure, reverse, ngram_length)
        ngram_entropy = phonestats._boundary_entropy_reverse(phones[5-ngram_length:])

        assert(unpredictability == ngram_entropy)

def test_get_reverse_boundary_entropy_at_late_position_returns_none():
    """
    For utterance "abbbc" checks if finding the ngram entropy at position 5-n
    returns None (since there is no ngram after that position).
    """

    phonestats = PhoneStats(5, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    measure = "ent" # getting boundary entropy
    reverse = True # reverse measure

    for ngram_length in range(1, 5):
        position = 5 - ngram_length + 1
        unpredictability = phonestats.get_unpredictability(phones, position, measure, reverse, ngram_length)

        assert(unpredictability == None)

def test_get_transitional_probability_computes_conditional_entropy():
    """
    For utterance "abbbc" checks if finding the ngram transitional probability at position n-1
    returns the entropy of the correct ngram. For the bigram case, this is "ab" at position 1.
    """

    phonestats = PhoneStats(5, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'b', 'b', 'b', 'c']
    phonestats.add_phones(phones)

    measure = "tp" # getting boundary entropy
    reverse = False # forward measure

    for ngram_length in range(1, 5):
        position = ngram_length
        unpredictability = phonestats.get_unpredictability(phones, position, measure, reverse, ngram_length)
        ngram_cond_prob = phonestats._conditional_probability_reverse(phones[:ngram_length], [phones[ngram_length]])

        assert(unpredictability == - ngram_cond_prob)

"""
----------------------------------------------
            WORD PROBABILITY TESTS
----------------------------------------------
"""

def test_word_probability_ngram_0_unsmoothed():

    phonestats = PhoneStats(1, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'a', 'b', 'c']
    phonestats.add_phones(phones)

    prob = phonestats.get_log_word_probability(phones, 0)
    
    # P('aabc') = -log2(P('a') * P('a') * P('b') * P('c'))
    assert(pow(2, -prob) == 1/2 * 1/2 * 1/4 * 1/4)

def test_word_probability_ngram_0_smoothed():

    phonestats = PhoneStats(1, smoothing=1, use_boundary_tokens=False)
    phones = ['a', 'a', 'b', 'c']
    phonestats.add_phones(phones)

    prob = phonestats.get_log_word_probability(phones, 0)
    
    # P('aabc') = -log2(P('a') * P('a') * P('b') * P('c'))
    assert(pow(2, -prob) - 3/5 * 3/5 * 2/5 * 2/5 < 1e-10)

def test_word_probability_ngram_1_unsmoothed():

    phonestats = PhoneStats(2, smoothing=0, use_boundary_tokens=True)
    phones = ['a', 'a', 'b', 'c']
    phonestats.add_phones(phones)

    prob = phonestats.get_log_word_probability(phones, 1)

    # P('aabc') = -log2(P('a'|ub) * P('a'|'a') * P('b'|'a') * P('c'|'b'))
    assert(abs(pow(2, -prob) - 1/2 * 1/2 * 1/2 * 1) < 1e-10)

def test_word_probability_ngram_1_smoothed():

    phonestats = PhoneStats(2, smoothing=1, use_boundary_tokens=True)
    phones = ['a', 'a', 'b', 'c']
    phonestats.add_phones(phones)

    prob = phonestats.get_log_word_probability(phones, 1)

    # P('aabc') = -log2(P('a'|ub) * P('a'|'a') * P('b'|'a') * P('c'|'b'))
    assert(abs(pow(2, -prob) - 2/3 * 2/3 * 2/3 * 1) < 1e-10)

def test_word_probability_ngram_0_unseen_unsmoothed():

    phonestats = PhoneStats(1, smoothing=0, use_boundary_tokens=False)
    phones = ['a', 'a', 'b', 'c']
    word = ['b', 'd']
    phonestats.add_phones(phones)

    prob = phonestats.get_log_word_probability(word, 0)

    # P('bd') = -log2(P('b') * P('d')) but 'd' is unseen, so log2(0) is set to -10000
    assert(abs(prob + np.log2(1/4) + -10000) < 1e-10)

def test_word_probability_ngram_0_unseen_smoothed():

    phonestats = PhoneStats(1, smoothing=1, use_boundary_tokens=False)
    phones = ['a', 'a', 'b', 'c']
    word = ['b', 'd']
    phonestats.add_phones(phones)

    prob = phonestats.get_log_word_probability(word, 0)

    # P('bd') = -log2(P('b') * P('d')) but 'd' is unseen, so P('d') = 1/5
    assert(abs(prob + np.log2(2/5) + np.log2(1/5)) < 1e-10)
