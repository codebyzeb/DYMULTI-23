""" Run tests for the Lexicon class """

from segmenter.lexicon import Lexicon

"""
----------------------------------------------
            INITIALISATION TESTS
----------------------------------------------
"""

def test_init_with_dict_sets_counts():

    lexicon = Lexicon({"a" : 1, "b" : 3})

    assert(lexicon.token_count == 4)
    assert(lexicon.type_count == 2)

def test_init_without_dict_sets_counts():

    lexicon = Lexicon()

    assert(lexicon.token_count == 0)
    assert(lexicon.type_count == 0)

"""
----------------------------------------------
            INCREASE COUNT TESTS
----------------------------------------------
"""

def test_increase_count():

    lexicon = Lexicon()

    lexicon.increase_count("word", 10)
    lexicon.increase_count("word", 10)
    lexicon.increase_count("other")

    assert(lexicon["word"] == 20)
    assert(lexicon["other"] == 1)
    assert(list(lexicon) == ["other", "word"])

def test_increase_count_invalid_word_does_nothing():

    lexicon = Lexicon()

    lexicon.increase_count(None, 10)
    lexicon.increase_count("", 10)

    assert(len(lexicon) == 0)

def test_increase_count_increases_token_count():

    lexicon = Lexicon()

    lexicon.increase_count("word", 10)
    lexicon.increase_count("thing", 1)

    assert(lexicon.token_count == 11)

def test_increase_count_increases_type_count():

    lexicon = Lexicon()

    lexicon.increase_count("word", 10)
    lexicon.increase_count("thing", 1)

    assert(lexicon.type_count == 2)

"""
----------------------------------------------
            ACCESS TESTS
----------------------------------------------
"""

def test_access_missing_word_gives_zero():

    lexicon = Lexicon()

    assert(lexicon["word"] == 0)

"""
----------------------------------------------
            RELATIVE FREQUENCY TESTS
----------------------------------------------
"""

def test_relative_frequency_seen():

    lexicon = Lexicon({"word" : 3, "thing" : 1})

    assert(lexicon.relative_freq("word") == 3/4)
    assert(lexicon.relative_freq("thing") == 1/4)

def test_relative_frequency_seen_consider_unseen():

    lexicon = Lexicon({"word" : 3, "thing" : 1})

    assert(lexicon.relative_freq("word", consider_unseen=True) == 3/6)
    assert(lexicon.relative_freq("thing", consider_unseen=True) == 1/6)

def test_relative_frequency_unseen():

    lexicon = Lexicon({"word" : 3, "thing" : 1})

    assert(lexicon.relative_freq("nope") == 0)

def test_relative_frequency_empty():

    lexicon = Lexicon()

    assert(lexicon.relative_freq("thing") == 0)

"""
----------------------------------------------
            UNSEEN FREQUENCY TESTS
----------------------------------------------
"""

def test_unseen_frequency_seen():

    lexicon = Lexicon({"word" : 3, "thing" : 1})

    assert(lexicon.unseen_freq() == 2/6)

def test_sum_of_unseen_and_seen_frequency_is_one():

    lexicon = Lexicon({"word" : 3, "thing" : 1})

    word_freq = lexicon.relative_freq("word", consider_unseen=True)
    thing_freq = lexicon.relative_freq("thing", consider_unseen=True)
    unseen_freq = lexicon.unseen_freq()

    assert(word_freq + thing_freq + unseen_freq == 1)
