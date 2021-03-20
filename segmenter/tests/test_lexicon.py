""" Run tests for the Lexicon class """

from segmenter.lexicon import Lexicon

def test_increase_count():

    lexicon = Lexicon()

    lexicon.increase_count("word", 10)
    lexicon.increase_count("word", 10)
    lexicon.increase_count("other")

    assert(lexicon["word"] == 20)
    assert(lexicon["other"] == 1)
    assert(list(lexicon) == ["other", "word"])

def test_access_missing_word_gives_zero():

    lexicon = Lexicon()

    assert(lexicon["word"] == 0)

def test_increase_count_invalid_word_does_nothing():

    lexicon = Lexicon()

    lexicon.increase_count(None, 10)
    lexicon.increase_count("", 10)

    assert(len(lexicon) == 0)