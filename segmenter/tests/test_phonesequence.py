""" Run tests for the PhoneSequence class """

from segmenter.phonesequence import PhoneSequence 

def test_init():
    phones = ['a', 'b']
    stress = ['0', '1']
    boundaries = [True, False]

    seq = PhoneSequence(phones, stress)

    assert(seq.phones == phones)
    assert(seq.boundaries == boundaries)
    assert(seq.stress == stress)

def test_get_words():
    phones = ['a', 'b', 'c', 'd']
    boundaries = [False, False, True, True]

    seq = PhoneSequence(phones)
    seq.boundaries = boundaries
    words = seq.get_words()

    assert(words == [['a', 'b'], ['c'], ['d']])

def test_get_word_stress():
    phones = ['a', 'b', 'c', 'd']
    stress = ['0', '1', '2', '3']
    boundaries = [False, False, True, True]

    seq = PhoneSequence(phones, stress)
    seq.boundaries = boundaries
    word_stresses = seq.get_word_stress()

    assert(word_stresses == [['0', '1'], ['2'], ['3']])

def test_str():
    phones = ['a', 'b', 'c', 'd']
    boundaries = [False, False, True, True]

    seq = PhoneSequence(phones)
    seq.boundaries = boundaries

    assert(str(seq) == "ab c d")

def test_len():
    phones = ['a', 'b', 'c', 'd']

    seq = PhoneSequence(phones)

    assert(len(seq) == 4)
