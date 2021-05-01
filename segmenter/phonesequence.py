""" PhoneSequence class to represet sequences of phonemes. """

class PhoneSequence:
    """
    Represents a sequence of phonemes, used to store utterances or words. When initialised,
    it is assumed that there is a boundary at the end of the list of phones.

    PhoneSequence.boundaries stores a list of word boundaries, where boundary[i] = True
    indicates that there is a word boundary AFTER phonemes[i]. 

    Parameters
    ----------
    phones : list of str, optional
        Initialises the list of phones in this sequence. Should not include spaces.

    """

    def __init__(self, phones=[], stress=[], boundaries=[]):
        self.phones = []
        for phone in phones:
            if phone != ' ':
                self.phones.append(phone)

        # Assign stress
        self.stress = stress
        if stress == []:
            self.stress = ['0'] * len(phones)

        # Assign boundaries
        self.boundaries = boundaries
        if self.boundaries == []:
            # Assume there is a boundary before the first phone
            self.boundaries = [True] + [False] * (len(phones) - 1)

    def get_words(self):
        """ Returns the words in the PhoneSequence, split according to the boundaries """
        words = []
        i = 0
        for j in range(len(self.phones)):
            if self.boundaries[j]:
                words.append(self.phones[i:j])
                i = j
        words.append(self.phones[i:])
        return words

    def get_word_stress(self):
        """ Returns the stress alignment in the PhoneSequence, split according to the boundaries """
        words = []
        i = 0
        for j in range(len(self.stress)):
            if self.boundaries[j]:
                words.append(self.stress[i:j])
                i = j
        words.append(self.stress[i:])
        return words

    def __str__(self):
        """ Return the phones sequence as a space-separated string using the word boundaries """
        seq = ""
        for i, phone in enumerate(self.phones):
            seq += (' ' if self.boundaries[i] else '') + phone
        return seq.strip()

    def __len__(self):
        return len(self.phones)
