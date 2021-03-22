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

    def __init__(self, phones=[]):
        self.phones = []
        for phone in phones:
            if phone != ' ':
                self.phones.append(phone)
        # Assume there is a boundary after the last phone
        self.boundaries = [False] * (len(phones) - 1) + [True]

    def get_words(self):
        """ Returns the words in the PhoneSequence, split according to the boundaries """
        words = []
        i = 0
        for j in range(len(self.phones)):
            if self.boundaries[j]:
                words.append(self.phones[i:j+1])
                i = j+1
        if i != j+1:
            words.append(self.phones[i:j+1])
        return words

    def __str__(self):
        """ Return the phones sequence as a space-separated string using the word boundaries """
        seq = ""
        for i, phone in enumerate(self.phones):
            seq += phone + (' ' if self.boundaries[i] else '')
        return seq.strip()

    def __len__(self):
        return len(self.phones)
    
