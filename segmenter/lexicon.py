""" A data structure for storing a proto-lexicon.

"""

from sortedcontainers import SortedDict

class Lexicon(SortedDict):
    """ Store a noisy lexicon as a sorted dictionary. The lexicon consists
        of frequency counts for each word.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_count = self.__len__()
        self.token_count = sum([self[k] for k in self])

    def increase_count(self, word, k=1):
        """ Increase the frequency count of a given word by k. `word` passed as a string. """

        if word is None or word == "":
            return
        if not word in self:
            self.setdefault(word, k)
            self.type_count += 1
        else:
            self[word] += k
        self.token_count += k

    def __getitem__(self, key):
        """ Override default implementation to return 0 if the key is not in the dictionary """

        if not key in self:
            return 0
        return super().__getitem__(key)

    def relative_freq(self, word, consider_unseen=False):
        """ Returns the relative frequency of the word. consider_unseen smooths the probability
        by adding the number of types to the denominator, accounting for unseen words """

        if consider_unseen:
            if self[word] == 0 or self.token_count + self.type_count == 0:
                return 0
            return self[word] / (self.token_count + self.type_count)
        else:
            if self[word] == 0 or self.token_count == 0:
                return 0
            return self[word] / self.token_count

    def unseen_freq(self):
        """ Returns the expected frequency of unseen words, such that the sum of all relative
        frequencies for each seen word (using consider_unseen=True) plus this value is 1. """

        if self.type_count == 0 or self.token_count == 0:
            return 0
        return self.type_count / (self.token_count + self.type_count)
