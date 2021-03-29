""" A lexicon-based algorithm for word segmentation.

TODO: Add documentation

"""

from sortedcontainers import SortedDict

from wordseg import utils

from segmenter.phonestats import PhoneStats
from segmenter.model import PeakModel

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

class LexiconFrequencyModel(PeakModel):
    """ A simple lexicon-based model for word segmentation.

    Based on "An explicit statistical model of learning lexical segmentation using multiple cues"
    (Çöltekin et al, 2014). Counts the frequency of previously-seen words that start and end at the
    possible boundary within the utterance and uses an increase or decrease of this sum to posit
    a boundary.
    
    For example, if a lexicon include "i", "it", "its", "a", "baby" and "by", all with
    frequency 1, then the count at the position after the "a" in "itsababy" is 2 since "a"
    occurs once and "baby" occurs once, wheras the count at the position before the y is 0
    since "y" is not a word in the lexicon, nor is any sequence of phonemes ending in "b". 

    Parameters
    ----------
    increase : bool, optional
        When true, place word boundaries when the total word frequency at a candidate boundary.
        When false, place word boundaries when the total word frequency decreases after a candidate boundary.
    use_presence : bool, optional
        When True, consider all words in the lexicon to be equally frequent. This is equivilent to using
        type counts, rather than token counts of previously-seen words.
    lexicon : Lexicon, optional
        If not provided, a Lexicon object will be created and updated to keep track of previously-seen
        words. 
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, increase=True, right=True, use_presence=False, lexicon=None, log=utils.null_logger()):
        super().__init__(increase, log)

        # Initialise model parameters
        self.use_presence = use_presence
        self.right = right

        # Initialise lexicon if not provided
        if lexicon is None:
            self._lexicon = Lexicon()
            self._updatelexicon = True
        else:
            self._lexicon = lexicon
            self._updatelexicon = False

    def __str__(self):
        return "LexiconFreqModel({},{},{})".format(
            "Increase" if self.increase else "Decrease",
            "Type Frequency" if self.use_presence else "Token Frequency",
            "Right Context" if self.right else "Left Context")

    # Overrides PeakModel.score()
    def score(self, utterance, position):
        """
        Returns a score for the candidate boundary after utterance.phones[i], calculated
        according to the words in the lexicon found in the utterance that start or end at that boundary.
        """

        if self.right:
            candidate_words = [''.join(utterance.phones[position+1:j]) for j in range(position+2, len(utterance)+1)]
        else:
            candidate_words = [''.join(utterance.phones[j:position+1]) for j in range(0, position+1)]

        if self.use_presence:
            word_count = sum([1 for word in candidate_words if self._lexicon[word] > 0])
        else:
            word_count = sum([self._lexicon[word] for word in candidate_words])

        return word_count

    # Overrides PeakModel.update()
    def update(self, segmented):
        """ Updates lexicon with newly found words. """
        if self._updatelexicon:
            for word in segmented.get_words():
                self._lexicon.increase_count(''.join(word))

class LexiconBoundaryModel(PeakModel):
    """ A simple lexicon-based model for word segmentation.

    Based on "An explicit statistical model of learning lexical segmentation using multiple cues"
    (Çöltekin et al, 2014). Calculates P(boundary | left context) or P(boundary | right context) using
    phoneme statistics gathered from the lexicon (treating each previously-seen word as its own utterance).
    
    For example, if a lexicon include "abb" and "cd" then P(right boundary | "bb") will be high, whereas
    P(right boundary | "ab") will be low (since no entries in the lexicon end in "ab").

    Parameters
    ----------
    ngram_length : int, optional
        The length of n-grams used in the context.
    increase : bool, optional
        When true, place word boundaries when the boundary probability increases at a candidate boundary.
        When false, place word boundaries when the boundary probability decreases after a candidate boundary.
    right : bool, optional
        When True, calculates P(boundary | right context).
        When False, calculates P(boundary | left context).
    lexicon : Lexicon, optional
        If not provided, a Lexicon object will be created and updated to keep track of previously-seen
        words. 
    phonestats : phonestats.PhoneStats, optional
        If not provided, a PhoneStats object will be created and updated to keep track of phoneme counts.
        Otherwise, the provided object will be used, but not updated.
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, ngram_length=1, increase=True, right=True, lexicon=None, phonestats=None, log=utils.null_logger()):
        super().__init__(increase, log)

        # Initialise model parameters
        self.ngram_length = ngram_length
        self.right = right

        # Initialise lexicon if not provided
        if lexicon is None:
            self._lexicon = Lexicon()
            self._updatelexicon = True
        else:
            self._lexicon = lexicon
            self._updatelexicon = False

        # Initialise phoneme statistics if not provided
        if phonestats is None:
            self._phonestats = PhoneStats(ngram_length+1, use_boundary_tokens=True)
            self._updatephonestats = True
        else:
            self._phonestats = phonestats
            self._updatephonestats = False

    def __str__(self):
        return "LexiconBoundaryModel({},{},{})".format(
            "N: " + str(self.ngram_length),
            "Increase" if self.increase else "Decrease",
            "Right Context" if self.right else "Left Context")

    # Overrides PeakModel.score()
    def score(self, utterance, position):
        """
        Returns a score for the candidate boundary after utterance.phones[i], calculated
        using the conditional boundary probability using the phonestats of the lexicon.
        """
        return self._phonestats.get_unpredictability(
            phones=utterance.phones, position=position, measure="bp",
            right=self.right, ngram_length=self.ngram_length)

    # Overrides PeakModel.update()
    def update(self, segmented):
        """ Updates lexicon and phonestats with newly found words. """
        for word in segmented.get_words():
            if self._updatelexicon:
                self._lexicon.increase_count(''.join(word))
            if self._updatephonestats:
                self._phonestats.add_phones(word)
