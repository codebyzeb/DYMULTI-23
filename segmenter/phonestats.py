""" A class for storing phoneme counts and calculating statistics while processing a corpus.

Provides various methods for probability and predictability.

"""

import collections
import sortedcontainers
import numpy as np

BOUNDARY_TOKEN = "BOUND"

class PhoneStats:
    """ Stores phoneme n-gram counts and provides methods for calculating information-theoretic measures

    Parameters
    ----------
    max_ng : int, optional
        The maximum length of ngrams to keep track of.
    smoothing : float, optional
        If non-zero, use add-k smoothing for probabilities.

    Raises
    ------
    ValueError
        If max_ng is less than 1.

    """

    def __init__(self, max_ngram=1, smoothing=0, use_boundary_tokens=False):

        if max_ngram < 1:
            raise(ValueError(str(max_ngram) + " is not a valid ngram length, cannot initialise PhoneStats object."))

        self.max_ngram = max_ngram
        self.types = {i+1 : sortedcontainers.SortedSet() for i in range(max_ngram)}
        self.ngrams = collections.Counter()
        self.ntokens = {i+1 : 0 for i in range(max_ngram)}

        self.smoothing = smoothing
        self.use_boundary_tokens = use_boundary_tokens

    def add_utterance(self, utterance):
        """ Update ngram counts given an utterance.
        
        Parameters
        ----------
        utterance : list of str
            An utterance represented by a list of phonemes.
        
        """

        # If using boundary tokens, pad the utterance with bondary tokens
        if self.use_boundary_tokens:
            utterance = [BOUNDARY_TOKEN] * (self.max_ngram - 1) + utterance + [BOUNDARY_TOKEN] * (self.max_ngram - 1)

        for n in range(1, self.max_ngram+1):
            # collapse ngrams to strings
            ngrams = [utterance[i:i+n] for i in range(len(utterance)-(n-1))]
            self.ngrams.update([''.join(ngram) for ngram in ngrams])
            self.ntokens[n] += len(ngrams)
            for ngram in ngrams:
                self.types[n].add(tuple(ngram))

    def _check_n(self, n):
        if n > self.max_ngram or n < 1:
            raise(ValueError("Ngrams of length " + str(n) + " not stored in this PhoneStats object."))
    
    def _types(self, n=1):
        """ Returns a list of ngram types of length n """

        self._check_n(n)
        return [list(ngram) for ngram in self.types[n]]

    def probability(self, ngram):
        """ Returns P(ngram), using add-k smoothing if self.smoothing > 0
        
        Parameters
        ----------
        ngram : str list
            The n-gram to calculate the probability of, given current counts. Represented as a 
            list of phonemes of length n.

        Returns
        -------
        p : double
            P(ngram) calculated as relative frequency of that ngram

        """

        n = len(ngram)
        self._check_n(n)

        # TODO: Implement n-gram backoff?
        freq = self.ngrams[''.join(ngram)]

        if self.ntokens[n] + self.smoothing == 0:
            return 0
        return (freq + self.smoothing) / (self.ntokens[n] + self.smoothing)
        
    def _conditional_probability(self, ngram_A, ngram_B):
        """ Return P(ngram_A | ngram_B) assuming ngram_B follows ngram_A

        Makes an assumption that P(ngram ends with ngram_B)
        can be approximated with P(ngram_B). This can lead to probabilities over 1 when the number
        counts of ngrams of different lengths diverge (as can happen when utterances are processed
        individually, but only if utterance boundaries aren't used).

        """

        self._check_n(len(ngram_A) + len(ngram_B))

        freq_AB = self.ngrams[''.join(ngram_A + ngram_B)]
        freq_B = self.ngrams[''.join(ngram_B)]

        if freq_B + self.smoothing == 0:
            return 0
        return (freq_AB + self.smoothing) / (freq_B + self.smoothing)

    def _conditional_probability_reverse(self, ngram_A, ngram_B):
        """ Returns P(ngram_B | ngram_A) assuming ngram_B follows ngram_A.

        Makes an assumption that P(ngram starts with ngram_A)
        can be approximated with P(ngram_A). This can lead to probabilities over 1 when the number
        counts of ngrams of different lengths diverge (as can happen when utterances are processed
        individually, but only if utterance boundaries aren't used).

        """

        self._check_n(len(ngram_A) + len(ngram_B))

        freq_AB = self.ngrams[''.join(ngram_A + ngram_B)]
        freq_A = self.ngrams[''.join(ngram_A)]

        if freq_A + self.smoothing == 0:
            return 0
        return (freq_AB + self.smoothing) / (freq_A + self.smoothing)

    def _boundary_entropy(self, ngram):
        """ Calculates H(ngram) using the boundary entropy equation """ 

        probs = np.array([self._conditional_probability_reverse(ngram, unigram) for unigram in self._types()])
        # Calculate reverse entropy. Only take log2 where the value is non-zero, otherwise
        # get division by zero error. 
        # This is ok since xlogx tends to 0 as x tends to 0, so we can ignore these terms.
        return - probs[probs != 0].dot(np.log2(probs[probs != 0]))

    def _boundary_entropy_reverse(self, ngram):
        """ Calculates H(ngram) using the reverse boundary entropy equation """ 

        probs = np.array([self._conditional_probability(unigram, ngram) for unigram in self._types()])
        # Calculate entropy. Only take log2 where the value is non-zero, otherwise
        # get division by zero error. 
        # This is ok since xlogx tends to 0 as x tends to 0, so we can ignore these terms.
        return - probs[probs != 0].dot(np.log2(probs[probs != 0]))

    def _successor_variety(self, ngram):
        """ Returns the number of ngrams that have the prefix ngram """

        self._check_n(len(ngram) + 1)
        return np.count_nonzero([self.ngrams[''.join(ngram + unigram)] for unigram in self._types()])

    def _successor_variety_reverse(self, ngram):
        """ Returns the number of ngrams that have the suffix ngram_X """

        self._check_n(len(ngram) + 1)
        return np.count_nonzero([self.ngrams[''.join(unigram + ngram)] for unigram in self._types()])

    def _mutual_information(self, ngram_A, ngram_B):
        """ Calculates the mutual information of ngram_A and ngram_B, assuming that B follows A """

        prod = (self.probability(ngram_A) * self.probability(ngram_B))
        # TODO: Check if this assumption is ok
        if prod == 0:
            return 0
        return np.log2(self.probability(ngram_A + ngram_B) / prod)

    def get_unpredictability(self, utterance, position, measure, reverse, ngram_length):
        """ Returns the unpredictability calculated at a particular position in the utterance.

        For example, for utterance "abcd", get_predictability(['a','b','c','d'], 1)
        with reverse=False, ngram_length=2 and measure="ent" will calculate
        the boundary entropy of the bigram "ab".

        Parameters
        ----------
        utterance : list of str
            An utterance represented by a list of phonemes.
        position : int
            The index in the utterance of the phoneme after which we are considering a boundary.
        measure : str
            The name of the measure to use to calculate predictability.
        reverse : bool
            Whether to use the reverse unpredictabilty measure.
        ngram_length : int
            The length of ngram to use in the calculation.

        Returns
        -------
        unpredictability : float
            The unpredictability of the ngram at the given position in the utterance.

        Raises
        ------
        ValueError
            If `measure` is not a valid measure or if ngram_length is an invalid length.
        """

        self._check_n(ngram_length)
        boundary = position + 1 # makes ranged indexing neater

        # Pad utterance with boundary tokens and shift boundary index accordingly
        if self.use_boundary_tokens:
            utterance = [BOUNDARY_TOKEN] * (self.max_ngram - 1) + utterance + [BOUNDARY_TOKEN] * (self.max_ngram - 1)
            boundary += self.max_ngram - 1

        if reverse:
            if not self.use_boundary_tokens and boundary + ngram_length > len(utterance):
                return None
            right_context = utterance[boundary:boundary+ngram_length]
            left_context = [utterance[boundary-1]]
        else:
            if not self.use_boundary_tokens and boundary < ngram_length:
                return None
            if measure in ["tp", "mi"]:
                if not self.use_boundary_tokens and boundary >= len(utterance):
                    return None
                else:
                    right_context = [utterance[boundary]]
            left_context = utterance[boundary-ngram_length:boundary]

        # Large boundary entropy = high unpredictability
        if measure == "ent" and reverse:
            return self._boundary_entropy_reverse(right_context)
        elif measure == "ent" and not reverse:
            return self._boundary_entropy(left_context)

        # Large transitional probability = low unpredictability (so return negative)
        elif measure == "tp" and reverse:
            return - self._conditional_probability(left_context, right_context)
        elif measure == "tp" and not reverse:
            return - self._conditional_probability_reverse(left_context, right_context)

        # Large successor variety indicates the end of a word (used like high unpredictability)
        elif measure == "sv" and reverse:
            return self._successor_variety_reverse(right_context)
        elif measure == "sv" and not reverse:
            return self._successor_variety(left_context)

        # Large mutual information = low unpredictability (so return negative)
        elif measure == "mi" and reverse:
            return - self._mutual_information(left_context, right_context)
        elif measure == "mi" and not reverse:
            return - self._mutual_information(left_context, right_context)

        else:
            raise ValueError("Unknown predictability measure: '{}'".format(measure))

