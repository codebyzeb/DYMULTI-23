""" A class for storing phoneme counts and calculating statistics while processing a corpus.

Provides various methods for probability and predictability.

"""

import collections
import numpy as np

BOUND = "BOUND"

class PhoneStats:
    """ Stores phoneme n-gram counts and provides methods for calculating information-theoretic measures

    Parameters
    ----------
    max_ng : int, optional
        The maximum length of ngrams to keep track of.
    smoothing : bool, optional
        If true, use add1 smoothing for probabilities.
    correct_conditional : bool, optional
        If true, use the correct method for calculating conditional probaiblities, otherwise
        use the assumption that lower n-gram counts can be used.

    Raises
    ------
    ValueError
        If max_ng is less than 1.

    """

    def __init__(self, max_ngram=1, smoothing=True, correct_conditional=False, use_boundary_tokens=False):

        if max_ngram < 1:
            raise(ValueError(str(max_ngram) + " is not a valid ngram length, cannot initialise PhoneStats object."))

        self.max_ngram = max_ngram
        self.ngrams = {i+1 : collections.Counter() for i in range(max_ngram)}
        self.ntokens = {i+1 : 0 for i in range(max_ngram)}

        self.smoothing = smoothing
        self.correct_conditional = correct_conditional
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
            utterance = [BOUND] * (self.max_ngram - 1) + utterance + [BOUND] * (self.max_ngram - 1)

        for n in range(1, self.max_ngram+1):
            # ngrams indexed as tuples
            ngrams = [tuple(utterance[i:i+n]) for i in range(len(utterance)-(n-1))]
            self.ngrams[n].update(ngrams)
            self.ntokens[n] += len(ngrams)

    def _check_n(self, n):
        if n > self.max_ngram or n < 1:
            raise(ValueError("Ngrams of length " + str(n) + " not stored in this PhoneStats object."))
    
    def _types(self, n=1):
        """ Returns a list of ngram types seen """

        self._check_n(n)
        return [list(key) for key in self.ngrams[n].keys()]

    def probability(self, ngram):
        """ Returns P(ngram), using add1 smoothing if smoothing is set
        
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
        freq = self.ngrams[n][tuple(ngram)]

        if self.smoothing:
            return (freq + 1) / (self.ntokens[n] + 1)
        else:
            return (freq / self.ntokens[n]) if self.ntokens[n] > 0 else 0

    def _conditional_probability(self, ngram_A, ngram_B):
        """ Return P(ngram_A | ngram_B) assuming ngram_B follows ngram_A

        If self.correct_conditional is False, makes an assumption that P(ngram ends with ngram_B)
        can be approximated with P(ngram_B). This can lead to probabilities over 1 when the number
        counts of ngrams of different lengths diverge (as can happen when utterances are processed
        individually).
        If self.correct_conditional is True, will calculate the conditional correctly, by counting
        the number of ngrams that end in ngram_B, but this is more expensive as we need to loop over
        ngrams.

        """

        len_A = len(ngram_A)
        len_B = len(ngram_B)
        self._check_n(len_A + len_B)

        freq_AB = self.ngrams[len_A + len_B][tuple(ngram_A + ngram_B)]
        freq_B = self.ngrams[len_B][tuple(ngram_B)]

        if not self.correct_conditional:
            if self.smoothing:
                return (freq_AB + 1) / (freq_B + 1)
            else:
                return freq_AB / freq_B if freq_B != 0 else 0
        else:
            ngrams_end_with_ngram_B = np.sum([self.ngrams[len_A + len_B][tuple(ngram_X + ngram_B)] for ngram_X in self._types(len_A)])
            if self.smoothing:
                return (freq_AB + 1) / (ngrams_end_with_ngram_B + 1)
            else:
                return freq_AB / ngrams_end_with_ngram_B if ngrams_end_with_ngram_B != 0 else 0

    def _conditional_probability_reverse(self, ngram_A, ngram_B):
        """ Returns P(ngram_B | ngram_A) assuming ngram_B follows ngram_A.

        If self.correct_conditional is False, makes an assumption that P(ngram starts with ngram_A)
        can be approximated with P(ngram_A). This can lead to probabilities over 1 when the number
        counts of ngrams of different lengths diverge (as can happen when utterances are processed
        individually).
        If self.correct_conditional is True, will calculate the conditional correctly, by counting
        the number of ngrams that start in ngram_A, but this is more expensive as we need to loop
        over ngrams.

        """

        len_A = len(ngram_A)
        len_B = len(ngram_B)
        self._check_n(len_A + len_B)

        freq_AB = self.ngrams[len_A + len_B][tuple(ngram_A + ngram_B)]
        freq_A = self.ngrams[len_A][tuple(ngram_A)]

        if not self.correct_conditional:
            if self.smoothing:
                return (freq_AB + 1) / (freq_A + 1)
            else:
                return freq_AB / freq_A if freq_A != 0 else 0
        else:
            ngrams_start_with_ngram_A = np.sum([self.ngrams[len_A + len_B][tuple(ngram_A + ngram_Y)] for ngram_Y in self._types(len_B)])
            if self.smoothing:
                return (freq_AB + 1) / (ngrams_start_with_ngram_A + 1)
            else:
                return freq_AB / ngrams_start_with_ngram_A if ngrams_start_with_ngram_A != 0 else 0

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

        ngram_length = len(ngram) + 1
        self._check_n(ngram_length)
        return np.count_nonzero([self.ngrams[ngram_length][tuple(ngram + unigram)] for unigram in self._types()])

    def _successor_variety_reverse(self, ngram):
        """ Returns the number of ngrams that have the suffix ngram_X """

        ngram_length = len(ngram) + 1
        self._check_n(ngram_length)
        return np.count_nonzero([self.ngrams[ngram_length][tuple(unigram + ngram)] for unigram in self._types()])

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

        if self.use_boundary_tokens:
            utterance = [BOUND] * (self.max_ngram - 1) + utterance + [BOUND] * (self.max_ngram - 1)
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

