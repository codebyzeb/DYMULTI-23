""" A class for storing phoneme counts and calculating statistics while processing a corpus.

Provides various methods for probability and predictability.

"""

import collections
import numpy as np

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

    def __init__(self, max_ng=1, smoothing=True, correct_conditional=False):

        if max_ng < 1:
            raise(ValueError(str(max_ng) + " is not a valid ngram length, cannot initialise PhoneStats object."))

        self.max_ng = max_ng
        self.ngrams = {i+1 : collections.Counter() for i in range(max_ng)}
        self.ntokens = {i+1 : 0 for i in range(max_ng)}

        self.smoothing = smoothing
        self.correct_conditional = correct_conditional
    
    def add_utterance(self, utterance):
        """ Update ngram counts given an utterance.
        
        Parameters
        ----------
        utterance : list of str
            An utterance represented by a list of phonemes.
        
        """

        for n in range(1, self.max_ng+1):
            # ngrams indexed as tuples
            ngrams = [tuple(utterance[i:i+n]) for i in range(len(utterance)-(n-1))]
            self.ngrams[n].update(ngrams)
            self.ntokens[n] += len(ngrams)

    def _check_n(self, n):
        if n > self.max_ng or n < 1:
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

    def _probability_ngram_start(self, ngram_X, ngram_length):
        """ Returns the probablity the probabilty that an ngram of length ngram_length
        starts with ngram_X. Uses add1 smoothing. """

        self._check_n(ngram_length)
        if len(ngram_X) > ngram_length:
            raise ValueError("Cannot calculate the probability that an ngram of length {}"
                            " starts with a longer ngram {} of length {}.".format(
                                ngram_length, ngram_X, len(ngram_X)))
        
        ngram_Y_length = ngram_length - len(ngram_X)
        ngrams_start_with_ngram_X = np.sum([self.ngrams[ngram_length][tuple(ngram_X + ngram_Y)] for ngram_Y in self._types(ngram_Y_length)])
        
        if self.smoothing:
            return (ngrams_start_with_ngram_X + 1) / (self.ntokens[ngram_length] + 1)
        else:
            return (ngrams_start_with_ngram_X / self.ntokens[ngram_length]) if self.ntokens[ngram_length] > 0 else 0

    def _probability_ngram_end(self, ngram_X, ngram_length):
        """ Returns the probablity the probabilty that an ngram of length ngram_length
        ends with ngram_X. Uses add1 smoothing. """

        self._check_n(ngram_length)
        if len(ngram_X) > ngram_length:
            raise ValueError("Cannot calculate the probability that an ngram of length {}"
                            " ends with a longer ngram {} of length {}.".format(
                                ngram_length, ngram_X, len(ngram_X)))
        
        ngram_Y_length = ngram_length - len(ngram_X)
        ngrams_end_with_ngram_X = np.sum([self.ngrams[ngram_length][tuple(ngram_Y + ngram_X)] for ngram_Y in self._types(ngram_Y_length)])
        
        if self.smoothing:
            return (ngrams_end_with_ngram_X + 1) / (self.ntokens[ngram_length] + 1)
        else:
            return (ngrams_end_with_ngram_X / self.ntokens[ngram_length]) if self.ntokens[ngram_length] > 0 else 0

    def _conditional_probability(self, ngram_A, ngram_B):
        """ Return P(ngram_A | ngram_B) assuming ngram_B follows ngram_A

        If self.correct_conditional is False, makes an assumption that P(ngram ends with ngram_B)
        can be approximated with P(ngram_B). This can lead to probabilities over 1 when the number
        counts of ngrams of different lengths diverge (as can happen when utterances are processed
        individually).
        If self.correct_conditional is True, will calculate the conditional correctly, using
        self._probability_ngram_end, but this is more expensive as we need to loop over ngrams.

        """

        if not self.correct_conditional:
            return self.probability(ngram_A + ngram_B) / self.probability(ngram_B)
        else:
            joint_prob = self.probability(ngram_A + ngram_B)
            prob_end_with_ngram_B = self._probability_ngram_end(ngram_B, len(ngram_A) + len(ngram_B))

            return joint_prob / prob_end_with_ngram_B

    def _conditional_probability_reverse(self, ngram_A, ngram_B):
        """ Returns P(ngram_B | ngram_A) assuming ngram_B follows ngram_A.

        If self.correct_conditional is False, makes an assumption that P(ngram starts with ngram_A)
        can be approximated with P(ngram_A). This can lead to probabilities over 1 when the number
        counts of ngrams of different lengths diverge (as can happen when utterances are processed
        individually).
        If self.correct_conditional is True, will calculate the conditional correctly, using
        self._probability_ngram_start, but this is more expensive as we need to loop over ngrams.

        """

        if not self.correct_conditional:
            return self.probability(ngram_A + ngram_B) / self.probability(ngram_A)
        else:
            joint_prob = self.probability(ngram_A + ngram_B)
            prob_start_with_ngram_A = self._probability_ngram_start(ngram_A, len(ngram_A) + len(ngram_B))

            return joint_prob / prob_start_with_ngram_A

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

        if measure == "ent":
            if reverse:
                if boundary + ngram_length > len(utterance):
                    return None
                else:
                    return self._boundary_entropy_reverse(utterance[boundary:boundary+ngram_length])
            else:
                if boundary < ngram_length:
                    return None
                else:
                    return self._boundary_entropy(utterance[boundary-ngram_length:boundary])
        else:
            raise ValueError("Unknown predictability measure: '{}'".format(measure))
