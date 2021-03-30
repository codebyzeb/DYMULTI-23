""" A class for storing phoneme counts and calculating statistics while processing a corpus.

Provides various methods for probability and predictability.

"""

import collections
import sortedcontainers
import numpy as np

BOUNDARY_TOKEN = "BOUND"

class PhoneStats:
    """ Stores phoneme n-gram counts and provides methods for calculating information-theoretic measures.
    
    Stores n-grams as strings, making the assumption than a sequence of characters representing phonemes
    can be uniquely translated list of n phonemes. E.g. the phoneme alphabet can not contain
    'a*', '*a' and '*' as then '*a*' can be decomposed as bi-phones ['*a', '*'] and ['*', 'a*'].

    Parameters
    ----------
    max_ng : int, optional
        The maximum length of ngrams to keep track of.
    smoothing : float, optional
        If non-zero, use add-k smoothing for probabilities.
    use_boundary_tokens : bool, optional
        If True, uses boundary tokens to help calculate probabilities at utterance boundaries.

    Raises
    ------
    ValueError
        If max_ng is less than 1.

    """

    def __init__(self, max_ngram=1, smoothing=0, use_boundary_tokens=True):

        if max_ngram < 1:
            raise(ValueError(str(max_ngram) + " is not a valid ngram length, cannot initialise PhoneStats object."))

        self.max_ngram = max_ngram
        self.types = {i+1 : sortedcontainers.SortedSet() for i in range(max_ngram)}
        self.ngrams = collections.Counter()
        self.ntokens = {i+1 : 0 for i in range(max_ngram)}

        self.smoothing = smoothing
        self.use_boundary_tokens = use_boundary_tokens

    def add_phones(self, phones):
        """ Update ngram counts given an utterance.
        
        Parameters
        ----------
        phones : list of str
            A sequence of phones representing the utterance.
        
        """

        if len(phones) == 0:
            return

        # If using boundary tokens, pad the utterance with bondary tokens
        phones = list(phones)
        if self.use_boundary_tokens:
            phones = [BOUNDARY_TOKEN] * (self.max_ngram - 1) + phones + [BOUNDARY_TOKEN] * (self.max_ngram - 1)

        for n in range(1, self.max_ngram+1):
            # collapse ngrams to strings
            ngrams = [phones[i:i+n] for i in range(len(phones)-(n-1))]
            self.ngrams.update([''.join(ngram) for ngram in ngrams])
            self.ntokens[n] += len(ngrams)
            for ngram in ngrams:
                self.types[n].add(tuple(ngram))

    def _check_n(self, n):
        if n > self.max_ngram or n < 1:
            raise(ValueError("Ngrams of length " + str(n) + " not stored in this PhoneStats object."))
    
    def _types(self, n=1):
        """ Returns a list of ngram types of length n. """

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
        prob = self.probability(ngram_A + ngram_B)
        # TODO: Check if this assumption is ok
        if prod == 0 or prob == 0:
            return 0
        return np.log2(prob / prod)

    def get_unpredictability(self, phones, position, measure, right, ngram_length):
        """ Returns the unpredictability calculated at a particular position in the utterance.

        For example, for utterance "abcd", get_predictability(['a','b','c','d'], 1)
        with reverse=False, ngram_length=2 and measure="ent" will calculate
        the boundary entropy of the bigram "ab".

        Parameters
        ----------
        phones : list of str
            A list of phones.
        position : int
            The index in the utterance of the phoneme after which we are considering a boundary.
        measure : str
            The name of the measure to use to calculate predictability.
        right : bool
            Whether to use the right context for the calculation. If False, use the left context.
        ngram_length : int
            The length of ngram to use in the calculation.

        Returns
        -------
        unpredictability : float
            The unpredictability of the ngram at the given position in the utterance.

        Raises
        ------
        ValueError
            If `measure` is not a valid measure, if ngram_length is an invalid length or
            if `measure` is "bp" and use_boundary_tokens is not set.
        """

        if measure == "bp" and not self.use_boundary_tokens:
            raise ValueError("Cannot calculate boundary probability if boundary tokens are not used. Try using -B.")

        self._check_n(ngram_length)
        boundary = position

        # Pad utterance with boundary tokens and shift boundary index accordingly
        phones = list(phones)
        if self.use_boundary_tokens:
            phones = [BOUNDARY_TOKEN] * (self.max_ngram - 1) + phones + [BOUNDARY_TOKEN] * (self.max_ngram - 1)
            boundary += self.max_ngram - 1

        if right:
            if not self.use_boundary_tokens and boundary + ngram_length > len(phones):
                return None
            right_context = phones[boundary:boundary+ngram_length]
            left_context = [phones[boundary-1]]
        else:
            if not self.use_boundary_tokens and boundary < ngram_length:
                return None
            if measure in ["tp", "mi"]:
                if not self.use_boundary_tokens and boundary >= len(phones):
                    return None
                else:
                    right_context = [phones[boundary]]
            left_context = phones[boundary-ngram_length:boundary]

        # Large boundary entropy = high unpredictability
        if measure == "ent" and right:
            return self._boundary_entropy_reverse(right_context)
        elif measure == "ent" and not right:
            return self._boundary_entropy(left_context)

        # Large transitional probability = low unpredictability (so return negative)
        elif measure == "tp" and right:
            return - self._conditional_probability(left_context, right_context)
        elif measure == "tp" and not right:
            return - self._conditional_probability_reverse(left_context, right_context)

        # Large successor variety indicates the end of a word (used like high unpredictability)
        elif measure == "sv" and right:
            return self._successor_variety_reverse(right_context)
        elif measure == "sv" and not right:
            return self._successor_variety(left_context)

        # Large mutual information = low unpredictability (so return negative)
        elif measure == "mi" and right:
            return - self._mutual_information(left_context, right_context)
        elif measure == "mi" and not right:
            return - self._mutual_information(left_context, right_context)

        elif measure == "bp" and right:
            return self._conditional_probability([BOUNDARY_TOKEN], right_context)
        elif measure =="bp" and not right:
            return self._conditional_probability_reverse(left_context, [BOUNDARY_TOKEN])

        else:
            raise ValueError("Unknown predictability measure: '{}'".format(measure))

    def get_log_word_probability(self, word, ngram_length):
        """ Returns the negative log probability of a word.

        The probability is calculated as the sum of the negative log conditional probabilities of each phone.
        If ngram_length=0, this assumes no context, so the probabilities are used instead of the conditional probabilities.

        E.g. for ngram_length = 1, P('abbc') = P('a'|boundary) * P('b'|'a') * P('b'|'b') * P('c'|'b') (but as log sum)
        For ngram_length = 0, P('abbc') = P('a') * P('b') * P('b') * P('c')

        Parameters
        ----------
        word : list of str
            A list of phones in the word.
        ngram_length : int
            The length of ngram context to use in the calculation.

        Returns
        -------
        probability : float
            The negative log probability of the word (non-negative).
        """

        prob = 0
        word = [BOUNDARY_TOKEN] * ngram_length + word
        for i in range(ngram_length, len(word)):
            if ngram_length == 0:
                p_phone = self.probability([word[i]])
            else:
                p_phone = self._conditional_probability_reverse(word[i-ngram_length:i], [word[i]])
            prob -= np.log2(p_phone) if p_phone > 0 else -10000
        return prob
