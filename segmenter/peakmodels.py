""" A set of partial-peak models for segmentation.

TODO: Add documentation

"""

from wordseg import utils
from abc import ABC, abstractmethod

from segmenter.model import Model
from segmenter.lexicon import Lexicon
from segmenter.phonesequence import PhoneSequence
from segmenter.phonestats import PhoneStats

PREDICTABILITY_MEASURES = ["mi", "tp", "ent", "sv", "bp"]

class PeakModel(Model):
    """ An abstract model for segmentation that places boundaries at peaks of a particular score.

    Child classes should implement the score() method, used to calculate the score at a position
    in the utterance.

    Parameters
    ----------
    increase : bool, optional
        When true, place word boundaries when the score increases at a candidate boundary.
        When false, place word boundaries when the score decreases after a candidate boundary.
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, increase=True, log=utils.null_logger()):
        super().__init__(log)

        # Initialise model parameters
        self.increase = increase

    def __str__(self):
        return "PeakModel({})".format("Increase" if self.increase else "Decrease")

    # Overrides Model.segment_utterance()
    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance by placing word boundaries at peaks of a particular score.

        Parameters
        ----------
        utterance : PhoneSequence
            A sequence of phones representing the utterance.
        update_model : bool
            If True, will call update() to update any internal state of child classes.

        Returns
        -------
        segmented : PhoneSequence
            A sequence of phones with word boundaries.
        """

        segmented = PhoneSequence(utterance.phones)

        last_score = None
        
        # i = 0 is the utterance boundary (the candidate position before phoneme 0)
        for i in range(len(utterance)+1):
            score = self.score(utterance, i)

            # Either place word boundaries when the score increases at candidate boundary
            # or when the score decreases after a candidate boundary
            if not score is None and not last_score is None:
                if self.increase and score > last_score and i < len(utterance):
                    segmented.boundaries[i] = True
                elif not self.increase and score < last_score:
                    segmented.boundaries[i-1] = True

            last_score = score

        if update_model:
            self.update(segmented)
        
        return segmented

    @abstractmethod
    def score(self, utterance, position):
        """ Peak models should implement a method for calculating some score at a candidate
        boundary in the utterance. The boundary is considered to be after the phoneme at utterance.phones[position].
        """
        pass
    
    @abstractmethod
    def update(self, segmented):
        """ A method that is called at the end of segment_utterance. Child classes should
        implement this if they wish to update any internal state based on the segmentation.

        Parameters
        ----------
        segmented : PhoneSequence
            A sequence of phones representing the segmented utterance.
        """
        pass

class PredictabilityModel(PeakModel):
    """ Train and segment using measures of predictability. 

    Only provides a single cue, should be used in combination with other ngram lengths and
    directions of change to provide better segmentation.

    Parameters
    ----------
    ngram_length : int, optional
        The length of n-grams used to calculate predictability.
    increase : bool, optional
        When true, place word boundaries when the unpredictability increases before a candidate boundary.
        When false, place word boundaries when the unpredictability decreases after a candidate boundary.
    measure : str, optional
        Which predictability measure to use. The default is "ent" for Boundary Entropy.
    right : bool, optional
        When True, use the right context for predictability calculations, otherwise use the left context.
    phonestats : phonestats.PhoneStats, optional
        If not provided, a PhoneStats object will be created and updated to keep track of phoneme counts.
        Otherwise, the provided object will be used, but not updated.
    log : logging.Logger, optional
        Where to send log messages

    Raises
    ------
    ValueError
        If `ngram` is less than 1 or `measure` is not a valid predictability measure.

    """

    def __init__(self, ngram_length=1, increase=True, measure="ent",
                right=False,  phonestats=None, log=utils.null_logger()):
        super().__init__(increase, log)

        # Initialise model parameters
        if ngram_length < 1:
            raise ValueError("Cannot initialise a Predictability Model with non-positive n-gram length.")
        if not measure in PREDICTABILITY_MEASURES:
            raise ValueError("Cannot initialise a Predictability Model with unknown predictability measure '{}'".format(measure))
        self.ngram_length = ngram_length
        self.measure = measure
        self.right = right

        # Initialise phoneme statistics if not provided
        # (usually need to store ngrams one larger than those queried for calculations)
        if phonestats is None:
            self._phonestats = PhoneStats(ngram_length+1)
            self._updatephonestats = True
        else:
            self._phonestats = phonestats
            self._updatephonestats = False

    def __str__(self):
        return "Predictability({},{}{}{})".format(
            "N: " + str(self.ngram_length),
            "Increase of " if self.increase else "Decrease of ",
            "Right Context " if self.right else "Left Context ",
            "Boundary Entropy" if self.measure == "ent" else 
            "Mutual Information" if self.measure == "mi" else
            "Successor Variety" if self.measure == "sv" else
            "Boundary Probability" if self.measure == "bp" else
            "Transitional Probability" if self.measure == "tp" else self.measure)

    # Overrides PeakModel.update()
    def score(self, utterance, position):
        """
        Returns a score for the candidate boundary before utterance.phones[i], calculated
        using some measure of (un)predictability at the boundary.
        """
        return self._phonestats.get_unpredictability(
            phones=utterance.phones, position=position, measure=self.measure,
            right=self.right, ngram_length=self.ngram_length)

    # Overrides PeakModel.update()
    def update(self, segmented):
        """ Update phonestats with phones in utterance. """
        if self._updatephonestats:
            self._phonestats.add_phones(segmented.phones)

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
        Returns a score for the candidate boundary before utterance.phones[i], calculated
        according to the words in the lexicon found in the utterance that start or end at that boundary.
        """

        if self.right:
            candidate_words = [''.join(utterance.phones[position:j]) for j in range(position+1, len(utterance)+1)]
        else:
            candidate_words = [''.join(utterance.phones[j:position]) for j in range(0, position)]

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
