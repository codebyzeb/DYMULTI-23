""" A predictabilty cue algorithm for word segmentation.

TODO: Add documentation

"""

from wordseg import utils

from segmenter.phonestats import PhoneStats
from segmenter.model import PeakModel

PREDICTABILITY_MEASURES = ["mi", "tp", "ent", "sv", "bp"]

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
        Returns a score for the candidate boundary after utterance.phones[i], calculated
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
