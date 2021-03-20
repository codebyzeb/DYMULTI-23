""" An abstract model for segmentation algorithms that use peak-based segmentation strategies.

TODO: Add documentation

"""

from wordseg import utils
from abc import abstractmethod

from segmenter.model import Model

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

    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance by placing word boundaries at peaks of a particular score..

        Parameters
        ----------
        utterance : str
            An utterance consisting of space-separated phonemes.
        update_model : bool
            Unused by this method, inherited from Model.

        Returns
        -------
        segmented : list of str
            The segmented utterance as a list of phonemes and spaces for word boundaries.
        """

        segmented = []
        utterance = utterance.strip().split(' ')

        last_score = None
        
        # -1 used to get the score at the first utterance boundary
        for i in range(-1, len(utterance)):
            score = self.score(utterance, i)

            if i == -1:
                last_score = score
                continue
            if score is None or last_score is None:
                segmented += utterance[i]
                last_score = score
                continue

            # Either place word boundaries when the score increases at candidate boundary
            # or when the score decreases after a candidate boundary
            if self.increase:
                segmented.append(utterance[i])
                if score > last_score:
                    segmented.append(' ')
            else:
                if score < last_score and i > 0:
                    segmented.append(' ')
                segmented.append(utterance[i])

            last_score = score
        
        return segmented

    @abstractmethod
    def score(self, utterance, position):
        """ Peak models should implement a method for calculating some score at a candidate
        boundary in the utterance. The boundary is considered to be after the phoneme at utterance[i].
        """
        pass
