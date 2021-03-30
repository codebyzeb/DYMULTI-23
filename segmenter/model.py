""" Abstract class to represent a segmentation model that provides a segment() method """

import time

from abc import ABC, abstractmethod
from wordseg import utils
from segmenter.phonesequence import PhoneSequence

class Model(ABC):

    def __init__(self, log=utils.null_logger()):
        self._log = log

    def __str__(self):
        return "AbstractModel"

    def segment(self, text, update_model=True):
        """ Segment some text using the model.
        
        Default implementation is to call segment_utterance for each utterance in the text,
        but subclasses can override this if additional implementation is needed.

        Parameters
        ----------
        text : seq of str
            A sequence of strings, each of which is considered an utterance and consists of space-separated phonemes.
        update_model : bool
            When True (default), updates the model online during segmentation.

        Yields
        ------
        utterances : generator
            The segmented utterances produced by the segmentation model.

        """

        t = time.time()

        for i, utterance in enumerate(text):
            if i % 100 == 0:
                self._log.info("Utterances segmented: " + str(i))
            segmented = str(self.segment_utterance(PhoneSequence(utterance.strip().split(' ')), update_model))
            yield segmented

        self._log.info("Total time to segment: " + str(time.time() - t))

    @abstractmethod
    def segment_utterance(self, utterance, update_model=True):
        """ All models should provide a method for segmenting a single utterance

        Parameters
        ----------
        utterance : PhoneSequence
            A sequence of phones representing the utterance.
        update_model : bool
            When True (default), updates the model online during segmentation.
        """
        pass

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
