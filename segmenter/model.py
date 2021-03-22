""" Abstract class to represent a segmentation model that provides a segment() method """

import time

from abc import ABC, abstractmethod
from wordseg import utils

class Model(ABC):

    def __init__(self, log=utils.null_logger()):
        self._log = log

    def __str__(self):
        return "Abstract"

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
            segmented = ''.join(self.segment_utterance(utterance, update_model)).strip()
            yield segmented

        self._log.info("Total time to segment: " + str(time.time() - t))

    @abstractmethod
    def segment_utterance(self, utterance, update_model=True):
        """ All models should provide a method for segmenting a single utterance """
        pass
    
    def update(self, utterance, segmented):
        """ This method will be called at the end of segment_utterance. Child classes should
        implement this if they wish to update any internal state used in the score() function.
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

    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance by placing word boundaries at peaks of a particular score.

        Parameters
        ----------
        utterance : str
            An utterance consisting of space-separated phonemes.
        update_model : bool
            If True, will call update() to update any internal state of child classes.

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

        if update_model:
            self.update(utterance, segmented)
        
        return segmented

    @abstractmethod
    def score(self, utterance, position):
        """ Peak models should implement a method for calculating some score at a candidate
        boundary in the utterance. The boundary is considered to be after the phoneme at utterance[i].
        """
        pass


