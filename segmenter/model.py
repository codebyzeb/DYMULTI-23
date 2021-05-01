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

    def segment(self, text, update_model=True, stress_lines=None):
        """ Segment some text using the model.
        
        Default implementation is to call segment_utterance for each utterance in the text,
        but subclasses can override this if additional implementation is needed.

        Parameters
        ----------
        text : seq of str
            A sequence of strings, each of which is considered an utterance and consists of space-separated phonemes.
        update_model : bool
            When True (default), updates the model online during segmentation.
        stress_lines: list of str
            A sequence of strings, each of which represents the stress aligned with the utterance.

        Yields
        ------
        utterances : generator
            The segmented utterances produced by the segmentation model.

        """

        t = time.time()

        for i, utterance in enumerate(text):
            if i % 1000 == 0:
                self._log.info("Utterances segmented: " + str(i))
            phones = utterance.strip().split(' ')
            if stress_lines:
                stress = list(''.join(stress_lines[i].strip().split(' ')))
                phone_sequence = PhoneSequence(phones=phones, stress=stress)
            else:
                phone_sequence = PhoneSequence(phones=phones)
            segmented = str(self.segment_utterance(phone_sequence, update_model))
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
