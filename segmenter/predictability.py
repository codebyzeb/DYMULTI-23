""" A predictabilty cue algorithm for word segmentation.

TODO: Add documentation

"""

import collections
import random
import numpy as np

from wordseg import utils

from segmenter.model import Model
from segmenter.phonestats import PhoneStats
from segmenter.multicue import MultiCueModel

PREDICTABILITY_MEASURES = ["mi", "tp", "ent", "sv"]

class PredictabilityModel(Model):
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
    reverse : bool, optional
        When True, the reverse predictability measure is used.
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
                reverse=False,  phonestats=None, log=utils.null_logger()):
        super().__init__(log)

        # Initialise model parameters
        if ngram_length < 1:
            raise ValueError("Cannot initialise a Predictability Model with non-positive n-gram length.")
        if not measure in PREDICTABILITY_MEASURES:
            raise ValueError("Cannot initialise a Predictability Model with unknown predictability measure '{}'".format(measure))
        self.ngram_length = ngram_length
        self.increase = increase
        self.measure = measure
        self.reverse = reverse

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
            "Reverse " if self.reverse else "",
            "Boundary Entropy" if self.measure == "ent" else 
            "Mutual Information" if self.measure == "mi" else
            "Successor Variety" if self.measure == "sv" else
            "Transitional Probability" if self.measure == "tp" else self.measure)

    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance using weighted majority voting.

        Parameters
        ----------
        utterance : str
            An utterance consisting of space-separated phonemes.
        update_model : bool
            Updates the model's phoneme counts if it controls its own PhoneStats object (if it wasn't provided
            one when initialised, it controls its own PhoneStats object, otherwise it does not).

        Returns
        -------
        segmented : str
            The segmented utterance as a string of phonemes with spaces at word boundaries.
        """

        segmented = ""
        utterance = utterance.strip().split(' ')

        last_unpredictability = None
        
        for i in range(len(utterance)):
            unpredictability = self._phonestats.get_unpredictability(utterance, i, self.measure, self.reverse, self.ngram_length)

            if unpredictability is None or last_unpredictability is None:
                segmented += utterance[i]
                last_unpredictability = unpredictability
                continue

            # Either place word boundaries when the unpredictability increases before a candidate boundary
            # or when unpredictability decreases after a candidate boundary
            if self.increase:
                segmented += utterance[i] + (' ' if unpredictability > last_unpredictability else '')
            else:
                segmented += (' ' if unpredictability < last_unpredictability else '') + utterance[i]

            last_unpredictability = unpredictability

        # Update ngram counts in phonestats object to refine unpredictability calculations
        if self._updatephonestats and update_model:
            self._phonestats.add_utterance(utterance)
        
        return segmented  

class MultiPredictabilityModel(MultiCueModel):
    """ Train and segment using measures of predictability with multiple models of varying lengths.

    Uses a Multiple Cue model for majority voting of each weak Predictability model, creating a 
    For each ngram length up to the maximum, it creates either 2 Predictability models (one for 
    an increase in unpredictability before a boundary, one for a decrease in unpredictability after 
    a boundary) or 4 (if "Both" is selected as the direction two models will be created for the 
    "Forwards" and the "Reverse" predictability calculation).

    Parameters
    ----------
    ngram_length : int, optional
        The maximum length of n-grams used to calculate predictability. A model is created for
        each ngram length from 1 to this value.
    measure : str, optional
        Which predictability measure to use. The default is "ent" for Boundary Entropy.
    direction : str, optional
        When "Forwards", only the forwards predictability measure is used.
        When "Reverse", only the reverse predictability measure is used. Otherwise,
        uses both measures.
    log : logging.Logger, optional
        Where to send log messages

    Raises
    ------
    ValueError
        If `ngram` is less than 1 or `measure` is not a valid predictability measure.

    """

    def __init__(self, max_ngram=1, measure="ent", direction="forwards", phonestats=None, log=utils.null_logger()):
        
        # Initialise model parameters
        if max_ngram < 1:
            raise ValueError("Cannot initialise a Multi Predictability Model with non-positive n-gram length.")
        if not measure in PREDICTABILITY_MEASURES:
            raise ValueError("Cannot initialise a Multi Predictability Model with unknown predictability measure '{}'".format(measure))
        
        self.max_ngram = max_ngram
        self.measure = measure
        self._phonestats = PhoneStats(max_ngram+1) if phonestats is None else phonestats

        # Create models
        models = []
        for n in range(1, max_ngram+1):
            if direction != "reverse":
                models.append(PredictabilityModel(ngram_length=n, increase=True, measure=measure, reverse=False, phonestats=self._phonestats, log=log))
                models.append(PredictabilityModel(ngram_length=n, increase=False, measure=measure, reverse=False, phonestats=self._phonestats, log=log))
            if direction != "forwards":
                models.append(PredictabilityModel(ngram_length=n, increase=True, measure=measure, reverse=True, phonestats=self._phonestats, log=log))
                models.append(PredictabilityModel(ngram_length=n, increase=False, measure=measure, reverse=True, phonestats=self._phonestats, log=log))

        # Give all predictability models to multicue model
        super().__init__(models=models, log=log)

    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance using weighted majority voting.

        Parameters
        ----------
        utterance : str
            An utterance consisting of space-separated phonemes.

        update_model : bool
            When True (default), updates the model online during segmentation.

        Returns
        ------
        segmented : str
            The segmented utterance as a string of phonemes with spaces at word boundaries.
        """

        # Call MultiCue model for weighted majority voting
        segmented = super().segment_utterance(utterance, update_model)

        # Update phoneme ngram counts using the utterance
        if update_model:
            self._phonestats.add_utterance(utterance.strip().split(' '))

        return segmented

    def __str__(self):
        return "MultiPredictability({})".format(", ".join([str(model) for model in self.models]))

def segment(text, max_ngram=1, measure="ent", direction="forwards", smoothing=0, use_boundary_tokens=False, log=utils.null_logger()):
    """ Segment using a Multi Cue segmenter model composed of a collection of Predictability models. """

    log.info('Using a Multi Predictability model to segment text.')

    # TODO: Check the input is valid
    log.info('{} smoothing for probability estimates'.format("Using add-"+str(smoothing) if smoothing else "Not using"))
    log.info('{} utterance boundary padding for ngram estimages'.format("Using" if use_boundary_tokens else "Not using"))

    phonestats = PhoneStats(max_ngram=max_ngram+1, smoothing=smoothing, use_boundary_tokens=use_boundary_tokens)
    model = MultiPredictabilityModel(max_ngram=max_ngram, measure=measure, direction=direction, phonestats=phonestats, log=log)
    
    return model.segment(text)

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    group = parser.add_argument_group('Multi Predictability Model Options')
    group.add_argument(
        '-n', '--max_ngram', type=int, default=1, metavar='<int>',
        help='the maximum length of ngram to use to calculate predictability, '
        'default is %(default)s')
    group.add_argument(
        '-m', '--measure', type=str, default="ent", metavar='<str>',
        help='the measure of predictability to use, select "ent" for Boundary Entropy. '
        'default is %(default)s')
    group.add_argument(
        '-d', '--direction', type=str, default="forwards", metavar='<str>',
        help='Select whether to use "forwards" predictability calculation, "backwards" or "both". '
        'default is %(default)s')
    group.add_argument(
        '-S', '--smooth', type=float, default=0.0, metavar='<float>',
        help='What value of k to use for add-k smoothing for probability calculations.')
    group.add_argument(
        '-B', '--useboundary', action='store_true',
        help='Whether to use boundary tokens for ngram estimates.')

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-multipredictability',
        description=__doc__,
        add_arguments=_add_arguments)

    segmented = segment(streamin, max_ngram=args.max_ngram, measure=args.measure, smoothing=args.smooth,
                        use_boundary_tokens=args.useboundary, direction=args.direction, log=log)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()
