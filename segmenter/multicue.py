""" A mulitple-cue algorithm for word segmentation.

TODO: Add documentation

"""

import random
import numpy as np

from wordseg import utils

from segmenter.model import Model
from segmenter.lexicon import Lexicon, LexiconBoundaryModel, LexiconFrequencyModel
from segmenter.phonesequence import PhoneSequence
from segmenter.phonestats import PhoneStats
from segmenter.predictability import PredictabilityModel

class MultiCueModel(Model):
    """ Train and segment using multiple models as individual cues

    Parameters
    ----------
    models : list of Model, optional
        A list of Model objects used for segmentation, whose suggestions are combined 
        using weighted majority voting to produce a final segmentation.
    corpus_phonestats : PhoneStats, optional
        Phoneme statistics updated with each utterance from the corpus.
    lexicon_phonestats : PhoneStats, optional
        Phoneme statistics updated with each word in the segmented utterance
        (each segmented word is treated as an utterance).
    lexicon : Lexicon, optional
        Lexicon updated with each word in the segmented utterance.
    log : logging.Logger, optional
        Where to send log messages

    Raises
    ------
    ValueError
        If any model in `models` is not an instance of Model or if `models` is empty.

    """

    def __init__(self, models=[], corpus_phonestats=None, lexicon_phonestats=None, lexicon=None, log=utils.null_logger()):
        super().__init__(log)

        # Check models
        if len(models) == 0:
            raise ValueError("Cannot initialise MultiCueModel without any sub models.")
        for model in models:
            if not isinstance(model, Model):
                raise ValueError("Object provided to MultiCueModel is not an instance of Model:"
                    + str(model))

        # Initialise phonestats and lexicon
        self.corpus_phonestats = corpus_phonestats
        self.lexicon_phonestats = lexicon_phonestats
        self.lexicon = lexicon

        # Initialise models
        self.models = models
        self.num_models = len(models)
        self._log.info("Initialising MultiCueModel with {} models: \n{}".format(
            self.num_models, ", ".join([str(model) for model in models])))

        # Weights and error counts associated with each model
        self.weights = np.ones(self.num_models)
        self.errors = np.zeros(self.num_models)

        # Total number of boundaries seen (used to calculate errors)
        self.num_boundaries = 0

    def __str__(self):
        return "MultiCue({})".format(", ".join([str(model) for model in self.models]))

    # Overrides Model.segment_utterance()
    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance using weighted majority voting.

        Parameters
        ----------
        utterance : PhoneSequence
            A sequence of phones representing the utterance.
        update_model : bool
            When True (default), updates the model online during segmentation.

        Returns
        ------
        segmented : list of str
            The segmented utterance as a list of phonemes and spaces for word boundaries.
        """

        # Get suggested segmentations from each model
        segmentations = [model.segment_utterance(utterance, update_model) for model in self.models]
        boundaries = np.array([segmentation.boundaries for segmentation in segmentations])

        # Use weighted majority voting at each boundary position to find best segmentation
        # We don't do majority voting for the last boundary (we assume all algorithms can
        # correctly place utterance boundaries)
        best_boundaries = [self._make_boundary_decision(boundary_votes, update_model)
                           for boundary_votes in boundaries.T[:-1]]

        # Appending utterance boundary (not a word boundary)
        best_boundaries.append(False)

        segmented = PhoneSequence(utterance.phones)
        segmented.boundaries = best_boundaries

        self._log.debug("Segmented utterance '{}' as '{}".format(utterance, segmented))
        self._log.debug("Current errors: {} out of {}".format(self.errors, self.num_boundaries))
        self._log.debug("Current weights: {}".format(self.weights))

        if update_model:
            self.update(segmented)

        return segmented

    def _make_boundary_decision(self, boundary_votes, update_model):
        """ Given votes cast by each model, determines whether a boundary should be placed.

        Uses the weighted majority algorithm to make a decision.

        Parameters
        ----------
        boundary_votes : array of bool
            An array of votes from each model where True represents a vote for a boundary, False otherwise.
        update_model : bool
            If true, updates the weights and error counts for each model according to the decision made.
        
        Returns
        -------
        boundary : bool
            A decision whether or not to place a boundary.
        """

        # Get each model's boundary decision
        votes_for_boundary = boundary_votes.astype(int)
        votes_for_no_boundary = np.ones(self.num_models) - votes_for_boundary

        # Find the weighted vote for a boundary vs no boundary
        weighted_vote_for_boundary = votes_for_boundary.dot(self.weights)
        weighted_vote_for_no_boundary = votes_for_no_boundary.dot(self.weights)

        # Set boundary accordingly, setting no boundary for ties
        boundary = weighted_vote_for_boundary > weighted_vote_for_no_boundary

        # Update weights according to errors made by each model
        if update_model:
            self.num_boundaries += 1
            self.errors += votes_for_no_boundary if boundary else votes_for_boundary
            self.weights = 2 * (0.5 - self.errors / self.num_boundaries)

        return boundary

    def update(self, segmented):
        """ A method that is called at the end of segment_utterance. Updates 
        self.corpus_phonestats, self.lexicon_phonestats and self.lexicon using
        the segmented utterance.

        Parameters
        ----------
        segmented : PhoneSequence
            A sequence of phones representing the segmented utterance.
        """
        if not self.corpus_phonestats is None:
            self.corpus_phonestats.add_phones(segmented.phones)
        if not self.lexicon_phonestats is None:
            for word in segmented.get_words():
                self.lexicon_phonestats.add_phones(word)
        if not self.lexicon is None:
            for word in segmented.get_words():
                self.lexicon.increase_count(''.join(word))   

def prepare_predictability_models(args, phonestats, log):
    # For each measure and for each direction and for each ngram length up to max_ngram,
    # create a pair of peak-based predictability models (an increase model and a decrease model). 
    models = []
    for measure in args.predictability_models.split(','):
        log.info('Setting up Predictability Cues for measure: {}'.format(measure))
        for n in range(1, args.max_ngram+1):
            if args.direction != "right":
                models.append(PredictabilityModel(ngram_length=n, increase=True, measure=measure, right=False, phonestats=phonestats, log=log))
                models.append(PredictabilityModel(ngram_length=n, increase=False, measure=measure, right=False, phonestats=phonestats, log=log))
            if args.direction != "left":
                models.append(PredictabilityModel(ngram_length=n, increase=True, measure=measure, right=True, phonestats=phonestats, log=log))
                models.append(PredictabilityModel(ngram_length=n, increase=False, measure=measure, right=True, phonestats=phonestats, log=log))
    return models

def prepare_lexicon_models(args, phonestats, lexicon, log):
    # For each direction and for each ngram length up to max_ngram, create
    # a pair of lexicon models (an increase and a decrease model). 
    # Also create a pair of frequency lexicon models for each direction. 
    models = []
    if args.lexicon_models != "boundary":
        log.info('Setting up Lexicon Frequency Cues')
        if args.direction != "right":
            models.append(LexiconFrequencyModel(increase=True, use_presence=True, right=False, lexicon=lexicon, log=log))
            models.append(LexiconFrequencyModel(increase=False, use_presence=True, right=False, lexicon=lexicon, log=log))
        if args.direction != "left":
            models.append(LexiconFrequencyModel(increase=True, use_presence=True, right=True, lexicon=lexicon, log=log))
            models.append(LexiconFrequencyModel(increase=False, use_presence=True, right=True, lexicon=lexicon, log=log))
    if args.lexicon_models != "frequency":
        log.info('Setting up Lexicon Boundary Cues')
        for n in range(1, args.max_ngram+1):
            if args.direction != "right":
                models.append(LexiconBoundaryModel(ngram_length=n, increase=True, right=False, lexicon=lexicon, phonestats=phonestats, log=log))
                models.append(LexiconBoundaryModel(ngram_length=n, increase=False, right=False, lexicon=lexicon, phonestats=phonestats, log=log))
            if args.direction != "left":
                models.append(LexiconBoundaryModel(ngram_length=n, increase=True, right=True, lexicon=lexicon, phonestats=phonestats, log=log))
                models.append(LexiconBoundaryModel(ngram_length=n, increase=False, right=True, lexicon=lexicon, phonestats=phonestats, log=log))
    return models

def segment(text, args, log=utils.null_logger()):
    """ Segment using a Multi Cue segmenter model composed of a collection of models using a variety of cues. """

    log.info('Using a Multiple Cue model to segment text.')
    log.info('{} smoothing for probability estimates'.format("Using add-"+str(args.smoothing) if args.smoothing else "Not using"))

    # Create multi-cue model
    corpus_phonestats = PhoneStats(max_ngram=args.max_ngram+1, smoothing=args.smoothing, use_boundary_tokens=True)
    lexicon = Lexicon()
    lexicon_phonestats = PhoneStats(max_ngram=args.max_ngram+1, smoothing=args.smoothing, use_boundary_tokens=True)
    models = []
    if args.predictability_models != "none":
        log.info('Setting up Predictability Models')
        models.extend(prepare_predictability_models(args, corpus_phonestats, log))
    if args.lexicon_models != "none":
        log.info('Setting up Lexicon Models')
        models.extend(prepare_lexicon_models(args, lexicon_phonestats, lexicon, log))
    model = MultiCueModel(models=models, corpus_phonestats=corpus_phonestats, lexicon_phonestats=lexicon_phonestats, lexicon=lexicon, log=log)
    
    return model.segment(text)

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    multi_options = parser.add_argument_group('Multicue Model Options')
    multi_options.add_argument(
        '-n', '--max_ngram', type=int, default=1, metavar='<int>',
        help='the maximum length of ngram to use to calculate predictability, '
        'default is %(default)s')
    multi_options.add_argument(
        '-d', '--direction', type=str, default="left", metavar='<str>',
        help='Select whether to use "left" context, "right" or "both" when creating models.'
        ' Default is %(default)s')
    multi_options.add_argument(
        '-S', '--smoothing', type=float, default=0.0, metavar='<float>',
        help='What value of k to use for add-k smoothing for probability calculations. Default is %(default)s')
    predictability_options = parser.add_argument_group('Predictability Model Options')
    predictability_options.add_argument(
        '-P', '--predictability_models', type=str, default="none", metavar='<str>',
        help='The measure of predictability to use. Select "ent" for Boundary Entropy, "tp" for Transitional Probability, '
        '"bp" for Boundary Probability, "mi" for Mutual Information and "sv" for Successor Variety or "none" for no predictability model. '
        'Can also select multiple measures using comma-separation. Default is %(default)s')
    lexicon_options = parser.add_argument_group('Lexicon Model Options')
    lexicon_options.add_argument(
        '-L', '--lexicon_models', type=str, default="none", metavar='<float>',
        help='Select which lexicon models to include. Select the "frequency" model, the '
        '"boundary" model, "both" or "none". Default is %(default)s')

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-multicue',
        description=__doc__,
        add_arguments=_add_arguments)

    segmented = segment(streamin, args, log=log)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()
