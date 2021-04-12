""" A mulitple-cue algorithm for word segmentation.

TODO: Add documentation

"""

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
    precision_weights : bool, optional
        If False, the weight of each cue is based on the accuracy of that cue. If True, a pair of weights
        is stored for each cue, based on the precision and recall of that cue. This allows the multicue model
        to learn which cues are very precise at placing boundaries and weigh votes from these models higher.
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

    def __init__(self, models=[], precision_weights=False, corpus_phonestats=None, lexicon_phonestats=None, lexicon=None, log=utils.null_logger()):
        super().__init__(log)

        # Check models
        if len(models) == 0:
            raise ValueError("Cannot initialise MultiCueModel without any sub models.")
        for model in models:
            if not isinstance(model, Model):
                raise ValueError("Object provided to MultiCueModel is not an instance of Model:"
                    + str(model))

        self.precision_weights = precision_weights

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
        if precision_weights:
            self.num_boundaries_not_placed = 0
            self.num_boundaries_placed = 0
            self.weights_positive = np.ones(self.num_models)
            self.weights_negative = np.ones(self.num_models)
            self.errors_positive = np.zeros(self.num_models)
            self.errors_negative = np.zeros(self.num_models)
        else:
            self.num_boundaries = 0
            self.weights = np.ones(self.num_models)
            self.errors = np.zeros(self.num_models)

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
        # We don't do majority voting for the first boundary (we assume all algorithms can
        # correctly place utterance boundaries)
        best_boundaries = [False] + [self._make_boundary_decision(boundary_votes, update_model)
                           for boundary_votes in boundaries.T[1:]]

        segmented = PhoneSequence(utterance.phones)
        segmented.boundaries = best_boundaries

        self._log.debug("Segmented utterance '{}' as '{}".format(utterance, segmented))

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

        if self.precision_weights:
            # Find the weighted vote for a boundary vs no boundary
            weighted_vote_for_boundary = votes_for_boundary.dot(self.weights_positive)
            weighted_vote_for_no_boundary = votes_for_no_boundary.dot(self.weights_negative)

            # Set boundary accordingly, setting no boundary for ties
            boundary = weighted_vote_for_boundary * sum(self.weights_negative) > weighted_vote_for_no_boundary * sum(self.weights_positive)

            # Update weights according to errors made by each model
            if update_model:
                if not boundary:
                    self.num_boundaries_not_placed += 1
                    self.errors_positive += votes_for_boundary
                    self.weights_positive = (1 - self.errors_positive / self.num_boundaries_not_placed)
                else:
                    self.num_boundaries_placed += 1
                    self.errors_negative += votes_for_no_boundary
                    self.weights_negative = (1 - self.errors_negative / self.num_boundaries_placed)

        else:
            # Find the weighted vote for a boundary and set boundary accordingly
            weighted_vote_for_boundary = votes_for_boundary.dot(self.weights)
            boundary = weighted_vote_for_boundary > 0.5 * sum(self.weights)

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
    if args.precision_weights:
        log.info('Using precision-based weights for weighted majority algorithm.')

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
    # models.append(ProbabilisticModel(ngram_length=0, model_type="venk", phonestats=corpus_phonestats, lexicon=lexicon, log=log))
    model = MultiCueModel(models=models, precision_weights=args.precision_weights, corpus_phonestats=corpus_phonestats, lexicon_phonestats=lexicon_phonestats, lexicon=lexicon, log=log)
    
    segmented = list(model.segment(text))

    log.info('Final weights:')
    if model.precision_weights:
        for m, weight_p, weight_n in zip(model.models, model.weights_positive, model.weights_negative):
            log.info('\t{}\t{}\t{}'.format(m, '%.4g' % weight_p, '%.4g' % weight_n))
    else:
        for m, weight in zip(model.models, model.weights):
            log.info('\t{}\t{}'.format(m, '%.4g' % weight))

    return segmented

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
    multi_options.add_argument(
        '-W', '--precision_weights', action='store_true',
        help='Use precision-based weights for each model rather than accuracy-based weights.')
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
