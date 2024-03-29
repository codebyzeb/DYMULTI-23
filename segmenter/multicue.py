""" A mulitple-cue algorithm for word segmentation.

"""

import numpy as np

from wordseg import utils

from segmenter.model import Model
from segmenter.lexicon import Lexicon
from segmenter.phonesequence import PhoneSequence
from segmenter.phonestats import PhoneStats
from segmenter.partialpeakindicators import PredictabilityIndicator, LexiconBoundaryIndicator, LexiconFrequencyIndicator, StressIndicator

class MultiCueModel(Model):
    """ Train and segment using multiple indicators as individual cues

    Parameters
    ----------
    indicators : list of Model, optional
        A list of Model objects used for segmentation, whose suggestions are combined 
        using weighted majority voting to produce a final segmentation.
    weight_type : str, optional
        If "accuracy", the weight of each cue is based on the accuracy of that cue. If "precision", a pair of weights
        is assigned for each cue, based on the precision and recall of that cue. This allows the multicue model
        to learn which cues are very precise at placing boundaries and weigh votes from these indicators higher. If "none",
        no weights are used.
    corpus_phonestats : PhoneStats, optional
        Phoneme statistics updated with each utterance from the corpus.
    lexicon_phonestats : PhoneStats, optional
        Phoneme statistics updated with each word in the segmented utterance
        (each segmented word is treated as an utterance).
    stressstats : PhoneStats, optional
        Stress statistics updated with each word in the segmented utterance
    lexicon : Lexicon, optional
        Lexicon updated with each word in the segmented utterance.
    log : logging.Logger, optional
        Where to send log messages

    Raises
    ------
    ValueError
        If any indicator in `indicators` is not an instance of Model or if `indicators` is empty.

    """

    def __init__(self, indicators=[], weight_type="accuracy", corpus_phonestats=None, lexicon_phonestats=None, stressstats=None, lexicon=None, log=utils.null_logger()):
        super().__init__(log)

        # Check indicators
        if len(indicators) == 0:
            raise ValueError("Cannot initialise MultiCueModel without any indicators.")
        for indicator in indicators:
            if not isinstance(indicator, Model):
                raise ValueError("Object provided to MultiCueModel is not an instance of Model:"
                    + str(indicator))

        self.weight_type = weight_type

        # Initialise phonestats and lexicon
        self.corpus_phonestats = corpus_phonestats
        self.lexicon_phonestats = lexicon_phonestats
        self.stressstats = stressstats
        self.lexicon = lexicon

        # Initialise indicators
        self.indicators = indicators
        self.num_indicators = len(indicators)
        self._log.info("Initialising MultiCueModel with {} models: \n{}".format(
            self.num_indicators, ", ".join([str(indicator) for indicator in indicators])))

        # Weights and error counts associated with each indicator
        if weight_type in ["precision", "recall", "f1"]:
            self.num_boundaries_not_placed = 1
            self.num_boundaries_placed = 1
            self.weights_positive = np.ones(self.num_indicators)
            self.weights_negative = np.ones(self.num_indicators)
            self.correct_positive = np.ones(self.num_indicators)
            self.correct_negative = np.ones(self.num_indicators)
            self.total_positive = np.ones(self.num_indicators)
            self.total_negative = np.ones(self.num_indicators)
        else:
            self.num_boundaries = 0
            self.weights = np.ones(self.num_indicators)
            self.errors = np.zeros(self.num_indicators)

    def __str__(self):
        return "MultiCue({})".format(", ".join([str(indicator) for indicator in self.indicators]))

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

        # Get suggested segmentations from each indicator
        segmentations = [indicator.segment_utterance(utterance, update_model) for indicator in self.indicators]
        boundaries = np.array([segmentation.boundaries for segmentation in segmentations])

        # Use weighted majority voting at each boundary position to find best segmentation
        # We don't do majority voting for the first boundary (we assume all algorithms can
        # correctly place utterance boundaries)
        best_boundaries = [False] + [self._make_boundary_decision(boundary_votes)[0] for boundary_votes in boundaries.T[1:]]

        segmented = PhoneSequence(utterance.phones, utterance.stress)
        segmented.boundaries = best_boundaries

        self._log.debug("Segmented utterance '{}' as '{}".format(utterance, segmented))

        if update_model:
            self.update(segmented, boundaries)

        return segmented

    def _make_boundary_decision(self, boundary_votes):
        """ Given votes cast by each indicator, determines whether a boundary should be placed.

        Uses the weighted majority algorithm to make a decision.

        Parameters
        ----------
        boundary_votes : array of bool
            An array of votes from each indicator where True represents a vote for a boundary, False otherwise.
        update_model : bool
            If true, updates the weights and error counts for each indicator according to the decision made.
        
        Returns
        -------
        boundary : bool
            A decision whether or not to place a boundary.
        weighted_vote_for_boundary : float
            The final vote for the boundary.
        weighted_vote_for_no_boundary : float
            The final vote for no boundary.
        """

        # Get each indicator's boundary decision
        votes_for_boundary = boundary_votes.astype(int)
        votes_for_no_boundary = np.ones(self.num_indicators) - votes_for_boundary

        if self.weight_type in ["precision", "recall", "f1"]:
            weighted_vote_for_boundary = votes_for_boundary.dot(self.weights_positive)
            weighted_vote_for_no_boundary = votes_for_no_boundary.dot(self.weights_negative)
            boundary = weighted_vote_for_boundary * sum(self.weights_negative) > weighted_vote_for_no_boundary * sum(self.weights_positive)
        else:
            weighted_vote_for_boundary = votes_for_boundary.dot(self.weights)
            weighted_vote_for_no_boundary = votes_for_no_boundary.dot(self.weights)
            boundary = weighted_vote_for_boundary > 0.5 * sum(self.weights)

        return boundary, weighted_vote_for_boundary, weighted_vote_for_no_boundary

    def update(self, segmented, boundaries):
        """ A method that is called at the end of segment_utterance. Updates 
        self.corpus_phonestats, self.lexicon_phonestats and self.lexicon using
        the segmented utterance. Also updates the weights of each indicator, based on the difference
        between their suggestions and the final segmentation.

        Parameters
        ----------
        segmented : PhoneSequence
            A sequence of phones representing the segmented utterance.
        boundaries : list of list of bool
            The boundaries suggested by each indicator.
        """
        if not self.corpus_phonestats is None:
            self.corpus_phonestats.add_phones(segmented.phones)
        if not self.lexicon_phonestats is None:
            for word in segmented.get_words():
                self.lexicon_phonestats.add_phones(word)
        if not self.stressstats is None:
            for word_stress in segmented.get_word_stress():
                self.stressstats.add_phones(word_stress)
        if not self.lexicon is None:
            for word in segmented.get_words():
                self.lexicon.increase_count(''.join(word))  

        # Don't update weights based on placement of the first utterance boundary
        for boundary_votes, true_boundary in zip(boundaries.T[1:], segmented.boundaries[1:]):
            votes_for_boundary = boundary_votes.astype(int)
            votes_for_no_boundary = np.ones(self.num_indicators) - votes_for_boundary
            
            if self.weight_type == "accuracy":
                self.num_boundaries += 1
                self.errors += votes_for_no_boundary if true_boundary else votes_for_boundary
                self.weights = (1 - self.errors / self.num_boundaries) 
            
            elif self.weight_type in ["precision", "recall", "f1"]:
                self.total_positive += votes_for_boundary
                self.total_negative += votes_for_no_boundary
                if true_boundary:
                    self.num_boundaries_placed += 1
                    self.correct_positive += votes_for_boundary
                else:
                    self.num_boundaries_not_placed += 1
                    self.correct_negative += votes_for_no_boundary
                precision_positive = self.correct_positive / self.total_positive
                precision_negative = self.correct_negative / self.total_negative
                recall_positive = self.correct_positive / self.num_boundaries_placed
                recall_negative = self.correct_negative / self.num_boundaries_not_placed
                if self.weight_type == "precision":
                    self.weights_positive = precision_positive
                    self.weights_negative = precision_negative
                elif self.weight_type == "recall":
                    self.weights_positive = recall_positive
                    self.weights_negative = recall_negative
                else:
                    self.weights_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
                    self.weights_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)                

def prepare_predictability_indicators(args, ngrams, phonestats, log):
    # For each measure and for each direction and for each ngram length up to max_ngram,
    # create a pair of peak-based predictability indicators (an increase indicator and a decrease indicator). 
    indicators = []
    for measure in args.predictability_indicators.split(','):
        log.info('Setting up Predictability Cues for measure: {}'.format(measure))
        for n in ngrams:
            if args.direction != "right":
                indicators.append(PredictabilityIndicator(ngram_length=n, increase=True, measure=measure, right=False, phonestats=phonestats, log=log))
                indicators.append(PredictabilityIndicator(ngram_length=n, increase=False, measure=measure, right=False, phonestats=phonestats, log=log))
            if args.direction != "left":
                indicators.append(PredictabilityIndicator(ngram_length=n, increase=True, measure=measure, right=True, phonestats=phonestats, log=log))
                indicators.append(PredictabilityIndicator(ngram_length=n, increase=False, measure=measure, right=True, phonestats=phonestats, log=log))
    return indicators

def prepare_lexicon_indicators(args, ngrams, phonestats, lexicon, log):
    # For each direction and for each ngram length up to max_ngram, create
    # a pair of lexicon indicators (an increase and a decrease indicator). 
    # Also create a pair of frequency lexicon indicators for each direction. 
    indicators = []
    if args.lexicon_indicators != "boundary":
        log.info('Setting up Lexicon Frequency Cues')
        if args.direction != "right":
            indicators.append(LexiconFrequencyIndicator(increase=True, use_presence=True, right=False, lexicon=lexicon, log=log))
            indicators.append(LexiconFrequencyIndicator(increase=False, use_presence=True, right=False, lexicon=lexicon, log=log))
        if args.direction != "left":
            indicators.append(LexiconFrequencyIndicator(increase=True, use_presence=True, right=True, lexicon=lexicon, log=log))
            indicators.append(LexiconFrequencyIndicator(increase=False, use_presence=True, right=True, lexicon=lexicon, log=log))
    if args.lexicon_indicators != "frequency":
        log.info('Setting up Lexicon Boundary Cues')
        for n in ngrams:
            if args.direction != "right":
                indicators.append(LexiconBoundaryIndicator(ngram_length=n, increase=True, right=False, lexicon=lexicon, phonestats=phonestats, log=log))
                indicators.append(LexiconBoundaryIndicator(ngram_length=n, increase=False, right=False, lexicon=lexicon, phonestats=phonestats, log=log))
            if args.direction != "left":
                indicators.append(LexiconBoundaryIndicator(ngram_length=n, increase=True, right=True, lexicon=lexicon, phonestats=phonestats, log=log))
                indicators.append(LexiconBoundaryIndicator(ngram_length=n, increase=False, right=True, lexicon=lexicon, phonestats=phonestats, log=log))
    return indicators

def prepare_stress_indicators(args, ngrams, stressstats, log):
    # For each direction and for each ngram length up to max_ngram,
    # create a pair of peak-based stress indicators (an increase indicator and a decrease indicator). 
    indicators = []
    log.info('Setting up Stress Cues')
    for n in ngrams:
        if args.direction != "right":
            indicators.append(StressIndicator(ngram_length=n, increase=True, right=False, stressstats=stressstats, log=log))
            indicators.append(StressIndicator(ngram_length=n, increase=False, right=False, stressstats=stressstats, log=log))
        if args.direction != "left":
            indicators.append(StressIndicator(ngram_length=n, increase=True, right=True, stressstats=stressstats, log=log))
            indicators.append(StressIndicator(ngram_length=n, increase=False, right=True, stressstats=stressstats, log=log))
    return indicators

def segment(text, args, log=utils.null_logger()):
    """ Segment using a Multi Cue segmenter model composed of a collection of indicators using a variety of cues. """

    log.info('Using a Multiple Cue model to segment text.')
    log.info('{} smoothing for probability estimates'.format("Using add-"+str(args.smoothing) if args.smoothing else "Not using"))
    log.info('Using "{}" weight type for weighted majority algorithm.'.format(args.weight_type))

    ngrams = [int(ngram) for ngram in args.ngrams.strip().split(',')]
    ngrams.sort()

    # Create multi-cue model
    corpus_phonestats = PhoneStats(max_ngram=ngrams[-1]+1, smoothing=args.smoothing, use_boundary_tokens=True)
    lexicon = Lexicon()
    lexicon_phonestats = PhoneStats(max_ngram=ngrams[-1]+1, smoothing=args.smoothing, use_boundary_tokens=True)
    stressstats = PhoneStats(max_ngram=ngrams[-1]+1, smoothing=args.smoothing, use_boundary_tokens=True)
    
    # Set up indicators for predictability, lexicon and stress
    indicators = []
    if args.predictability_indicators != "none":
        log.info('Setting up Predictability Indicators')
        indicators.extend(prepare_predictability_indicators(args, ngrams, corpus_phonestats, log))
    if args.lexicon_indicators != "none":
        log.info('Setting up Lexicon Indicators')
        indicators.extend(prepare_lexicon_indicators(args, ngrams, lexicon_phonestats, lexicon, log))
    if args.stress_file:
        log.info('Loading stress alignment information at {}'.format(args.stress_file))
        indicators.extend(prepare_stress_indicators(args, ngrams, stressstats, log))

    model = MultiCueModel(indicators=indicators, weight_type=args.weight_type, corpus_phonestats=corpus_phonestats, lexicon_phonestats=lexicon_phonestats, stressstats=stressstats, lexicon=lexicon, log=log)
    
    segmented = list(model.segment(text, stress_lines=list(open(args.stress_file, 'r')) if args.stress_file else None))

    log.info('Final weights:')
    if model.weight_type in ["precision", "recall", "f1"]:
        for m, weight_p, weight_n in zip(model.models, model.weights_positive, model.weights_negative):
            log.info('\t{}\t{}\t{}'.format(m, '%.4g' % weight_p, '%.4g' % weight_n))
    elif model.weight_type == "accuracy":
        for m, weight in zip(model.indicators, model.weights):
            log.info('\t{}\t{}'.format(m, '%.4g' % weight))
    else:
        log.info(' -- no weights used --')

    return segmented

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    multi_options = parser.add_argument_group('Multicue Model Options')
    multi_options.add_argument(
        '-n', '--ngrams', type=str, default=1, metavar='<str>',
        help='the ngram sizes used to use to calculate measures, '
        'default is %(default)s')
    multi_options.add_argument(
        '-d', '--direction', type=str, default="left", metavar='<str>',
        help='Select whether to use "left" context, "right" or "both" when creating indicators.'
        ' Default is %(default)s')
    multi_options.add_argument(
        '-S', '--smoothing', type=float, default=0.0, metavar='<float>',
        help='What value of k to use for add-k smoothing for probability calculations. Default is %(default)s')
    multi_options.add_argument(
        '-X', '--stress_file', type=str, metavar='<str>',
        help='If a stress alignment file is provided, stress indicators will be added to the model.')
    multi_options.add_argument(
        '-W', '--weight_type', type=str, default="accuracy", metavar='<str>',
        help='Type of weights to use for the majority vote algorithm. Select "precision", "recall", "f1" or '
        '"accuracy" for weights based on those measures and "none" for no weights. Default is %(default)s.')
    predictability_options = parser.add_argument_group('Predictability Indicator Options')
    predictability_options.add_argument(
        '-P', '--predictability_indicators', type=str, default="none", metavar='<str>',
        help='The measure of predictability to use. Select "ent" for Boundary Entropy, "tp" for Transitional Probability, '
        '"bp" for Boundary Probability, "mi" for Mutual Information and "sv" for Successor Variety or "none" for no predictability indicator. '
        'Can also select multiple measures using comma-separation. Default is %(default)s')
    lexicon_options = parser.add_argument_group('Lexicon Indicator Options')
    lexicon_options.add_argument(
        '-L', '--lexicon_indicators', type=str, default="none", metavar='<float>',
        help='Select which lexicon indicators to include. Select the "frequency" indicators, the '
        '"boundary" indicators, "both" or "none". Default is %(default)s')

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
