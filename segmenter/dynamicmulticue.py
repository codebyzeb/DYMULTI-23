""" A derived multiple-cue algorithm for segmentation that uses a Viterbi algorithm to select the best boundaries. """

import numpy as np

from wordseg import utils

from segmenter.phonestats import PhoneStats
from segmenter.lexicon import Lexicon
from segmenter.phonesequence import PhoneSequence
from segmenter.probabilistic import SYLLABIC_SOUNDS
from segmenter.multicue import MultiCueModel

class DynamicMultiCueModel(MultiCueModel):

    """ A derived multiple-cue statistical model for word segmentation, based on probabilistic models.

    Combining ideas from various models that find the most-likely probability for segmenting an utterance using the Viterbi
    algorithm, calculated by multiplying together the probability of each word in the segmentation. Instead of using word probability,
    the score of each word depends on the score of the boundaries to that word, calculated using a boundary-based weighted majority vote
    algorithm that uses information from several cues. 

    Parameters
    ----------
    models : list of Model, optional
        A list of Model objects used for segmentation, whose suggestions are combined 
        using weighted majority voting to produce a final segmentation.
    weight_type : str, optional
        If "accuracy", the weight of each cue is based on the accuracy of that cue. If "precision", a pair of weights
        is assigned for each cue, based on the precision and recall of that cue. This allows the multicue model
        to learn which cues are very precise at placing boundaries and weigh votes from these models higher. If "none",
        no weights are used.
    recognition_weight : float, optional
        The weight to assign previously-seen words. If set to 1, the model will always try to place boundaries around
        previously-seen words. If set to 0, no lexical recognition will take place.
    corpus_phonestats : PhoneStats, optional
        Phoneme statistics updated with each utterance from the corpus.
    lexicon_phonestats : PhoneStats, optional
        Phoneme statistics updated with each word in the segmented utterance
        (each segmented word is treated as an utterance).
    lexicon : Lexicon, optional
        Lexicon updated with each word in the segmented utterance.
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, models=[], weight_type="accuracy", recognition_weight=0, corpus_phonestats=None, lexicon_phonestats=None, stressstats=None, lexicon=None, log=utils.null_logger()):
        super().__init__(models=models, weight_type=weight_type, corpus_phonestats=corpus_phonestats, lexicon_phonestats=lexicon_phonestats, stressstats=stressstats, lexicon=lexicon, log=log)
        self.recognition_weight = recognition_weight

    def __str__(self):
        return "Dynamic"+super().__str__()

    # Overrides MultiCueModel.segment_utterance()
    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance combining weighted majority voting with a Viterbi algorithm.

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

        segmented = PhoneSequence(utterance.phones, utterance.stress)
        n = len(segmented)

        # Get suggested segmentations from each model, with the weighted vote for each boundary position
        segmentations = [model.segment_utterance(utterance) for model in self.models]
        boundaries = np.array([segmentation.boundaries for segmentation in segmentations])
        boundary_votes = [self._make_boundary_decision(boundary_votes)[1] for boundary_votes in boundaries.T]
        no_boundary_votes = [self._make_boundary_decision(boundary_votes)[2] for boundary_votes in boundaries.T]

        # Normalise votes around 0, such that votes greater than 0 represent where the MultiCueModel would have placed a boundary
        if self.weight_type in ["precision", "recall", "f1"]:
            normalised_votes = [boundary_vote/sum(self.weights_positive) - no_boundary_vote/sum(self.weights_negative)
                                for (boundary_vote, no_boundary_vote) in zip(boundary_votes, no_boundary_votes)]
        else:
            normalised_votes = [2 * (vote/sum(self.weights) - 0.5) for vote in boundary_votes]
        normalised_votes[0] = 1 # Always vote for utterance boundaries

        # Memoisation grids where best_scores[i] stores the best score for utterance[0:i]
        # and best_segpoints[i] scores the best place to split utterance[0:i] (best_segpoint[3] == 1 means the best way
        # to split utterance[0:3] is as utterance[0:1] | utterance[1:3].)
        best_scores = []
        best_segpoints = []

        # Get the best score for utterance[0:i]
        for i in range(0, n+1):

            # First consider no segmentation
            best_score = self.word_score(segmented.phones[:i], normalised_votes[:i])
            best_segpoint = i
            
            # Then consider each split of the sub-utterance to find best split,
            # making use of previously-stored sub-utterance scores.
            for j in range(1, i):
                # score of utterance[0:j] + score of word[j:i]
                score = best_scores[j] + self.word_score(segmented.phones[j:i], normalised_votes[j:i])
                if score > best_score:
                    best_score = score
                    best_segpoint = j
            
            #print('best score for utterance[0:{}] is to split as {},{}'.format(i, segmented.phones[0:best_segpoint], segmented.phones[best_segpoint:i]))
            best_scores.append(best_score)
            best_segpoints.append(best_segpoint)

        # Reconstruct best split.
        segpoint = n
        while segpoint != 0:
            # If best_segpoints[i] == i then the best way to segment utt[:i] is with no segmentation
            new_segpoint = best_segpoints[segpoint]
            if new_segpoint == segpoint:
                self._log.debug("Score for {} is {}".format(''.join(segmented.phones[0:segpoint]), pow(2, -self.word_score(segmented.phones[0:segpoint], normalised_votes[0:segpoint]))))
                break
            self._log.debug("Score for {} is {}".format(''.join(segmented.phones[new_segpoint:segpoint]), pow(2, -self.word_score(segmented.phones[new_segpoint:segpoint], normalised_votes[new_segpoint:segpoint]))))
            segmented.boundaries[new_segpoint] = True
            segpoint = new_segpoint

        self._log.debug("Segmented utterance '{}' as '{}".format(utterance, segmented))
        if update_model:
            self.update(segmented, boundaries)
        return segmented

    def update(self, segmented, boundaries):
        super().update(segmented, boundaries)

    def word_score(self, word, boundary_scores):
        """ Return a score for the word.

        Parameters
        ----------
        word : list of str
            A sequence of phones representing the word.
        boundary_scores : list of float
            A list of scores associated with each boundary position in the word.
        """

        if len(word) == 0:
            return 0

        # Score for familiar words
        word_str = ''.join(word)
        lexicon_score = self.recognition_weight if word_str in self.lexicon else 0

        # The score as just the boundary score at the start of the word
        boundary_score = boundary_scores[0]

        # Strongly reject words that don't contain a syllabic sound
        syllabic = sum([phoneme in word_str for phoneme in SYLLABIC_SOUNDS]) > 0 
        if not syllabic:
            return -100

        return (boundary_score + lexicon_score)

from segmenter.multicue import _add_arguments as _add_arguments_multicue, prepare_predictability_models, prepare_lexicon_models, prepare_stress_models

def _add_arguments(parser):
    _add_arguments_multicue(parser)
    dymulti_options = parser.add_argument_group('Dynamic Multicue Model Options')
    dymulti_options.add_argument(
        '-a', '--alpha', type=float, default=0.0, metavar='<float>',
        help='the weight to give previously-seen words, '
        'default is %(default)s')

def segment(text, args, log=utils.null_logger()):
    """ Segment using a Multi Cue segmenter model composed of a collection of models using a variety of cues. """

    log.info('Using a Dynamic Multiple Cue model to segment text.')
    log.info('{} smoothing for probability estimates'.format("Using add-"+str(args.smoothing) if args.smoothing else "Not using"))
    log.info('Using "{}" weight type for weighted majority algorithm.'.format(args.weight_type))
    log.info('Using lexical recognition with weight {}'.format(args.alpha) if args.alpha else 'Not using lexical recognition.')

    ngrams = [int(ngram) for ngram in args.ngrams.strip().split(',')]
    ngrams.sort()

    # Create multi-cue model
    corpus_phonestats = PhoneStats(max_ngram=ngrams[-1]+1, smoothing=args.smoothing, use_boundary_tokens=True)
    lexicon = Lexicon()
    lexicon_phonestats = PhoneStats(max_ngram=ngrams[-1]+1, smoothing=args.smoothing, use_boundary_tokens=True)
    stressstats = PhoneStats(max_ngram=ngrams[-1]+1, smoothing=args.smoothing, use_boundary_tokens=True)

    # Set up submodels for predictability, lexicon and stress
    models = []
    if args.predictability_models != "none":
        log.info('Setting up Predictability Models')
        models.extend(prepare_predictability_models(args, ngrams, corpus_phonestats, log))
    if args.lexicon_models != "none":
        log.info('Setting up Lexicon Models')
        models.extend(prepare_lexicon_models(args, ngrams, lexicon_phonestats, lexicon, log))
    if args.stress_file:
        log.info('Loading stress alignment information at {}'.format(args.stress_file))
        models.extend(prepare_stress_models(args, ngrams, stressstats, log))

    model = DynamicMultiCueModel(models=models, weight_type=args.weight_type, recognition_weight=args.alpha, corpus_phonestats=corpus_phonestats, lexicon_phonestats=lexicon_phonestats, stressstats=stressstats, lexicon=lexicon, log=log)
    
    segmented = list(model.segment(text))

    log.info('Final weights:')
    if model.weight_type in ["precision", "recall", "f1"]:
        for m, weight_p, weight_n in zip(model.models, model.weights_positive, model.weights_negative):
            log.info('\t{}\t{}\t{}'.format(m, '%.4g' % weight_p, '%.4g' % weight_n))
    elif model.weight_type == "accuracy":
        for m, weight in zip(model.models, model.weights):
            log.info('\t{}\t{}'.format(m, '%.4g' % weight))
    else:
        log.info(' -- no weights used --')

    return segmented

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-dynamicmulticue',
        description=__doc__,
        add_arguments=_add_arguments)

    segmented = segment(streamin, args, log=log)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()
