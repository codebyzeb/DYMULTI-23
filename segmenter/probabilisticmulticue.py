""" A set of probabilistic, lexicon-building algorithms for word segmentation """

from numpy import log2

from wordseg import utils

from segmenter.model import Model
from segmenter.lexicon import Lexicon
from segmenter.phonestats import PhoneStats
from segmenter.phonesequence import PhoneSequence
from segmenter.probabilistic import ProbabilisticModel, BR_CORPUS_SYLLABIC_SOUNDS

class ProbabilisticMultiCueModel(ProbabilisticModel):

    """ A derived multiple-cue statistical model for word segmentation, based on probabilistic models.

    Combining ideas from various models that find the most-likely probability for segmenting an utterance using the Viterbi
    algorithm, calculated by multiplying together the probability of each word in the segmentation. Also considers boundary-based
    statistical models that combine information from many cues.

    Parameters
    ----------
    ngram_length : int, optional
        The length of n-grams used in the phoneme language model, used to initialise PhoneStats if not provided.
    phonestats : PhoneStats, optional
        Phoneme statistics updated with each utterance from the corpus. If not provided, a PhoneStats object
        will be created and updated by this model.
    lexicon : Lexicon, optional
        If not provided, a Lexicon object will be created and updated to keep track of previously-seen
        words. 
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, ngram_length=1, phonestats=None, lexicon=None, log=utils.null_logger()):
        super().__init__(ngram_length=ngram_length, phonestats=phonestats, lexicon=lexicon, log=log)
        
    def __str__(self):
        return "ProbabilisticMultiCueModel(N: {})".format(self.ngram_length)

    def word_score(self, word):
        """ Get P(word) 

        Parameters
        ----------
        word : list of str
            A sequence of phones representing the word.
        """

        scores = []

        for n in range(1, self.ngram_length+1):

            word_final = word[-min(len(word), n):]
            word_start = word[:min(len(word), n)]

            #sv_score = max(1, self._phonestats._successor_variety(word_final)) / 52
            #pv_score = max(1, self._phonestats._successor_variety_reverse(word_start)) / 52

            bp_score_forward = self._phonestats._boundary_probability(word_final)
            bp_score_backward = self._phonestats._boundary_probability_reverse(word_start)
        
            scores.extend([bp_score_backward, bp_score_forward])
        
        syllabic_score = sum([phoneme in BR_CORPUS_SYLLABIC_SOUNDS for phoneme in word]) > 0

        #lexicon_score = self._lexicon.relative_freq(''.join(word), consider_unseen=True) if ''.join(word) in self._lexicon else self._lexicon.unseen_freq()
        lexicon_score = ''.join(word) in self._lexicon
        scores.append(lexicon_score * 10)

        word_score = self._phonestats.get_log_word_probability(word, 0)

        prob = -log2(sum(scores) * syllabic_score) #* pow(2, -word_score)
        return prob

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    group = parser.add_argument_group('probabilistic multicue model options')
    group.add_argument(
        '-n', '--ngram_context', type=int, default=0, metavar='<int>',
        help='the context size used for calculating unseen word probabilities. '
        'default is %(default)s, indicating no context used (unigram assumption).')

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-probabilistic-multicue',
        description=__doc__,
        add_arguments=_add_arguments)

    # Create model
    model = ProbabilisticMultiCueModel(ngram_length=args.ngram_context, log=log)

    # Segment text
    segmented = model.segment(streamin)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()