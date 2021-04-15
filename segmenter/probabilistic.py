""" A set of probabilistic, lexicon-building algorithms for word segmentation """

from numpy import log2

from wordseg import utils

from segmenter.model import Model
from segmenter.lexicon import Lexicon
from segmenter.phonestats import PhoneStats
from segmenter.phonesequence import PhoneSequence

MODEL_TYPES = ["lm", "venk", "blanch"]
BR_CORPUS_SYLLABIC_SOUNDS = ['I', 'E', '&', 'A', 'a', 'O', 'U', '6', 'i', 'e', 'u', 'o', '9', 'Q', '7', '3', 'R', '#', '%', '*', '(', ')', 'L', '~']

class ProbabilisticModel(Model):

    """ A variety of lexicon-based probabilistic models for word segmentation.

    Based various models that find the most-likely probability for segmenting an utterance using the Viterbi
    algorithm, calculated by multiplying together the probability of each word in the segmentation. 
    
    Model types:

        lm:    Based on 'LM' model in "An explicit statistical model of learning lexical segmentation using multiple cues"
        (Çöltekin et al, 2014). P(w) is either the relative probability from the lexicon
        or if w is unknown, the joint probability of the phonemes in the word.

        venk:   Based on Venkararaman's model in "A Statistical Model for Word Discovery in Transcribed Speech" (Venkararaman, 2001).
        P(w) is either the relative probability from the lexicon, multiplied by the probability of seeing a previously-seen word, or 
        if w is unknown, the joint probability of the phonemes in the word, multiplied by the probability of seeing a new word.

        blanch: Based on Blanchard et al's model in "Modeling the contribution of phonotactic cues to the problem of word segmentation"
        (Blanchard et al, 2010). Is a direct extension of venk, adding the "require syllabic sound" constraint in unseen words.
        The phoneme n-gram extension also in this paper is available for all models via the `ngram_length` option.
    
    Parameters
    ----------
    alpha : float, optional
        The only parameter of the LM model type, used to weigh unseen words against seen words. Default is 0.5.
    ngram_length : int, optional
        The length of n-grams used in the phoneme language model, used to initialise PhoneStats if not provided.
    type : str, optional
        The type of probabilistic model to use. Default is "lm". 
    phonestats : PhoneStats, optional
        Phoneme statistics updated with each utterance from the corpus. If not provided, a PhoneStats object
        will be created and updated by this model.
    lexicon : Lexicon, optional
        If not provided, a Lexicon object will be created and updated to keep track of previously-seen
        words. 
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, alpha=0.5, ngram_length=1, model_type="lm", phonestats=None, lexicon=None, log=utils.null_logger()):
        super().__init__(log)

        if not model_type in MODEL_TYPES:
            raise ValueError("Cannot initialise Probabilistic Model with unknown model type: '{}'".format(model_type))
        
        # Initialise model parameters
        self.alpha = alpha
        self.ngram_length = ngram_length
        self.type = model_type

        # Initialise lexicon if not provided
        if lexicon is None:
            self._lexicon = Lexicon()
            self._updatelexicon = True
        else:
            self._lexicon = lexicon
            self._updatelexicon = False

        # Initialise phoneme statistics if not provided
        if phonestats is None:
            self._phonestats = PhoneStats(ngram_length+1, use_boundary_tokens=True)
            self._updatephonestats = True
        else:
            self._phonestats = phonestats
            self._updatephonestats = False

    def __str__(self):
        return "ProbabilisticModel(N: {},alpha={},type={})".format(self.ngram_length, self.alpha, self.type)

    # Overrides Model.segment_utterance()
    def segment_utterance(self, utterance, update_model=True):
        """ Segments utterance using Viterbi algorithm.

        Parameters
        ----------
        utterance : PhoneSequence
            A sequence of phones representing the utterance.
        update_model : bool
            When True (default), updates the model online during segmentation.
        """

        segmented = PhoneSequence(utterance.phones)
        n = len(segmented)
        
        # Initial best score is for no segmentation
        best_score = self.word_score(segmented.phones)
        best_segpoint = n

        # Memoisation grids where best_scores[i] stores the best score for utterance[0:i]
        # and best_segpoints[i] scores the best place to split utterance[0:i] (best_segpoint[3] == 1 means the best way
        # to split utterance[0:3] is as utterance[0:1] | utterance[1:3].)
        best_scores = []
        best_segpoints = []

        # Get the best score for utterance[0:i]
        for i in range(n+1):

            # First consider no segmentation
            best_score = self.word_score(segmented.phones[:i])
            best_segpoint = i
            
            # Then consider each split of the sub-utterance to find best split,
            # making use of previously-stored sub-utterance scores.
            for j in range(1, i):
                # score of utterance[0:j] + score of word[j:i]
                score = best_scores[j] + self.word_score(segmented.phones[j:i])
                if score < best_score:
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
                self._log.debug("Probability for {} is {}".format(''.join(segmented.phones[0:segpoint]), pow(2, -self.word_score(segmented.phones[0:segpoint]))))
                break
            self._log.debug("Probability for {} is {}".format(''.join(segmented.phones[new_segpoint:segpoint]), pow(2, -self.word_score(segmented.phones[new_segpoint:segpoint]))))
            #print('placing segpoint at {}'.format(segpoint))
            segmented.boundaries[new_segpoint] = True
            segpoint = new_segpoint
        self._log.debug("Segmenting as {} with score {}".format(segmented, best_scores[n]))

        if update_model:
            self.update(segmented)

        return segmented

    def word_score(self, word):
        """ Get P(word) 

        Parameters
        ----------
        word : list of str
            A sequence of phones representing the word.
        """

        # Score for an empty word is 0
        if len(word) == 0:
            return 0

        if self.type == "lm":
            # If the word is in the lexicon, return the relative frequency
            word_str = ''.join(word)
            if self._lexicon[word_str] != 0 and self.alpha != 1:
                p_w = self._lexicon.relative_freq(word_str, consider_unseen=False)
                return -log2((1 - self.alpha) * p_w)
            
            # Otherwise, return the probability given by the phonemes in the word
            p_w = self._phonestats.get_log_word_probability(word, self.ngram_length)
            return -log2(self.alpha) + p_w

        elif self.type == "venk" or self.type == "blanch":
            # If the word is in the lexicon, return the relative frequency, scaled
            # by the probability of seeing a known word
            word_str = ''.join(word)
            if self._lexicon[word_str] != 0:
                p_w = self._lexicon.relative_freq(word_str, consider_unseen=True)
                return -log2(p_w)
            
            # If using the blanchard model, reject unseen words without syllabic sounds (require-syllabic-sound constraint).
            if self.type == "blanch":
                has_syllabic_sound = False
                for phoneme in word:
                    if phoneme in BR_CORPUS_SYLLABIC_SOUNDS:
                        has_syllabic_sound = True
                        break
                if not has_syllabic_sound:
                    return 10000 # approximation for log(0), made much higher than those below

            # Otherwise, return the probability given by the phonemes in the word, scaled by 
            # the probability of seeing an unknown word. Also considers the probability of a boundary, 
            # calculated as the number of words segmented (tokens) divided by the number of phonemes seen (tokens).
            p_w = self._phonestats.get_log_word_probability(word, self.ngram_length)
            unseen_prob = self._lexicon.unseen_freq()
            boundary_prob = self._lexicon.token_count / self._phonestats.ntokens[1] if self._phonestats.ntokens[1] != 0 else 0
            if p_w >= 10000 or boundary_prob == 0 or boundary_prob == 1 or unseen_prob == 0:
                # print(p_w, boundary_prob, unseen_prob)
                return 10000 # approximation for log(0)
            boundary_factor = boundary_prob/(1-boundary_prob)
            return + p_w - log2(unseen_prob) - log2(boundary_factor)

        else:
            # Shouldn't happen, should be caught by initialisation
            raise ValueError("Unknown model type: '{}'".format(self.type))

    def update(self, segmented):
        """ A method that is called at the end of segment_utterance. Child classes should
        implement this if they wish to update any internal state based on the segmentation.

        Parameters
        ----------
        segmented : PhoneSequence
            A sequence of phones representing the segmented utterance.
        """
        
        """ Updates lexicon and phonestats with newly found words. """
        for word in segmented.get_words():
            if self._updatelexicon:
                self._lexicon.increase_count(''.join(word))
            if self._updatephonestats:
                self._phonestats.add_phones(word)

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    group = parser.add_argument_group('probabilistic model options')
    group.add_argument(
        '-A', '--alpha', type=float, default=0.5, metavar='<float>',
        help='the value of alpha to use, '
        'default is %(default)s')
    group.add_argument(
        '-n', '--ngram_context', type=int, default=0, metavar='<int>',
        help='the context size used for calculating unseen word probabilities. '
        'default is %(default)s, indicating no context used (unigram assumption).')
    group.add_argument(
        '-m', '--model_type', type=str, default="lm", metavar='<str>',
        help='the type of probabilistic model to use. select from "lm" or "venk", default is %(default)s.'
    )

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-probabilistic',
        description=__doc__,
        add_arguments=_add_arguments)

    # Create model
    model = ProbabilisticModel(alpha=args.alpha, ngram_length=args.ngram_context, model_type=args.model_type, log=log)

    # Segment text
    segmented = model.segment(streamin)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()