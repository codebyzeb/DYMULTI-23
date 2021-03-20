""" A lexicon-based algorithm for word segmentation.

TODO: Add documentation

"""

from sortedcontainers import SortedDict

from wordseg import utils

from segmenter.phonestats import PhoneStats
from segmenter.multicue import MultiCueModel
from segmenter.peakmodel import PeakModel
from segmenter import utils as utilities

class Lexicon(SortedDict):
    """ Store a noisy lexicon as a sorted dictionary. The lexicon consists
        of frequency counts for each word.
    """

    def increase_count(self, word, k=1):
        """ Increase the frequency count of a given word by k. """

        if word is None or word == "":
            return
        if not word in self:
            self.setdefault(word, k)
        else:
            self[word] += k

    def __getitem__(self, key):
        """ Override default implementation to return 0 if the key is not in the dictionary """

        if not key in self:
            return 0
        return super().__getitem__(key)

class LexiconFrequencyModel(PeakModel):
    """ A simple lexicon-based model for word segmentation.

    Based on "An explicit statistical model of learning lexical segmentation using multiple cues"
    (Çöltekin et al, 2014). Counts the frequency of previously-seen words that start and end at the
    possible boundary within the utterance and uses an increase or decrease of this sum to posit
    a boundary.
    
    For example, if a lexicon include "i", "it", "its", "a", "baby" and "by", all with
    frequency 1, then the count at the position after the "a" in "itsababy" is 2 since "a"
    occurs once and "baby" occurs once, wheras the count at the position before the y is 0
    since "y" is not a word in the lexicon, nor is any sequence of phonemes ending in "b". 

    Parameters
    ----------
    increase : bool, optional
        When true, place word boundaries when the total word frequency at a candidate boundary.
        When false, place word boundaries when the total word frequency decreases after a candidate boundary.
    use_presence : bool, optional
        When True, consider all words in the lexicon to be equally frequent. This is equivilent to using
        type counts, rather than token counts of previously-seen words.
    lexicon : Lexicon, optional
        If not provided, a Lexicon object will be created and updated to keep track of previously-seen
        words. 
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, increase=True, use_presence=False, lexicon=None, log=utils.null_logger()):
        super().__init__(increase, log)

        # Initialise model parameters
        self.use_presence = use_presence

        # Initialise lexicon if not provided
        if lexicon is None:
            self._lexicon = Lexicon()
            self._updatelexicon = True
        else:
            self._lexicon = lexicon
            self._updatelexicon = False

    def __str__(self):
        return "LexiconFreqModel({},{})".format(
            "Increase" if self.increase else "Decrease",
            "Type Frequency" if self.use_presence else "Token Frequency")

    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance by placing word boundaries where there are many words
        in the lexicon that match the phonemes to the left and right of the word boundary.

        Parameters
        ----------
        utterance : str
            An utterance consisting of space-separated phonemes.
        update_model : bool
            Updates the model's lexicon if it controls its own Lexicon object (if it wasn't provided
            one when initialised, it controls its own Lexicon object, otherwise it does not).

        Returns
        -------
        segmented : list of str
            The segmented utterance as a list of phonemes and spaces for word boundaries.
        """

        segmented = super().segment_utterance(utterance, update_model)

        # Update lexicon with newly found words
        if update_model and self._updatelexicon:
            for word in utilities.split_segmented_utterance(segmented):
                self._lexicon.increase_count(''.join(word))
        
        return segmented

    def score(self, utterance, position):
        # Get possible words that end or start at the boundary
        candidate_words = ([''.join(utterance[j:position+1]) for j in range(0, position+1)] +
                            [''.join(utterance[position+1:j]) for j in range(position+2, len(utterance)+1)])

        if self.use_presence:
            word_count = sum([1 for word in candidate_words if self._lexicon[word] > 0])
        else:
            word_count = sum([self._lexicon[word] for word in candidate_words])

        return word_count

class LexiconBoundaryModel(PeakModel):
    """ A simple lexicon-based model for word segmentation.

    Based on "An explicit statistical model of learning lexical segmentation using multiple cues"
    (Çöltekin et al, 2014). Calculates P(boundary | left context) or P(boundary | right context) using
    phoneme statistics gathered from the lexicon (treating each previously-seen word as its own utterance).
    
    For example, if a lexicon include "abb" and "cd" then P(right boundary | "bb") will be high, whereas
    P(right boundary | "ab") will be low (since no entries in the lexicon end in "ab").

    Parameters
    ----------
    ngram_length : int, optional
        The length of n-grams used in the context.
    increase : bool, optional
        When true, place word boundaries when the boundary probability increases at a candidate boundary.
        When false, place word boundaries when the boundary probability decreases after a candidate boundary.
    right : bool, optional
        When True, calculates P(boundary | right context).
        When False, calculates P(boundary | left context).
    lexicon : Lexicon, optional
        If not provided, a Lexicon object will be created and updated to keep track of previously-seen
        words. 
    phonestats : phonestats.PhoneStats, optional
        If not provided, a PhoneStats object will be created and updated to keep track of phoneme counts.
        Otherwise, the provided object will be used, but not updated.
    log : logging.Logger, optional
        Where to send log messages

    """

    def __init__(self, ngram_length=1, increase=True, right=True, lexicon=None, phonestats=None, log=utils.null_logger()):
        super().__init__(increase, log)

        # Initialise model parameters
        self.ngram_length = ngram_length
        self.right = right

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
        return "LexiconBoundaryModel({},{},{})".format(
            "N: " + str(self.ngram_length),
            "Increase" if self.increase else "Decrease",
            "Right Context" if self.right else "Left Context")

    def segment_utterance(self, utterance, update_model=True):
        """ Segment a single utterance by placing word boundaries where the left or right context
        at the boundary is often found at the edge of words in the lexicon.

        Parameters
        ----------
        utterance : str
            An utterance consisting of space-separated phonemes.
        update_model : bool
            Updates the model's lexicon if it controls its own Lexicon object (if it wasn't provided
            one when initialised, it controls its own Lexicon object, otherwise it does not).

        Returns
        -------
        segmented : list of str
            The segmented utterance as a list of phonemes and spaces for word boundaries.
        """

        segmented = super().segment_utterance(utterance, update_model)

        # Update lexicon and phonestats with newly found words
        if update_model:
            for word in utilities.split_segmented_utterance(segmented):
                if self._updatelexicon:
                    self._lexicon.increase_count(''.join(word))
                if self._updatephonestats:
                    self._phonestats.add_utterance(word)
        
        return segmented  

    def score(self, utterance, position):
        """ Get the conditional boundary probability using the phonestats of the lexicon """

        return self._phonestats.get_unpredictability(utterance=utterance, position=position,
                    measure="bp", reverse=self.right, ngram_length=self.ngram_length)

class MultiLexiconModel(MultiCueModel):
    """ Train and segment using multiple lexicon-based cues .

    Uses a Multiple Cue model for majority voting of each weak lexicon-based model.
    Creates a LexiconFrequencyModel for each direction and a LexiconBoundaryModel
    for each direction and each ngram length up to the maximum.

    Parameters
    ----------
    ngram_length : int, optional
        The maximum length of n-grams used to calculate predictability. A model is created for
        each ngram length from 1 to this value.
    direction : str, optional
        When "Forwards", only the forwards cues are used.
        When "Reverse", only the backwards cues are used. Otherwise,
        uses both measures.
    smoothing : float, optional
        If non-zero, use add-k smoothing for probabilities.
    log : logging.Logger, optional
        Where to send log messages

    Raises
    ------
    ValueError
        If `ngram` is less than 1.

    """

    def __init__(self, max_ngram=1, direction="forwards", smoothing=0, model_selection="both", log=utils.null_logger()):
        
        # Initialise model parameters
        if max_ngram < 1:
            raise ValueError("Cannot initialise a Multi Predictability Model with non-positive n-gram length.")

        self.max_ngram = max_ngram
        self._phonestats = PhoneStats(max_ngram+1, use_boundary_tokens=True, smoothing=smoothing)
        self._lexicon = Lexicon()

        # Create models
        models = []
        if model_selection != "boundary":
            models.append(LexiconFrequencyModel(increase=True, use_presence=False, lexicon=self._lexicon, log=log))
            models.append(LexiconFrequencyModel(increase=False, use_presence=False, lexicon=self._lexicon, log=log))
        if model_selection != "frequency":
            for n in range(1, max_ngram+1):
                if direction != "reverse":
                    models.append(LexiconBoundaryModel(ngram_length=n, increase=True, right=False, lexicon=self._lexicon, phonestats=self._phonestats, log=log))
                    models.append(LexiconBoundaryModel(ngram_length=n, increase=False, right=False, lexicon=self._lexicon, phonestats=self._phonestats, log=log))
                if direction != "forwards":
                    models.append(LexiconBoundaryModel(ngram_length=n, increase=True, right=True, lexicon=self._lexicon, phonestats=self._phonestats, log=log))
                    models.append(LexiconBoundaryModel(ngram_length=n, increase=False, right=True, lexicon=self._lexicon, phonestats=self._phonestats, log=log))

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

        # Update lexicon and phonestats with newly found words
        if update_model:
            for word in utilities.split_segmented_utterance(segmented):
                self._lexicon.increase_count(''.join(word))
                self._phonestats.add_utterance(word)

        return segmented

    def __str__(self):
        return "MultiLexiconModel({})".format(", ".join([str(model) for model in self.models]))

def segment(text, max_ngram=1, direction="forwards", smoothing=0, model_selection="both", log=utils.null_logger()):
    """ Segment using a Multi Cue segmenter model composed of a collection of Predictability models. """

    log.info('Using a Multi Lexicon model to segment text.')

    # TODO: Check the input is valid
    log.info('{} smoothing for probability estimates'.format("Using add-"+str(smoothing) if smoothing else "Not using"))

    model = MultiLexiconModel(max_ngram=max_ngram, direction=direction, smoothing=smoothing, model_selection=model_selection, log=log)
    
    return model.segment(text)

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    group = parser.add_argument_group('Multi Lexicon Model Options')
    group.add_argument(
        '-n', '--max_ngram', type=int, default=1, metavar='<int>',
        help='the maximum length of ngram to use to calculate predictability, '
        'default is %(default)s')
    group.add_argument(
        '-d', '--direction', type=str, default="forwards", metavar='<str>',
        help='Select whether to use "forwards" context, "backwards" context or "both". '
        'default is %(default)s')
    group.add_argument(
        '-S', '--smooth', type=float, default=0.0, metavar='<float>',
        help='What value of k to use for add-k smoothing for probability calculations.')
    group.add_argument(
        '-M', '--models', type=str, default="both", metavar='<float>',
        help='Whether to use the "frequency" model, the "boundary" model or "both".')

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-multilexicon',
        description=__doc__,
        add_arguments=_add_arguments)

    segmented = segment(streamin, max_ngram=args.max_ngram, smoothing=args.smooth,
                        direction=args.direction, model_selection=args.models, log=log)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()
