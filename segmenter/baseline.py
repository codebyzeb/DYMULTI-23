""" A baseline algorithm for word segmentation.

Randomly add word boundaries after the input tokens with a given probability.

"""

import random

from wordseg import utils

from segmenter.model import Model
from segmenter.phonesequence import PhoneSequence

class BaselineModel(Model):

    """ Train and segment using a baseline probability model.

    Places word boundaries with a given probability p, 0 ≤ p ≤ 1.

    Parameters
    ----------
    probability : float, optional
        Probability used to place word boundaries at each candidate boundary.
    log : logging.Logger, optional
        Where to send log messages

    Raises
    ------
    ValueError
        If `probability` is not in the range (0, 1).


    """

    def __init__(self, probability=0.5, log=utils.null_logger()):
        super().__init__(log)

        if (probability < 0 or probability > 1):
            raise ValueError("Cannot initialise Baseline Model, p={} is not a valid probability.".format(probability))
        self.probability = probability

    def __str__(self):
        return "Baseline(P={})".format(self.probability)

    # Overrides Model.segment_utterance()
    def segment_utterance(self, utterance, update_model=True):
        """ Segment an utterance randomly using a boundary probability.
    
        Parameters
        ----------
        utterance : PhoneSequence
            A sequence of phones representing the utterance.
        update_model : bool
            Inherited by parent but not used for this simple model (no state to update).

        Returns
        -------
        segmented : PhoneSequence
            A sequence of phones with word boundaries.
        """

        segmented = PhoneSequence(utterance.phones)
        for i in range(len(segmented)):
            if random.random() < self.probability:
                segmented.boundaries[i] = True

        return segmented

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    parser.add_argument(
        '-r', '--random', type=int, default=None, metavar='<int>',
        help='the seed for initializing the random number generator, '
        'default is based on system time')

    group = parser.add_argument_group('probability of word boundary')
    group.add_argument(
        '-P', '--probability', type=float, default=0.5, metavar='<float>',
        help='the probability to have a word boundary after a phone, '
        'default is %(default)s')

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-baseline',
        description=__doc__,
        add_arguments=_add_arguments)
    
    # setup seed for random number generator
    if args.random:
        log.info('setup random seed to %s', args.random)
    random.seed(args.random)

    # Create baseline model
    log.info('P(word boundary) = %s', args.probability)
    model = BaselineModel(args.probability, log)

    # Segment text
    segmented = model.segment(streamin)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()