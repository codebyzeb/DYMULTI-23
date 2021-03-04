""" A mulitple-cue algorithm for word segmentation.

TODO: Add documentation

"""

import random
import numpy as np

from wordseg import utils

from segmenter.model import Model
from segmenter.baseline import BaselineModel

class MultiCueModel(Model):
    """ Train and segment using multiple models as individual cues

    Parameters
    ----------
    models : list of Model
        A list of Model objects used for segmentation, whose suggestions are combined 
        using weighted majority voting to produce a final segmentation.
    log : logging.Logger, optional
        Where to send log messages

    Raises
    ------
    ValueError
        If any model in `models` is not an instance of Model

    """

    def __init__(self, models=[], log=utils.null_logger()):
        super().__init__(log)

        # Default implementation - use a baseline model if no others provided
        if len(models) == 0:
            models = [BaselineModel(probability=0.5, log=log)]

        for model in models:
            if not isinstance(model, Model):
                raise ValueError("Object provided to MultiCueModel is not an instance of Model:"
                    + str(model))

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

        # Get suggested segmentations from each model
        segmentations = [model.segment_utterance(utterance, update_model) for model in self.models]
        boundaries = np.array([self._segmented_utterance_to_boundaries(segmentation) for segmentation in segmentations])

        # Use weighted majority voting at each boundary position to find best segmentation
        # We don't do majority voting for the last boundary (we assume all algorithms can
        # correctly place utterance boundaries)
        best_boundaries = [self._make_boundary_decision(boundary_votes, update_model)
                           for boundary_votes in boundaries.T[:-1]]

        # Appending utterance boundary (not a word boundary)
        best_boundaries.append(False)

        segmented = self._boundaries_to_segmented_utterance(utterance, best_boundaries)

        self._log.debug("Segmented utterance '{}' as '{}".format(''.join(utterance.strip().split(' ')), segmented))
        self._log.debug("Current errors: {} out of {}".format(self.errors, self.num_boundaries))
        self._log.debug("Current weights: {}".format(self.weights))

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

    # TODO: Maybe move these two methods to a "utilities" file, as they don't depend on this class

    def _segmented_utterance_to_boundaries(self, segmented_utterance):
        """ Convert a segmented utterance to an array representing boundary positions.
        
        Parameters
        ----------
        segmented_utterance : str
            The segmented utterance as a string of phonemes with spaces at word boundaries.

        Returns
        -------
        boundaries : list of bool
            The boundary positions where True indicates a boundary after the associated phoneme.
        
        """

        boundaries = []
        for c in segmented_utterance:
            if c == ' ':
                boundaries[-1] = True
            else:
                boundaries.append(False)
        
        # Boundaries array should be the same length as the unsegmented utterance
        assert(len(boundaries) == len(segmented_utterance) - segmented_utterance.count(' '))

        return boundaries

    def _boundaries_to_segmented_utterance(self, utterance, boundaries):
        """ Combines an unsegmented utterance with a list of boundaries to produce
        a segmented utterance

        Parameters
        ----------
        utterance : str
            An utterance consisting of space-separated phonemes.
        boundaries : list of bool
            Boundary positions where True indicates a boundary after the associated phoneme.

        Returns
        -------
        segmented_utterance : str
            The segmented utterance as a string of phonemes with spaces at word boundaries.
        """

        return ''.join(
            token + ' ' if boundary else token
            for (token, boundary) in zip(utterance.strip().split(' '), boundaries))


def segment(text, log=utils.null_logger()):
    """ Segment using a Multicue segmenter model """

    log.info('Using a Multiple Cue model to segment text.')

    # TODO: Add better parameters here, instead of this default behaviour
    model = MultiCueModel(models=[BaselineModel(1), BaselineModel(0)], log=log)
    return model.segment(text)

def _add_arguments(parser):
    """ Add algorithm specific options to the parser """

    # TODO: Add arguments here for submodel selection
    parser.add_argument(
        '-r', '--random', type=int, default=None, metavar='<int>',
        help='the seed for initializing the random number generator, '
        'default is based on system time')

def main():
    """ Entry point """
    streamin, streamout, _, log, args = utils.prepare_main(
        name='segmenter-multicue',
        description=__doc__,
        add_arguments=_add_arguments)
    
    # setup seed for random number generator
    if args.random:
        log.info('setup random seed to %s', args.random)
    random.seed(args.random)

    segmented = segment(streamin, log=log)
    streamout.write('\n'.join(segmented) + '\n')

if __name__ == '__main__':
    main()
