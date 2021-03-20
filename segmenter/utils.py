""" Utility methods for models """

def segmented_utterance_to_boundaries(segmented_utterance):
    """ Convert a segmented utterance to an array representing boundary positions.
    
    Parameters
    ----------
    segmented_utterance : list of str
        The segmented utterance as a list of phonemes and spaces for word boundaries.
    
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

def boundaries_to_segmented_utterance(utterance, boundaries):
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
    segmented_utterance : list of str
        The segmented utterance as a list of phonemes and spaces for word boundaries.
    
    """

    segmented = []
    for (token, boundary) in zip(utterance.strip().split(' '), boundaries):
        segmented.append(token)
        if boundary:
            segmented.append(' ')
    return segmented

def split_segmented_utterance(segmented_utterance):
    """ Takes a segmented utterance and splits it at the spaces 

    Parameters
    ----------
    segmented_utterance : str
        The segmented utterance as a string of phonemes with spaces at word boundaries.

    Returns
    -------
    words : list of list of str
        A list of words, where each word is a string of phonemes.

    """

    words = []
    w = []
    for phoneme in segmented_utterance:
        if phoneme == ' ':
            if w != []:
                words.append(w)
            w = []
        else:
            w.append(phoneme)
    if w != []:
        words.append(w)
    return words