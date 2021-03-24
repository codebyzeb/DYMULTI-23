import codecs

import matplotlib.pyplot as plt
from matplotlib import style

from wordseg import evaluate, utils

def get_boundaries(utt, prepared_utt):
    # prepared_utt is space-separated phonemes (e.g 'ab c d e')
    # utt is segmented utterance (e.g. 'abc de')
    # does not return utterance boundaries
    boundaries = []
    phones = prepared_utt.strip().split(' ')
    n = len(utt)
    i = 0
    for phone in phones:
        i += len(phone)
        if i >= n:
            break
        if utt[i] == ' ':
            boundaries.append(True)
            i += 1
        else:
            boundaries.append(False)
    return boundaries

def get_overundersegmentations(text, gold, prepared):
    """ Calculates undersegmentation and oversegmentation """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for (text_utt, gold_utt, prepared_utt) in zip(text, gold, prepared):
        text_boundaries = get_boundaries(text_utt, prepared_utt)
        gold_boundaries = get_boundaries(gold_utt, prepared_utt)
        for i, b in enumerate(text_boundaries):
            if b and gold_boundaries[i]:
                tp += 1
            if b and not gold_boundaries[i]:
                fp += 1
            if not b and gold_boundaries[i]:
                fn += 1
            if not b and not gold_boundaries[i]:
                tn += 1
    overseg = fp / (fp + tn) if (fp + tn) != 0 else 0
    underseg = fn / (fn + tp) if (fn + tp) != 0 else 0
    return (overseg, underseg)

def plot_measure(text, gold, measure, interval=200):
    # Set up graph
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Utterances")

    # Get data
    n = len(text)
    scores = []
    indices = [i for i in range(0, n, interval)]
    for i in indices:
        j = n if (n - i < interval) else i+interval
        results = evaluate.evaluate(text[i:j], gold[i:j])
        scores.append(results[measure])

    # Plot data
    ax.plot(indices, scores, label=measure, linewidth=1.0)
    plt.show()

def plot_overundersegmentation(text, gold, prepared, interval=100):
    # Set up graph
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Utterances")

    # Get data
    n = len(text)
    overs = []
    unders = []
    indices = [i for i in range(0, n, interval)]
    for i in indices:
        j = n if (n - i < interval) else i+interval
        over, under = get_overundersegmentations(text[i:j], gold[i:j], prepared[i:j])
        overs.append(over)
        unders.append(under)
    
    # Plot data
    ax.plot(indices, overs, label="Oversegmentation", linewidth=1.0)
    ax.plot(indices, unders, label="Undersegmentation", linewidth=1.0)
    ax.legend()
    plt.show()

def _add_arguments(parser):
    """Defines custom command-line arguments for wordseg-eval"""
    parser.add_argument(
        'gold', metavar='<gold-file>',
        help='gold file to evaluate the input data on')
    parser.add_argument(
        'prepared', metavar='<prep-file>',
        help='prepared file as outputted by wordseg-prep (space-separated phonemes), '
        'required for a true calculation of true negatives since phonemes may vary in size.')
    parser.add_argument(
        '-m', '--measure', metavar='<str>', default=None,
        help='measure to plot, default is %(default)s.'
    )

def main():
    streamin, streamout, _, log, args = utils.prepare_main(
        name='wordseg-eval',
        description=__doc__,
        add_arguments=_add_arguments)

    log.info('loading input, gold and prepared texts')

    # load the gold text as a list of utterances, remove empty lines
    gold = evaluate._load_text(codecs.open(args.gold, 'r', encoding='utf8'))

    # load the text as a list of utterances, remove empty lines
    text = evaluate._load_text(streamin)

    # load prepared phonemes document, remove empty lines
    prepared = evaluate._load_text(codecs.open(args.prepared, 'r', encoding='utf8'))

    if args.measure is None:
        over, under = get_overundersegmentations(text, gold, prepared)
        print('{}\t{}'.format("oversegmentation", '%.4g' % over))
        print('{}\t{}'.format("undersegmentation", '%.4g' % under))
    elif args.measure == "segmentation":
        log.info('plotting undersegmentation and oversegmentation rates')
        plot_overundersegmentation(text, gold, prepared)
    else:
        log.info('plotting ' + args.measure)
        plot_measure(text, gold, args.measure)

if __name__ == '__main__':
    main()