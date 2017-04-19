import logging
import random
import sys

import numpy as np

from config import load_parameters
from data_engine.prepare_data import build_dataset
from viddesc_model import VideoDesc_Model

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def build(params):
    ds = build_dataset(params)
    params['OUTPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    vocab = ds.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']

    # We only want the model for decoding
    video_model = VideoDesc_Model(params,
                                  type=params['MODEL_TYPE'],
                                  verbose=0,
                                  model_name=params['MODEL_NAME'],
                                  vocabularies=ds.vocabulary,
                                  store_path=params['STORE_PATH'],
                                  set_optimizer=False)

    return ds, vocab, video_model


def sample(ds, vocab, video_model, n_samples, split='train', verbose=1):
    truth_data = np.random.randint(0, high=eval('ds.len_' + split), size=n_samples)

    matches = 0
    misses = 0
    guesses = 0

    [truth_X, truth_Y] = ds.getXY_FromIndices('train', truth_data)

    truth_Xs = video_model.decode_predictions_beam_search(np.asarray(truth_X[-2]), vocab, verbose=0, pad_sequences=True)
    truth_Ys = video_model.decode_predictions_one_hot(np.asarray(truth_Y[0][0]), vocab)

    for i, (truth_X, truth_Y) in enumerate(zip(truth_Xs, truth_Ys)):
        try:
            fake_data = np.random.randint(0, high=eval('ds.len_' + split), size=n_samples)
            [fake_X, fake_Y] = ds.getXY_FromIndices('train', fake_data)
            fake_Xs = video_model.decode_predictions_beam_search(np.asarray(fake_X[-2]), vocab, verbose=0,
                                                                 pad_sequences=True)
            fake_Ys = video_model.decode_predictions_one_hot(np.asarray(fake_Y[0][0]), vocab)

            print "Input", i, ":", truth_X
            print "Which is the following event?"

            answer_list = [truth_Y] + fake_Ys
            correctness_list = [True] + [False] * len(fake_Ys)
            answer_correctness_list = list(zip(answer_list, correctness_list))
            random.shuffle(answer_correctness_list)
            shuffled_answer_list, shuffled_correctness_list = zip(*answer_correctness_list)
            for j, answer in enumerate(shuffled_answer_list):
                print "\t", j, ":", answer
            action = int(raw_input('Select the upcoming event. \n'))
            if shuffled_correctness_list[action]:
                matches += 1
                if verbose:
                    print "Correct!"
            else:
                misses += 1
                if verbose:
                    print "Not correct!. The correct one was:", shuffled_answer_list[
                        shuffled_correctness_list.index(True)]
            guesses += 1
            print ""
            print ""
        except KeyboardInterrupt:
            return matches, misses, guesses

    return matches, misses, guesses


if __name__ == "__main__":

    parameters = load_parameters()
    ###########
    ds, vocab, model = build(parameters)
    total_matches = 0
    total_misses = 0
    total_guesses = 0
    while True:
        try:
            matches, misses, guesses = sample(ds, vocab, model, 4, split='train', verbose=0)
            total_matches += matches
            total_misses += misses
            total_guesses += guesses
        except KeyboardInterrupt:
            print "Interrupted!"
            print "Total number of matches: %d/%d" % (total_matches, total_guesses)
            print "Total number of misses: %d/%d" % (total_misses, total_guesses)
            print "Precision: %f" % (float(total_matches) / total_guesses)
            sys.exit(0)
