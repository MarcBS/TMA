from keras_wrapper.dataset import Dataset, saveDataset, loadDataset
from keras_wrapper.extra.read_write import pkl2dict
import copy
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def build_dataset(params):
    if params['REBUILD_DATASET']:  # We build a new dataset instance
        if params['VERBOSE'] > 0:
            silence = False
            logging.info('Building ' + params['DATASET_NAME'] + ' dataset')
        else:
            silence = True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME']
        ds = Dataset(name, base_path, silence=silence)

        # OUTPUT DATA
        # Let's load the train, val and test splits of the descriptions (outputs)
        #    the files include a description per line. In this dataset a variable number
        #    of descriptions per video are provided.
        ds.setOutput(base_path + '/' + params['DESCRIPTION_FILES']['train'],
                     'train',
                     type='text',
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     build_vocabulary=True,
                     tokenization=params['TOKENIZATION_METHOD'],
                     fill=params['FILL'],
                     pad_on_batch=True,
                     max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                     sample_weights=params['SAMPLE_WEIGHTS'],
                     min_occ=params['MIN_OCCURRENCES_VOCAB'])

        ds.setOutput(base_path + '/' + params['DESCRIPTION_FILES']['val'],
                     'val',
                     type='text',
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     build_vocabulary=True,
                     pad_on_batch=True,
                     tokenization=params['TOKENIZATION_METHOD'],
                     sample_weights=params['SAMPLE_WEIGHTS'],
                     max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                     min_occ=params['MIN_OCCURRENCES_VOCAB'])

        ds.setOutput(base_path + '/' + params['DESCRIPTION_FILES']['test'],
                     'test',
                     type='text',
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     build_vocabulary=True,
                     pad_on_batch=True,
                     tokenization=params['TOKENIZATION_METHOD'],
                     sample_weights=params['SAMPLE_WEIGHTS'],
                     max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                     min_occ=params['MIN_OCCURRENCES_VOCAB'])

        # INPUT DATA
        # Let's load the associated videos (inputs)
        #    we must take into account that in this dataset we have a different number of sentences per video, 
        #    for this reason we introduce the parameter 'repeat_set'=num_captions, where num_captions is a list
        #    containing the number of captions in each video.

        num_captions_train = np.load(base_path + '/' + params['DESCRIPTION_COUNTS_FILES']['train'])
        num_captions_val = np.load(base_path + '/' + params['DESCRIPTION_COUNTS_FILES']['val'])
        num_captions_test = np.load(base_path + '/' + params['DESCRIPTION_COUNTS_FILES']['test'])

        for feat_type in params['FEATURE_NAMES']:
            for split, num_cap in zip(['train', 'val', 'test'],
                                      [num_captions_train, num_captions_val, num_captions_test]):
                list_files = base_path + '/' + params['FRAMES_LIST_FILES'][split] % feat_type
                counts_files = base_path + '/' + params['FRAMES_COUNTS_FILES'][split] % feat_type

                ds.setInput([list_files, counts_files],
                            split,
                            type=params['INPUT_DATA_TYPE'],
                            id=params['INPUTS_IDS_DATASET'][0],
                            repeat_set=num_cap,
                            max_video_len=params['NUM_FRAMES'],
                            feat_len=params['IMG_FEAT_SIZE'])

        if len(params['INPUTS_IDS_DATASET']) > 1:
            ds.setInput(base_path + '/' + params['DESCRIPTION_FILES']['train'],
                        'train',
                        type='text',
                        id=params['INPUTS_IDS_DATASET'][1],
                        required=False,
                        tokenization=params['TOKENIZATION_METHOD'],
                        pad_on_batch=True,
                        build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                        offset=1,
                        fill=params['FILL'],
                        max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                        max_words=params['OUTPUT_VOCABULARY_SIZE'],
                        min_occ=params['MIN_OCCURRENCES_VOCAB'])

            ds.setInput(None, 'val', type='ghost', id=params['INPUTS_IDS_DATASET'][1], required=False)
            ds.setInput(None, 'test', type='ghost', id=params['INPUTS_IDS_DATASET'][1], required=False)

        # Set inputs for temporally-linked samples
        if '-linked' in params['DATASET_NAME']:
            # Set input captions from previous event/video
            if '-upperbound' not in params['DATASET_NAME']:
                if '-video' in params['DATASET_NAME']:
                    ds, repeat_images = insertTemporallyLinkedCaptions(ds, params,
                                                                       set_names=['train', 'val', 'test'],
                                                                       video='-video' in params['DATASET_NAME'])
                    num_captions_val = repeat_images['val']
                    num_captions_test = repeat_images['test']
                else:
                    ds, repeat_images = insertTemporallyLinkedCaptions(ds, params)
                    ds.setInput([],
                                'val',
                                type='text',
                                id=params['INPUTS_IDS_DATASET'][2],
                                build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                                tokenization=params['TOKENIZATION_METHOD'],
                                fill=params['FILL'],
                                pad_on_batch=True,
                                max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                                min_occ=params['MIN_OCCURRENCES_VOCAB'],
                                required=False)
                    ds.setInput([],
                                'test',
                                type='text',
                                id=params['INPUTS_IDS_DATASET'][2],
                                build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                                tokenization=params['TOKENIZATION_METHOD'],
                                fill=params['FILL'],
                                pad_on_batch=True,
                                max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                                min_occ=params['MIN_OCCURRENCES_VOCAB'],
                                required=False)
            else:
                ds, repeat_images = insertTemporallyLinkedCaptions(ds,
                                                                   params,
                                                                   set_names=['train', 'val', 'test'],
                                                                   upperbound=True,
                                                                   video= '-video' in params['DATASET_NAME'],
                                                                   copy= '-copy' in params['DATASET_NAME'],
                                                                   force_nocopy='-nocopy' in params['DATASET_NAME'],
                                                                   prev= '-prev' in params['DATASET_NAME'])
                num_captions_val = repeat_images['val']
                num_captions_test = repeat_images['test']


        # Process dataset for keeping only one caption per video and storing the rest in a dict() with the following format:
        #        ds.extra_variables[set_name][id_output][img_position] = [cap1, cap2, cap3, ..., capN]
        keep_n_captions(ds, repeat=[num_captions_val, num_captions_test], n=1, set_names=['val', 'test'])

        if '-linked' in params['DATASET_NAME'] and '-upperbound' not in params['DATASET_NAME'] and '-video' not in params['DATASET_NAME']:
            # Set previous data indices
            for s, file in params['LINK_SAMPLE_FILES'].iteritems():
                if s in repeat_images:
                    rep = repeat_images[s]
                else:
                    rep = 1
                ds.setInput(base_path + '/' + file,
                            s,
                            type='id',
                            id=params['INPUTS_IDS_DATASET'][3],
                            repeat_set=rep)

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME'] + '.pkl')

    # Load vocabulary-related parameters of dataset used for pre-training
    if params['PRE_TRAINED_DATASET_NAME'] is not None:
        dataset_pretrained = loadDataset(params['DATASET_STORE_PATH'] + 'Dataset_' + params['PRE_TRAINED_DATASET_NAME'] + '.pkl')
        for id_new, id_old in params['VOCABULARIES_MAPPING'].iteritems():
            ds.vocabulary[id_new] = copy.deepcopy(dataset_pretrained.vocabulary[id_old])
            ds.vocabulary_len[id_new] = copy.deepcopy(dataset_pretrained.vocabulary_len[id_old])
    elif params['PRE_TRAINED_VOCABULARY_NAME'] is not None:
        dataset_pretrained_vocabulary = pkl2dict(params['DATASET_STORE_PATH'] + params['PRE_TRAINED_VOCABULARY_NAME'] + '.pkl')
        for id_new, id_old in params['VOCABULARIES_MAPPING'].iteritems():
            ds.vocabulary[id_new] = copy.deepcopy(dataset_pretrained_vocabulary[id_old])
            ds.vocabulary_len[id_new] = len(dataset_pretrained_vocabulary[id_old]['idx2words'])

    return ds


def keep_n_captions(ds, repeat, n=1, set_names=['val', 'test']):
    ''' Keeps only n captions per image and stores the rest in dictionaries for a later evaluation
    '''

    for s, r in zip(set_names, repeat):
        logging.info('Keeping ' + str(n) + ' captions per input on the ' + str(s) + ' set.')

        ds.extra_variables[s] = dict()
        exec ('n_samples = ds.len_' + s)

        # Process inputs
        for id_in in ds.ids_inputs:
            new_X = []
            if id_in in ds.optional_inputs:
                try:
                    exec ('X = ds.X_' + s)
                    i = 0
                    for next_repeat in r:
                        for j in range(n):
                            new_X.append(X[id_in][i + j])
                        i += next_repeat
                    exec ('ds.X_' + s + '[id_in] = new_X')
                except:
                    pass
            else:
                exec ('X = ds.X_' + s)
                i = 0
                for next_repeat in r:
                    for j in range(n):
                        new_X.append(X[id_in][i + j])
                    i += next_repeat
                exec ('ds.X_' + s + '[id_in] = new_X')
        # Process outputs
        for id_out in ds.ids_outputs:
            new_Y = []
            exec ('Y = ds.Y_' + s)
            dict_Y = dict()
            count_samples = 0
            i = 0
            for next_repeat in r:
                dict_Y[count_samples] = []
                for j in range(next_repeat):
                    if j < n:
                        new_Y.append(Y[id_out][i + j])
                    dict_Y[count_samples].append(Y[id_out][i + j])
                count_samples += 1
                i += next_repeat
            exec ('ds.Y_' + s + '[id_out] = new_Y')
            # store dictionary with vid_pos -> [cap1, cap2, cap3, ..., capNi]
            ds.extra_variables[s][id_out] = dict_Y

        new_len = len(new_Y)
        exec ('ds.len_' + s + ' = new_len')
        logging.info('Samples reduced to ' + str(new_len) + ' in ' + s + ' set.')


def insertTemporallyLinkedCaptions(ds, params, set_names=['train'],
                                   upperbound=False,
                                   video=False, copy=False, force_nocopy=False, prev=False):
    """
        Inserts an additional input consisting of the desired captions from the previous segment/event
        in chronological order. Example:
            <video1, in_caption1> : <out_caption1>
            <video1, in_caption1> : <out_caption2>
            .
            .
            .
            <video1, in_captionM> : <out_captionN>
            <video2, in_caption1> : <out_caption1>
            .
            .
            .

        :param ds: dataset to modify
        :param params: parameters from config
        :param set_names: names of the splits that will be modified (default 'train' only)
        :param upperbound: whether we want to generate a dataset for an upper bound comparison by using the same captions both as input and output
        :param video: whether we use the previous' event video as input instead of the previous caption
        :param copy: generates an upperbound dataset only intending to copy giving only matching input-output sequences (only valid if upperbound=True)
        :param force_nocopy: generates an upperbound dataset using the same captions both as input and output but avoiding direct copies
        :param prev: indicates if we want to use the previous event's caption as input for the next, or use the current event's output instead

        :return: dataset modified with the additional input
    """
    base_path = params['DATA_ROOT_PATH']
    repeat_images = dict()

    for s in set_names:
        # retrieve number of output captions per sample
        num_cap = np.load(base_path + '/' + params['DESCRIPTION_COUNTS_FILES'][s])

        # get temporal links
        links = []
        with open(base_path + '/' + params['LINK_SAMPLE_FILES'][s], 'r') as f_links:
            for line in f_links:
                links.append(int(line.strip()))

        outputs = []
        with open(base_path + '/' + params['DESCRIPTION_FILES'][s], 'r') as f_outs:
            for line in f_outs:
                outputs.append(line.strip())

        # get outputs
        if video:
            prev_videos = []
            for feat_type in params['FEATURE_NAMES']:
                list_files = base_path + '/' + params['FRAMES_LIST_FILES'][s] % feat_type
                counts_files = base_path + '/' + params['FRAMES_COUNTS_FILES'][s] % feat_type
                with open(list_files, 'r') as f_outs, open(counts_files, 'r') as f_outs_counts:
                    prev_videos.append([[line.strip() for line in f_outs], [int(line.strip()) for line in f_outs_counts]])

        # modify outputs and prepare inputs
        images_repeat = []
        upperbound_images_repeat = []
        final_outputs = []
        if video:
            final_inputs = dict()
            for feat_type in params['FEATURE_NAMES']:
                final_inputs[feat_type] = [[], []]
        else:
            final_inputs = []
        for i, link in enumerate(links):
            ini_out = np.sum(num_cap[:i])
            these_outputs = outputs[ini_out:ini_out + num_cap[i]]

            if upperbound:
                if copy:
                    images_repeat.append(num_cap[i])
                    upperbound_images_repeat.append(num_cap[i])
                    for out in these_outputs:
                        final_outputs.append(out)
                        final_inputs.append(out)
                elif prev:
                    # first sample in the temporally-linked sequence
                    if link == -1:
                        images_repeat.append(num_cap[i])
                        upperbound_images_repeat.append(num_cap[i])
                        for out in these_outputs:
                            final_outputs.append(out)
                            final_inputs.append('')
                    else:
                        prev_ini_out = np.sum(num_cap[:link])
                        prev_outputs = outputs[prev_ini_out:prev_ini_out + num_cap[link]]
                        images_repeat.append(num_cap[i] * num_cap[link])
                        for n in range(num_cap[link]):
                            upperbound_images_repeat.append(num_cap[i])
                            for out in these_outputs:
                                final_outputs.append(out)
                                final_inputs.append(prev_outputs[n])
                elif force_nocopy:
                    raise NotImplementedError()
                    prev_outputs = these_outputs
                    images_repeat.append(num_cap[i] * (num_cap[i]-1))
                    for n in range(num_cap[i]):
                        upperbound_images_repeat.append(num_cap[i]-1)
                        for nthese, out in enumerate(these_outputs):
                            if nthese != n:
                                final_outputs.append(out)
                                final_inputs.append(prev_outputs[n])
                else:
                    prev_outputs = these_outputs
                    images_repeat.append(num_cap[i]*num_cap[i])
                    for n in range(num_cap[i]):
                        upperbound_images_repeat.append(num_cap[i])
                        for out in these_outputs:
                            final_outputs.append(out)
                            final_inputs.append(prev_outputs[n])
            else:
                if video:
                    # first sample in the temporally-linked sequence
                    if link == -1:
                        images_repeat.append(num_cap[i])
                        for out in these_outputs:
                            final_outputs.append(out)
                        for ifeat, feat_type in enumerate(params['FEATURE_NAMES']):
                            #final_inputs[feat_type][0].append('')
                            #final_inputs[feat_type][1].append(0)
                            final_inputs[feat_type][1] += [0]
                    else:
                        images_repeat.append(num_cap[i])
                        for out in these_outputs:
                            final_outputs.append(out)
                        for ifeat, feat_type in enumerate(params['FEATURE_NAMES']):
                            #final_inputs[feat_type][0].append(prev_videos[ifeat][0][link])
                            #final_inputs[feat_type][1].append(prev_videos[ifeat][1][link])
                            if link > 0:
                                init_frame = int(sum(prev_videos[ifeat][1][:link]))
                            else:
                                init_frame = 0
                            this_count = prev_videos[ifeat][1][link]
                            final_inputs[feat_type][0] += prev_videos[ifeat][0][init_frame:init_frame+this_count]
                            final_inputs[feat_type][1] += [this_count]
                else:
                    # first sample in the temporally-linked sequence
                    if link == -1:
                        images_repeat.append(num_cap[i])
                        for out in these_outputs:
                            final_outputs.append(out)
                            final_inputs.append('')
                    else:
                        prev_ini_out = np.sum(num_cap[:link])
                        prev_outputs = outputs[prev_ini_out:prev_ini_out + num_cap[link]]
                        images_repeat.append(num_cap[i] * num_cap[link])
                        for n in range(num_cap[link]):
                            for out in these_outputs:
                                final_outputs.append(out)
                                final_inputs.append(prev_outputs[n])

        # Overwrite input images assigning the new repeat pattern
        for feat_type in params['FEATURE_NAMES']:
            list_files = base_path + '/' + params['FRAMES_LIST_FILES'][s] % feat_type
            counts_files = base_path + '/' + params['FRAMES_COUNTS_FILES'][s] % feat_type

            ds.setInput([list_files, counts_files],
                        s,
                        type=params['INPUT_DATA_TYPE'],
                        id=params['INPUTS_IDS_DATASET'][0],
                        repeat_set=images_repeat,
                        max_video_len=params['NUM_FRAMES'],
                        feat_len=params['IMG_FEAT_SIZE'],
                        overwrite_split=True)

        if not video:
            # Overwrite outputs assigning the new outputs repeat pattern
            ds.setOutput(final_outputs,
                         s,
                         type='text',
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         build_vocabulary=True,
                         tokenization=params['TOKENIZATION_METHOD'],
                         fill=params['FILL'],
                         pad_on_batch=True,
                         max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                         sample_weights=params['SAMPLE_WEIGHTS'],
                         min_occ=params['MIN_OCCURRENCES_VOCAB'],
                         overwrite_split=True)

            # Overwrite the input state_below assigning the new outputs repeat pattern
            ds.setInput(final_outputs,
                        s,
                        type='text',
                        id=params['INPUTS_IDS_DATASET'][1],
                        required=False,
                        tokenization=params['TOKENIZATION_METHOD'],
                        pad_on_batch=True,
                        build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                        offset=1,
                        fill=params['FILL'],
                        max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                        max_words=params['OUTPUT_VOCABULARY_SIZE'],
                        min_occ=params['MIN_OCCURRENCES_VOCAB'],
                        overwrite_split=True)

        if video:
            for feat_type in params['FEATURE_NAMES']:
                ds.setInput(final_inputs[feat_type],
                        s,
                        type=params['INPUT_DATA_TYPE'],
                        id=params['INPUTS_IDS_DATASET'][2],
                        repeat_set=images_repeat,
                        max_video_len=params['NUM_FRAMES'],
                        feat_len=params['IMG_FEAT_SIZE'],
                        overwrite_split=True)
        else:
            # Set new input captions from previous temporally-linked event/video
            ds.setInput(final_inputs,
                        s,
                        type='text',
                        id=params['INPUTS_IDS_DATASET'][2],
                        build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                        tokenization=params['TOKENIZATION_METHOD'],
                        fill=params['FILL'],
                        pad_on_batch=True,
                        max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                        min_occ=params['MIN_OCCURRENCES_VOCAB'])

        if upperbound:
            images_repeat = upperbound_images_repeat
        repeat_images[s] = images_repeat

    return ds, repeat_images
