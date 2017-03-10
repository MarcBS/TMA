import logging
from timeit import default_timer as timer

from config import load_parameters
from data_engine.prepare_data import build_dataset
from viddesc_model import VideoDesc_Model
from keras_wrapper.utils import decode_predictions_beam_search, decode_predictions
from keras_wrapper.cnn_model import loadModel, saveModel, transferWeights, updateModel
from keras_wrapper.extra.callbacks import EvalPerformance, Sample
from keras_wrapper.extra.read_write import dict2pkl, list2file
from keras_wrapper.extra.evaluation import selectMetric

import sys
import ast
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(params):
    """
    Training function. Sets the training parameters from params. Build or loads the model and launches the training.
    :param params: Dictionary of network hyperparameters.
    :return: None
    """

    if params['RELOAD'] > 0:
        logging.info('Resuming training.')

    check_params(params)


    ########### Load data
    dataset = build_dataset(params)
    if not '-vidtext-embed' in params['DATASET_NAME']:
        params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    else:
        params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]
    ###########


    ########### Build model

    if params['MODE'] == 'finetuning':
        #video_model = loadModel(params['PRE_TRAINED_MODEL_STORE_PATHS'], params['RELOAD'])
        video_model = VideoDesc_Model(params,
                                            type=params['MODEL_TYPE'],
                                            verbose=params['VERBOSE'],
                                            model_name=params['MODEL_NAME'] + '_reloaded',
                                            vocabularies=dataset.vocabulary,
                                            store_path=params['STORE_PATH'],
                                            set_optimizer=False,
                                            clear_dirs=False)
        video_model = updateModel(video_model, params['RELOAD_PATH'], params['RELOAD'], reload_epoch=False)
        video_model.setParams(params)

        # Define the inputs and outputs mapping from our Dataset instance to our model
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            if len(video_model.ids_inputs) > i:
                pos_source = dataset.ids_inputs.index(id_in)
                id_dest = video_model.ids_inputs[i]
                inputMapping[id_dest] = pos_source
        video_model.setInputsMapping(inputMapping)

        outputMapping = dict()
        for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
            if len(video_model.ids_outputs) > i:
                pos_target = dataset.ids_outputs.index(id_out)
                id_dest = video_model.ids_outputs[i]
                outputMapping[id_dest] = pos_target
        video_model.setOutputsMapping(outputMapping)

        video_model.setOptimizer()
        params['MAX_EPOCH'] += params['RELOAD']

    else:
        if params['RELOAD'] == 0 or params['LOAD_WEIGHTS_ONLY']: # build new model
            video_model = VideoDesc_Model(params,
                                          type=params['MODEL_TYPE'],
                                          verbose=params['VERBOSE'],
                                          model_name=params['MODEL_NAME'],
                                          vocabularies=dataset.vocabulary,
                                          store_path=params['STORE_PATH'],
                                          set_optimizer=True)
            dict2pkl(params, params['STORE_PATH'] + '/config')

            # Define the inputs and outputs mapping from our Dataset instance to our model
            inputMapping = dict()
            for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
                if len(video_model.ids_inputs) > i:
                    pos_source = dataset.ids_inputs.index(id_in)
                    id_dest = video_model.ids_inputs[i]
                    inputMapping[id_dest] = pos_source
            video_model.setInputsMapping(inputMapping)

            outputMapping = dict()
            for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
                if len(video_model.ids_outputs) > i:
                    pos_target = dataset.ids_outputs.index(id_out)
                    id_dest = video_model.ids_outputs[i]
                    outputMapping[id_dest] = pos_target
            video_model.setOutputsMapping(outputMapping)

            # Only load weights from pre-trained model
            if params['LOAD_WEIGHTS_ONLY'] and params['RELOAD'] > 0:
                for i in range(0, len(params['RELOAD'])):
                    old_model = loadModel(params['PRE_TRAINED_MODEL_STORE_PATHS'][i], params['RELOAD'][i])
                    video_model = transferWeights(old_model, video_model, params['LAYERS_MAPPING'][i])
                video_model.setOptimizer()
                params['RELOAD'] = 0
        else: # resume from previously trained model
            video_model = loadModel(params['PRE_TRAINED_MODEL_STORE_PATHS'], params['RELOAD'])
            video_model.params['LR'] = params['LR']
            video_model.setOptimizer()

            if video_model.model_path != params['STORE_PATH']:
                video_model.setName(params['MODEL_NAME'], models_path=params['STORE_PATH'], clear_dirs=False)
    # Update optimizer either if we are loading or building a model
    video_model.params = params
    video_model.setOptimizer()
    ###########


    ########### Test model saving/loading functions
    #saveModel(video_model, params['RELOAD'])
    #video_model = loadModel(params['STORE_PATH'], params['RELOAD'])
    ###########


    ########### Callbacks
    callbacks = buildCallbacks(params, video_model, dataset)
    ###########


    ########### Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'], 'batch_size': params['BATCH_SIZE'],
                       'homogeneous_batches': params['HOMOGENEOUS_BATCHES'], 'maxlen': params['MAX_OUTPUT_TEXT_LEN'],
                       'lr_decay': params['LR_DECAY'], 'lr_gamma': params['LR_GAMMA'],
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                       'eval_on_sets': params['EVAL_ON_SETS_KERAS'], 'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks, 'reload_epoch': params['RELOAD'], 'epoch_offset': params['RELOAD'],
                       'data_augmentation': params['DATA_AUGMENTATION'],
                       'patience': params.get('PATIENCE', 0),# early stopping parameters
                       'metric_check': params.get('STOP_METRIC', None),
                       'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
                       'each_n_epochs': params.get('EVAL_EACH', 1),
                       'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0)
                       }

    video_model.trainNet(dataset, training_params)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))


def apply_Video_model(params):
    """
        Function for using a previously trained model for sampling.
    """
    
    ########### Load data
    dataset = build_dataset(params)
    if not '-vidtext-embed' in params['DATASET_NAME']:
        params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    else:
        params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]
    ###########
    
    
    ########### Load model
    video_model = loadModel(params['STORE_PATH'], params['SAMPLING_RELOAD_POINT'],
                            reload_epoch=params['SAMPLING_RELOAD_EPOCH'])
    video_model.setOptimizer()
    ###########
    

    ########### Apply sampling
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
    extra_vars['language'] = params.get('TRG_LAN', 'en')

    for s in params["EVAL_ON_SETS"]:

        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': [s]}

        # Convert predictions into sentences
        if not '-vidtext-embed' in params['DATASET_NAME']:
            vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        else:
            vocab = None

        if params['BEAM_SEARCH']:
            params_prediction['beam_size'] = params['BEAM_SIZE']
            params_prediction['maxlen'] = params['MAX_OUTPUT_TEXT_LEN_TEST']
            params_prediction['optimized_search'] = params['OPTIMIZED_SEARCH'] and '-upperbound' not in params['DATASET_NAME']
            params_prediction['model_inputs'] = params['INPUTS_IDS_MODEL']
            params_prediction['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            params_prediction['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            params_prediction['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            params_prediction['normalize'] = params['NORMALIZE_SAMPLING']
            params_prediction['alpha_factor'] = params['ALPHA_FACTOR']
            params_prediction['temporally_linked'] = '-linked' in params['DATASET_NAME'] and '-upperbound' not in params['DATASET_NAME'] and '-video' not in params['DATASET_NAME']
            predictions = video_model.predictBeamSearchNet(dataset, params_prediction)[s]
            predictions = decode_predictions_beam_search(predictions, vocab, verbose=params['VERBOSE'])
        else:
            predictions = video_model.predictNet(dataset, params_prediction)[s]
            predictions = decode_predictions(predictions, 1, vocab, params['SAMPLING'], verbose=params['VERBOSE'])

        # Store result
        filepath = video_model.model_path+'/'+ s +'_sampling.pred' # results file
        if params['SAMPLING_SAVE_MODE'] == 'list':
            list2file(filepath, predictions)
        else:
            raise Exception, 'Only "list" is allowed in "SAMPLING_SAVE_MODE"'


        # Evaluate if any metric in params['METRICS']
        for metric in params['METRICS']:
            logging.info('Evaluating on metric ' + metric)
            filepath = video_model.model_path + '/' + s + '_sampling.' + metric  # results file

            # Evaluate on the chosen metric
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            metrics = selectMetric[metric](
                pred_list=predictions,
                verbose=1,
                extra_vars=extra_vars,
                split=s)

            # Print results to file
            with open(filepath, 'w') as f:
                header = ''
                line = ''
                for metric_ in sorted(metrics):
                    value = metrics[metric_]
                    header += metric_ + ','
                    line += str(value) + ','
                f.write(header + '\n')
                f.write(line + '\n')
            logging.info('Done evaluating on metric ' + metric)


def buildCallbacks(params, model, dataset):
    """
    Builds the selected set of callbacks run during the training of the model.

    :param params: Dictionary of network hyperparameters.
    :param model: Model instance on which to apply the callback.
    :param dataset: Dataset instance on which to apply the callback.
    :return:
    """

    callbacks = []

    if params['METRICS']:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params['TOKENIZATION_METHOD'])}

        if not '-vidtext-embed' in params['DATASET_NAME']:
            vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
        else:
            vocab = None
            extra_vars['n_classes'] = len(dataset.dic_classes[params['OUTPUTS_IDS_DATASET'][0]].values())
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                extra_vars[s]['references'] = eval('dataset.Y_'+s+'["'+params['OUTPUTS_IDS_DATASET'][0]+'"]')

        if params['BEAM_SEARCH']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] =  params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True) and '-upperbound' not in params['DATASET_NAME']
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['normalize'] =  params.get('NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] =  params.get('ALPHA_FACTOR', 1.)
            extra_vars['temporally_linked'] = '-linked' in params['DATASET_NAME'] and '-upperbound' not in params['DATASET_NAME'] and '-video' not in params['DATASET_NAME']
            input_text_id = None
            vocab_src = None

            callback_metric = EvalPerformance(model,
                                               dataset,
                                               gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                               metric_name=params['METRICS'],
                                               set_name=params['EVAL_ON_SETS'],
                                               batch_size=params['BATCH_SIZE'],
                                               each_n_epochs=params['EVAL_EACH'],
                                               extra_vars=extra_vars,
                                               reload_epoch=params['RELOAD'],
                                               is_text=True,
                                               input_text_id=input_text_id,
                                               index2word_y=vocab,
                                               index2word_x=vocab_src,
                                               sampling_type=params['SAMPLING'],
                                               beam_search=params['BEAM_SEARCH'],
                                               save_path=model.model_path,
                                               start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                               write_samples=True,
                                               write_type=params['SAMPLING_SAVE_MODE'],
                                               eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                               save_each_evaluation=params['SAVE_EACH_EVALUATION'],
                                               verbose=params['VERBOSE'])
        else:
            callback_metric = EvalPerformance(model,
                                             dataset,
                                             gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                             metric_name=params['METRICS'],
                                             set_name=params['EVAL_ON_SETS'],
                                             batch_size=params['BATCH_SIZE'],
                                             each_n_epochs=params['EVAL_EACH'],
                                             extra_vars=extra_vars,
                                             reload_epoch=params['RELOAD'],
                                             save_path=model.model_path,
                                             start_eval_on_epoch=params[
                                                 'START_EVAL_ON_EPOCH'],
                                             write_samples=True,
                                             write_type=params['SAMPLING_SAVE_MODE'],
                                             eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                             save_each_evaluation=params[
                                                 'SAVE_EACH_EVALUATION'],
                                             verbose=params['VERBOSE'])

        callbacks.append(callback_metric)

    if params['SAMPLE_ON_SETS']:
        # Write some samples
        extra_vars = {'language': params.get('TRG_LAN', 'en'), 'n_parallel_loaders': params['PARALLEL_LOADERS']}
        if not '-vidtext-embed' in params['DATASET_NAME']:
            vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        else:
            vocab = None
        if params['BEAM_SEARCH']:
            extra_vars['beam_size'] = params['BEAM_SIZE']
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params['MAX_OUTPUT_TEXT_LEN_TEST']
            extra_vars['optimized_search'] = params['OPTIMIZED_SEARCH'] and '-upperbound' not in params['DATASET_NAME']
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['normalize'] = params['NORMALIZE_SAMPLING']
            extra_vars['alpha_factor'] = params['ALPHA_FACTOR']
            extra_vars['temporally_linked'] = '-linked' in params['DATASET_NAME'] and '-upperbound' not in params['DATASET_NAME'] and '-video' not in params['DATASET_NAME']

        callback_sampling = Sample(model,
                                   dataset,
                                   gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                   set_name=params['SAMPLE_ON_SETS'],
                                   n_samples=params['N_SAMPLES'],
                                   each_n_updates=params['SAMPLE_EACH_UPDATES'],
                                   extra_vars=extra_vars,
                                   reload_epoch=params['RELOAD'],
                                   batch_size=params['BATCH_SIZE'],
                                   is_text=True,
                                   index2word_y=vocab,  # text info
                                   in_pred_idx=params['INPUTS_IDS_DATASET'][0],
                                   sampling_type=params['SAMPLING'],  # text info
                                   beam_search=params['BEAM_SEARCH'],
                                   start_sampling_on_epoch=params['START_SAMPLING_ON_EPOCH'],
                                   verbose=params['VERBOSE'])
        callbacks.append(callback_sampling)

    return callbacks



def check_params(params):
    if 'Glove' in params['MODEL_TYPE'] and params['GLOVE_VECTORS'] is None:
        logger.warning("You set a model that uses pretrained word vectors but you didn't specify a vector file."
                       "We'll train WITHOUT pretrained embeddings!")
    if params["USE_DROPOUT"] and params["USE_BATCH_NORMALIZATION"]:
        logger.warning("It's not recommended to use both dropout and batch normalization")


if __name__ == "__main__":

    parameters = load_parameters()
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            parameters[k] = ast.literal_eval(v)
    except ValueError:
        print 'Overwritten arguments must have the form key=Value'
        exit(1)
    check_params(parameters)
    if parameters['MODE'] == 'training' or parameters['MODE'] == 'finetuning':
        logging.info('Running training.')
        train_model(parameters)
    elif parameters['MODE'] == 'sampling':
        logging.info('Running sampling.')
        apply_Video_model(parameters)

    logging.info('Done!')
