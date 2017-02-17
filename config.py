def load_parameters():
    """
        Loads the defined parameters
    """
    # Input data params
    #DATA_ROOT_PATH = '/media/HDD_2TB/DATASETS/EDUB-SegDesc/'        # Root path to the data
    DATA_ROOT_PATH = '/media/HDD_3TB/DATASETS/EDUB-SegDesc/'

    # preprocessed features
    DATASET_NAME = 'EDUB-SegDesc_features-linked'   # Dataset name (add '-linked' suffix for using
                                                    # dataset with temporally-linked training data)
                                                    #
                                                    #    -linked
                                                    #    -linked-upperbound
                                                    #    -linked-upperbound-copy
                                                    #    -linked-upperbound-prev
                                                    #    -linked-upperbound-nocopy
                                                    #    -linked-video
                                                    #    -vidtext-embed
                                                    #

    PRE_TRAINED_DATASET_NAME = 'MSVD_features'      # Dataset name for reusing vocabulary of pre-trained model
                                                    # (only applicable if we are using a pre-trained model, default None)
    VOCABULARIES_MAPPING = {'description': 'description',
                        'state_below': 'description',
                        'prev_description': 'description'}

    PRE_TRAINED_VOCABULARY_NAME = None #'1BillionWords_vocabulary'      # Dataset name for reusing vocabulary of pre-trained model
    """
    VOCABULARIES_MAPPING = {'description': 'target_text',
                            'state_below': 'target_text',
                            'prev_description': 'target_text'}
    """

    # Input data
    INPUT_DATA_TYPE = 'video-features'                          # 'video-features' or 'video'
    NUM_FRAMES = 26                                             # fixed number of input frames per video
    
    #### Features from video frames
    FRAMES_LIST_FILES = {'train': 'Annotations/%s/train_feat_list.txt',                 # Feature frames list files
                         'val': 'Annotations/%s/val_feat_list.txt',
                         'test': 'Annotations/%s/test_feat_list.txt',
                        }
    FRAMES_COUNTS_FILES = {  'train': 'Annotations/%s/train_feat_counts.txt',           # Frames counts files
                             'val': 'Annotations/%s/val_feat_counts.txt',
                             'test': 'Annotations/%s/test_feat_counts.txt',
                          }
    FEATURE_NAMES = ['ImageNet'] # append '_L2' at the end of each feature type if using their L2 version

    #FEATURE_NAMES = ['ImageNetFV', 'ImageNet', 'Scenes', 'C3D'] # append '_L2' at the end of each feature type if using their L2 version
    
    # Output data
    DESCRIPTION_FILES = {'train': 'Annotations/train_descriptions.txt',                 # Description files
                         'val': 'Annotations/val_descriptions.txt',
                         'test': 'Annotations/test_descriptions.txt',
                        }
    DESCRIPTION_COUNTS_FILES = { 'train': 'Annotations/train_descriptions_counts.npy',  # Description counts files
                                 'val': 'Annotations/val_descriptions_counts.npy',
                                 'test': 'Annotations/test_descriptions_counts.npy',
                               }

    # Dataset parameters
    if not '-vidtext-embed' in DATASET_NAME:
        INPUTS_IDS_DATASET = ['video', 'state_below']  # Corresponding inputs of the dataset
        OUTPUTS_IDS_DATASET = ['description']  # Corresponding outputs of the dataset
        INPUTS_IDS_MODEL = ['video', 'state_below']  # Corresponding inputs of the built model
        OUTPUTS_IDS_MODEL = ['description']  # Corresponding outputs of the built model
    else:
        INPUTS_IDS_DATASET = ['video', 'description']  # Corresponding inputs of the dataset
        OUTPUTS_IDS_DATASET = ['match']  # Corresponding outputs of the dataset
        INPUTS_IDS_MODEL = ['video', 'description']  # Corresponding inputs of the built model
        OUTPUTS_IDS_MODEL = ['match']  # Corresponding outputs of the built model


    if '-linked' in DATASET_NAME:

        LINK_SAMPLE_FILES = {'train': 'Annotations/train_link_samples.txt',     # Links index files
                             'val': 'Annotations/val_link_samples.txt',
                             'test': 'Annotations/test_link_samples.txt',
                            }

        INPUTS_IDS_DATASET.append('prev_description')
        INPUTS_IDS_MODEL.append('prev_description')

        if '-upperbound' not in DATASET_NAME and '-video' not in DATASET_NAME:
            INPUTS_IDS_DATASET.append('link_index')
            INPUTS_IDS_MODEL.append('link_index')

    # Evaluation params
    METRICS = ['coco']  # Metric used for evaluating model after each epoch (leave empty if only prediction is required)
    EVAL_ON_SETS = ['val', 'test']                 # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []                        # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 0                        # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1                                 # Sets the evaluation frequency (epochs or updates)

    # Search parameters
    SAMPLING = 'max_likelihood'                   # Possible values: multinomial or max_likelihood (recommended)
    TEMPERATURE = 1                               # Multinomial sampling parameter
    if not '-vidtext-embed' in DATASET_NAME:
        BEAM_SEARCH = True                            # Switches on-off the beam search procedure
    else:
        BEAM_SEARCH = False
    BEAM_SIZE = 10                                # Beam size (in case of BEAM_SEARCH == True)
    BEAM_SEARCH_COND_INPUT = 1                    # Index of the conditional input used in beam search (i.e., state_below)
    OPTIMIZED_SEARCH = True                       # Compute annotations only a single time per sample
    NORMALIZE_SAMPLING = False                    # Normalize hypotheses scores according to their length
    ALPHA_FACTOR = .6                             # Normalization according to length**ALPHA_FACTOR
                                                  # (see: arxiv.org/abs/1609.08144)

    # Sampling params: Show some samples during training
    SAMPLE_ON_SETS = ['train', 'val']             # Possible values: 'train', 'val' and 'test'
    N_SAMPLES = 5                                 # Number of samples generated
    START_SAMPLING_ON_EPOCH = 0                   # First epoch where the model will be evaluated
    SAMPLE_EACH_UPDATES = 50                     # Sampling frequency (default 450)

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_icann'        # Select which tokenization we'll apply:
                                                  #  tokenize_basic, tokenize_aggressive, tokenize_soft,
                                                  #  tokenize_icann or tokenize_questions

    FILL = 'end'                                  # whether we fill the 'end' or the 'start' of the sentence with 0s
    TRG_LAN = 'en'                                # Language of the outputs (mainly used for the Meteor evaluator)
    PAD_ON_BATCH = True                           # Whether we take as many timesteps as the longes sequence of the batch
                                                  # or a fixed size (MAX_OUTPUT_TEXT_LEN)

    # Input image parameters
    DATA_AUGMENTATION = False                      # Apply data augmentation on input data (noise on features)
    IMG_FEAT_SIZE = 1024                           # Size of the image features

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0                    # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MAX_OUTPUT_TEXT_LEN = 30                      # Maximum length of the output sequence
                                                  # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = 50                 # Maximum length of the output sequence during test time
    MIN_OCCURRENCES_VOCAB = 0                     # Minimum number of occurrences allowed for the words in the vocabulay.

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    CLASSIFIER_ACTIVATION = 'softmax'

    OPTIMIZER = 'Adam'                            # Optimizer
    LR = 0.001#0.001                                    # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 10.                                  # During training, clip gradients to this norm
    SAMPLE_WEIGHTS = True                         # Select whether we use a weights matrix (mask) for the data outputs
    LR_DECAY = 1                                  # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.95                               # Multiplier used for decreasing the LR

    # Training parameters
    MAX_EPOCH = 50                                # Stop when computed this number of epochs
    BATCH_SIZE = 64                               # ABiViRNet trained with BATCH_SIZE = 64

    HOMOGENEOUS_BATCHES = False                   # Use batches with homogeneous output lengths for every minibatch (Possibly buggy!)
    PARALLEL_LOADERS = 8                          # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = True                             # Turns on/off the early stop protocol
    PATIENCE = 10                                 # We'll stop if the val STOP_METRIC does not improve after this
                                                  # number of evaluations

    if not '-vidtext-embed' in DATASET_NAME:
        STOP_METRIC = 'Bleu_4'                        # Metric for the stop
    else:
        STOP_METRIC = 'accuracy'

    # Model parameters
    MODEL_TYPE = 'TemporallyLinkedVideoDescriptionAtt'       # 'ArcticVideoCaptionWithInit'
                                                    # 'TemporallyLinkedVideoDescriptionNoAtt'
                                                    # 'TemporallyLinkedVideoDescriptionAtt'
                                                    # 'VideoTextEmbedding'

    RNN_TYPE = 'LSTM'                             # RNN unit type ('LSTM' and 'GRU' supported)

    # Input text parameters
    TARGET_TEXT_EMBEDDING_SIZE = 301              # Source language word embedding size (ABiViRNet 301)
    TRG_PRETRAINED_VECTORS = None # DATA_ROOT_PATH + '/Annotations/word2vec.%s.npy' % TRG_LAN                 # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True      # Finetune or not the target word embedding vectors.

    # Encoder configuration
    ENCODER_HIDDEN_SIZE = 717                     # For models with RNN encoder (ABiViRNet 717)
    BIDIRECTIONAL_ENCODER = True                  # Use bidirectional encoder
    N_LAYERS_ENCODER = 1                          # Stack this number of encoding layers
    BIDIRECTIONAL_DEEP_ENCODER = True             # Use bidirectional encoder in all encoding layers


    # Previous sentence encoder
    PREV_SENT_ENCODER_HIDDEN_SIZE = 717           # For models with previous sentence RNN encoder (484)
    BIDIRECTIONAL_PREV_SENT_ENCODER = True        # Use bidirectional encoder
    N_LAYERS_PREV_SENT_ENCODER = 1                # Stack this number of encoding layers
    BIDIRECTIONAL_DEEP_PREV_SENT_ENCODER = True   # Use bidirectional encoder in all encoding layers

    DECODER_HIDDEN_SIZE = 484                     # For models with LSTM decoder (ABiViRNet 484)
    ADDITIONAL_OUTPUT_MERGE_MODE = 'sum'          # Merge mode for the skip connections
    WEIGHTED_MERGE = False       # Wether we want to apply a conventional or a weighted merge

    IMG_EMBEDDING_LAYERS = []  # FC layers for visual embedding
                               # Here we should specify the activation function and the output dimension
                               # (e.g IMG_EMBEDDING_LAYERS = [('linear', 1024)]

    # Fully-Connected layers for initializing the first RNN state
    #       Here we should only specify the activation function of each layer
    #       (as they have a potentially fixed size)
    #       (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']

    # Additional Fully-Connected layers's sizes applied before softmax.
    #       Here we should specify the activation function and the output dimension
    #       (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu', 400), ('relu', 200)])
    DEEP_OUTPUT_LAYERS = []

    # Regularizers
    WEIGHT_DECAY = 1e-4                           # L2 regularization
    RECURRENT_WEIGHT_DECAY = 0.                   # L2 regularization in recurrent layers

    USE_DROPOUT = False                           # Use dropout
    DROPOUT_P = 0.5                               # Percentage of units to drop

    USE_RECURRENT_DROPOUT = False                 # Use dropout in recurrent layers # DANGEROUS!
    RECURRENT_DROPOUT_P = 0.5                     # Percentage of units to drop in recurrent layers

    USE_NOISE = True                             # Use gaussian noise during training
    NOISE_AMOUNT = 0.01                           # Amount of noise

    USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

    USE_PRELU = False                             # use PReLU activations as regularizer
    USE_L2 = False                                # L2 normalization on the features

    # Results plot and models storing parameters
    EXTRA_NAME = 'vidtext_embed'                    # This will be appended to the end of the model name
    MODEL_NAME = DATASET_NAME + '_' + MODEL_TYPE +\
                 '_txtemb_' + str(TARGET_TEXT_EMBEDDING_SIZE) + \
                 '_imgemb_' + '_'.join([layer[0] for layer in IMG_EMBEDDING_LAYERS]) + \
                 '_lstmenc_' + str(ENCODER_HIDDEN_SIZE) + \
                 '_lstm_' + str(DECODER_HIDDEN_SIZE) + \
                 '_additional_output_mode_' + str(ADDITIONAL_OUTPUT_MERGE_MODE) + \
                 '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                 '_' + OPTIMIZER + '_decay_' + str(LR_DECAY) + '-' + str(LR_GAMMA)

    MODEL_NAME += EXTRA_NAME

    # Name and location of the pre-trained model (only if RELOAD > 0)
    PRE_TRAINED_MODELS = ['MSVD_best_model']#['MSVD_best_model', '1BillionWords'] # default: MODEL_NAME
    PRE_TRAINED_MODEL_STORE_PATHS = map(lambda x: 'trained_models/' + x  + '/', PRE_TRAINED_MODELS) if isinstance(PRE_TRAINED_MODELS, list) else 'trained_models/'+PRE_TRAINED_MODELS+'/'
    LOAD_WEIGHTS_ONLY = True                           # Load weights of pre-trained model or complete Model_Wrapper instance
    # Layers' mapping from old to new model if LOAD_WEIGHTS_ONLY
    #   You can check the layers of a model with [layer.name for layer in model_wrapper.model.layers]
    if '-video' in DATASET_NAME:
        LAYERS_MAPPING = [{'bidirectional_encoder': 'bidirectional_encoder_LSTM',
                           'bidirectional_encoder': 'prev_desc_emb_bidirectional_encoder_LSTM',
                          'initial_state': 'initial_state',
                          'initial_memory': 'initial_memory',
                          'attlstmcond_1': 'decoder_AttLSTMCond2Inputs',  # 'decoder_AttLSTMCond',
                          'target_word_embedding': 'target_word_embedding',
                          'logit_ctx': 'logit_ctx',
                          'logit_lstm': 'logit_lstm',
                          'description': 'description'
                          }
                        ]
    else:
        LAYERS_MAPPING = [{'bidirectional_encoder': 'bidirectional_encoder_LSTM',
                      'initial_state': 'initial_state',
                      'initial_memory': 'initial_memory',
                      'attlstmcond_1': 'decoder_AttLSTMCond2Inputs',  # 'decoder_AttLSTMCond',
                      #'target_word_embedding': 'target_word_embedding',
                      'logit_ctx': 'logit_ctx',
                      'logit_lstm': 'logit_lstm',
                      #'description': 'description'
                      },
                      {'bidirectional_encoder_LSTM': 'prev_desc_emb_bidirectional_encoder_LSTM', #'prev_desc_emb_encoder_LSTM',
                      'target_word_embedding': 'target_word_embedding',
                      'decoder_AttLSTMCond': 'decoder_AttLSTMCond2Inputs', #'decoder_AttLSTMCond',
                      'target_text': 'description'
                      }
                    ]

    STORE_PATH = 'trained_models/' + MODEL_NAME  + '/' # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'                   # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'list'                        # 'list' or 'vqa'
    VERBOSE = 1                                        # Vqerbosity level
    RELOAD = [2] #[2, 0]                                        # If 0 start training from scratch, otherwise the model
                                                       # Saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True                             # Build again or use stored instance
    MODE = 'training'                                  # 'training' or 'sampling' (if 'sampling' then RELOAD must
                                                       # be greater than 0 and EVAL_ON_SETS will be used)
    SAMPLING_RELOAD_EPOCH = False
    SAMPLING_RELOAD_POINT = 300
    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False  # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False  # force building a new vocabulary from the training samples applicable if RELOAD > 1

    # ============================================
    parameters = locals().copy()
    return parameters
