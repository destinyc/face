class Config:
    models_base_dir = 'model/center_softmax'
    gpu_memory_fraction = 0.80
    pretrained_model = None
    data_dir = '../data/webface'
    model_def = 'inception_resnet_v1'
    max_nrof_epochs = 70
    batch_size = 90
    image_size = 160
    epoch_size = 1000
    embedding_size = 128

    test_number = 5000

    random_crop = False
    random_flip = False
    random_rotate = False
    keep_prob = 1.0
    weight_decay = 0.0
    center_loss_weight = 1
    center_loss_alfa = 0.95
    optimizer = 'ADAGRAD'
    learning_rate = 0.1
    learning_rate_decay_epochs = 100
    learning_rate_decay_factor = 1.0
    moving_average_decay = 0.9999
    nrof_preprocess_threads = 4

    model = 'model/center_softmax/71_epoch'




