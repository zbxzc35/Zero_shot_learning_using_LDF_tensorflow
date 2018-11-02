# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：configuration of zero-shot-learning baseline using resnet
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


class FLAGS(object):
    # dictory configuration

    attrs_per_class_dir = \
        '../../data/Dataset_train_total/attributes_per_class.txt'
    img_dir = \
        '../../data/Dataset_train_total/train/'
    train_file = \
        '../../data/Dataset_train_total/train.txt'
    label_list = '../../data/Dataset_train_total/label_list.txt'
    class_wordembeddings = '../../data/Dataset_train_total/class_wordembeddings.txt'

    ''' 
    # configuration of dictory in part A
    attrs_per_class_dir = \
        '../../data/DatasetA_train_20180813/attributes_per_class.txt'
    img_dir = \
        '../../data/DatasetA_train_20180813/train/'
    train_file = \
        '../../data/DatasetA_train_20180813/train.txt'
    '''

    # model configuration
    weight_decay = 0.0002
    num_residual_blocks = 25

    # pattern configuration
    img_type = 'RGB'
    use_gpu = True

    # augment configuration
    if_crop_augment = True
    crop_scale = 0.7

    if_flip_augment = True

    if_size_augment = True
    min_compress_ratio = 0.8
    max_compress_ratio = 1.35

    if_rotate_augment = True
    max_rotation = 180

    if_color_augment = False
    color_augment_choices = [0.6, 0.8, 1.2, 1.4]

    # input configuration
    if if_crop_augment:
        size_before_crop = 256
    else:
        size_before_crop = 224

    img_size = 224
    img_width = 224
    img_height = 224
    img_depth = 3
    normalize = True

    # train configuration
    training_epoch = 200
    batch_size = 64
    attribute_label_cnt = 30
    num_class = 285

    # learning rate configuration
    dropout_keep_prob = 0.5
    learning_rate = 0.0005
    lr_decay_rate = 0.96
    lr_decay_step = 5000
    fixed_lr = 0.0001
    square_optimizer = 'Adam'

    # triplet loss configuration
    triplet_strategy = 'batch_hard'
    squared = False
    margin = 1.0
    learning_rate_triplet = 0.0001
    lr_decay_rate_triplet = 0.96
    lr_decay_step_triplet = 4000
    triplet_optimizer = 'Adam'

    # train SA to LA regression configuration
    regression_training_epoch = 100
    fc_lr = 0.001
    lambda1 = 1.0

    # subprocess configuration
    validation_interval = 20

    # test configuration
    '''
    # configuration of dictory in part A
    test_img_dir = '../../data/DatasetA_test_20180813/DatasetA_test/test'
    test_file = '../../data/DatasetA_test_20180813/DatasetA_test/image.txt'
    '''
    test_img_dir = '../../data/DatasetB_20180919/test'
    test_file = '../../data/DatasetB_20180919/image.txt'

    # params that are set usually
    network_def = 'resnet_50'
    version = 'v2'

    gpu_id = '0'

    is_train = True
    pretrained_model = False

    batch_size_test = 64
    aug_num = 10

    # experiment setting
    experiment_id = '1'
