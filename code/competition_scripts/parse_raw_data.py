# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：parse raw data of ZhijiangLab Cup zero-shot picture recognition competition
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import cv2

from config import FLAGS


def parse_train_image2represent_label_map(filepath):
    image2represent_label = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            image_name = line.split('	')[0]
            image_represent_label = line.split('	')[1].replace('\n', '')
            image2represent_label[image_name] = image_represent_label

    return image2represent_label


def parse_test_image_list(filepath):
    test_images = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            test_images.append(line.replace('\n', ''))

    return test_images


def parse_represent_label2true_label_map(filepath):
    represent_label2true_label = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            represent_label = line.split('	')[0]
            true_label = line.split('	')[1].replace('\n', '')
            represent_label2true_label[represent_label] = true_label

    return represent_label2true_label


def parse_attribute_list(filepath):
    attributes = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            attr_item = line.split('	')[1].replace('\n', '')
            attributes.append(attr_item)

    return attributes


def parse_attribute_per_class(filepath):
    attribute_per_class = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            class_represent = line.replace('\n', '').split('	')[0]
            attribute_vec = line.replace('\n', '').split('	')[1:]
            attribute_vec_float = []
            for item in attribute_vec:
                attribute_vec_float.append(float(item))
            attribute_per_class[class_represent] = attribute_vec_float

    return attribute_per_class


def parse_word_embedding_per_class(filepath):
    word_embedding_per_class = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            class_true_label = line.replace('\n', '').split(' ')[0]
            word_embedding_vec = line.replace('\n', '').split(' ')[1:]
            word_embedding_vec_float = []
            for item in word_embedding_vec:
                word_embedding_vec_float.append(float(item))
                word_embedding_per_class[class_true_label] = word_embedding_vec_float

    return word_embedding_per_class


def parse_repre_label2num_label_map(filepath):
    repre_label2num_label_map = {}
    i = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            class_represent = line.replace('\n', '').split('	')[0]
            repre_label2num_label_map[class_represent] = i
            i += 1
    print('i: ', i)
    return repre_label2num_label_map


def parse_repre_label2one_hot_map(filepath):
    repre_label2one_hot_map = {}

    class_represent_list = []
    whole_attr_list = []
    integer_encodes = []
    i = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            class_represent = line.replace('\n', '').split('	')[0]
            class_represent_list.append(class_represent)

            attribute_vec = line.replace('\n', '').split('	')[1:]
            attribute_vec_float = []
            for item in attribute_vec:
                attribute_vec_float.append(float(item))
            whole_attr_list.append(attribute_vec_float)

            integer_encodes.append(i)
            i += 1

    print('i: ', i)
    print('class_represent_list', class_represent_list)
    integer_encodes = np.array(integer_encodes)
    print('integer_encodes', integer_encodes)

    whole_attr_np = np.array(whole_attr_list)
    print('whole attr np:', whole_attr_np)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    integer_encodes = integer_encodes.reshape(len(integer_encodes), 1)
    onehot_encodes = onehot_encoder.fit_transform(integer_encodes)
    print('onehot encodes: ', onehot_encodes, onehot_encodes.shape)

    for i in range(len(class_represent_list)):
        repre_label2one_hot_map[class_represent_list[i]] = onehot_encodes[i]
    repre_label2one_hot_map[class_represent_list[i]] = onehot_encodes[i]

    print('Example', class_represent_list[0], integer_encodes[0], whole_attr_np[0], onehot_encodes[0])

    return class_represent_list, whole_attr_np, repre_label2one_hot_map


def read_single_sample(tfrecord_file):
    queue = tf.train.string_input_producer([tfrecord_file], shuffle=True, num_epochs=FLAGS.training_epoch)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)

    features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
                                                                     'label': tf.FixedLenFeature(shape=[], dtype=tf.int64)})
    image = tf.decode_raw(features['image'], out_type=tf.uint8)
    image = tf.reshape(image, shape=[224, 224, 3])

    label = features['label']
    label = tf.cast(label, tf.int32)
    return image, label


def tmp_test():
    '''
    train_image2represent_label_map_filepath = '../../data/DatasetA_train_20180813/train.txt'
    train_image2represent_label_map = parse_train_image2represent_label_map(train_image2represent_label_map_filepath)
    print(train_image2represent_label_map['a6394b0f513290f4651cc46792e5ac86.jpeg'])
    print('Total number of train images:', train_image2represent_label_map)

    represent_label2true_label_map_filepath = \
        '../../data/DatasetA_train_20180813/label_list.txt'
    represent_label2true_label_map = parse_represent_label2true_label_map(represent_label2true_label_map_filepath)
    print(represent_label2true_label_map['ZJL1'])
    print('Total number of classes:', len(represent_label2true_label_map))

    attribute_list_filepath = '../../data/DatasetA_train_20180813/attribute_list.txt'
    attribute_list = parse_attribute_list(attribute_list_filepath)
    print(attribute_list[0])
    print('Total number of attributes:', len(attribute_list))

    represent_label2attribute_vec_map_filepath = \
        '../../data/DatasetA_train_20180813/attributes_per_class.txt'
    represent_label2attribute_vec_map = parse_attribute_per_class(represent_label2attribute_vec_map_filepath)
    print('Total number of classes Attr:',represent_label2attribute_vec_map)
    print('Total number of class and attributes per class:',
          len(represent_label2attribute_vec_map), len(represent_label2attribute_vec_map['ZJL1']))

    represent_label2num_label_map = parse_repre_label2num_label_map(represent_label2attribute_vec_map_filepath)
    print('Total number of classes Num:', represent_label2num_label_map)

    parse_word_embedding_per_class_filepath = \
        '../../data/DatasetA_train_20180813/class_wordembeddings.txt'
    true_label2word_embedding_vec_map = parse_word_embedding_per_class(parse_word_embedding_per_class_filepath)
    print(true_label2word_embedding_vec_map['book'])
    print('Total number of class and word embeddings per class:',
          len(true_label2word_embedding_vec_map), len(true_label2word_embedding_vec_map['book']))

    parse_test_image_list_filepath = '../../data/DatasetA_test_20180813/DatasetA_test/image.txt'
    test_images_list = parse_test_image_list(parse_test_image_list_filepath)
    print(test_images_list[0])
    print('Total number of train images:', len(test_images_list))


    represent_label2attribute_vec_map_filepath = \
        '../../data/DatasetA_train_20180813/attributes_per_class.txt'
    whole_class_repre_list, whole_attr_np, repre_label2one_hot_map = parse_repre_label2one_hot_map(represent_label2attribute_vec_map_filepath)
    '''

    image, label = read_single_sample('../../data/Dataset_train_total/train.tfrecords')
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=1, capacity=5*FLAGS.batch_size,
                                                      min_after_dequeue=2*FLAGS.batch_size)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            image_val, label_val = sess.run([image_batch, label_batch])
            cv2.imshow('imgae', image_val[0])
            cv2.waitKey(1000)
            print(image_val, label_val, image_val.shape, label_val.shape)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tmp_test()