# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：Loss definition of LDF
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


import tensorflow as tf

from config import FLAGS
from triplet_loss import *


def build_loss_softmax_with_score(logits, gt_onehot_labels, whole_attr_labels, variable_to_train=None, freeze=False, optimizer='Adam'):
    logits = tf.reshape(logits, [-1, FLAGS.attribute_label_cnt])
    # logits = tf.sigmoid(logits)
    print('logits:', logits)

    # (16,30) x (230,30).T output (16, 230), compatibility score
    whole_inner_product = tf.matmul(logits, tf.transpose(whole_attr_labels))

    with tf.name_scope('softmax_loss_with_score'):
        softmax_loss_with_score = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=whole_inner_product, labels=gt_onehot_labels))
        tf.summary.scalar('softmax_loss_with_score', softmax_loss_with_score)

    with tf.variable_scope('train_softmax_loss_with_score'):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step=global_step,
                                        decay_rate=FLAGS.lr_decay_rate,
                                        decay_steps=FLAGS.lr_decay_step)
        tf.summary.scalar('learning_rate_softmax_loss_with_score', lr)

        if freeze:
            optimizer_2 = tf.train.AdamOptimizer(lr)
            train_op = tf.contrib.slim.learning.create_train_op(softmax_loss_with_score, optimizer_2, variables_to_train=variable_to_train)
        else:
            train_op = tf.contrib.layers.optimize_loss(loss=softmax_loss_with_score,
                                                       global_step=global_step,
                                                       learning_rate=lr,
                                                       optimizer=optimizer)

    return softmax_loss_with_score, train_op


def build_triplet_loss(logits, labels, margin, squared=False, triplet_strategy='batch_hard'):
    y_conv = tf.reshape(logits, [-1, FLAGS.attribute_label_cnt])

    # Define triplet loss
    with tf.name_scope('triplet_loss'):
        if triplet_strategy == "batch_all":
            loss, fraction = batch_all_triplet_loss(labels=labels, embeddings=y_conv, margin=margin, squared=squared)
            tf.summary.scalar('triplet_loss', loss)
        elif triplet_strategy == "batch_hard":
            loss = batch_hard_triplet_loss(labels=labels, embeddings=y_conv, margin=margin, squared=squared)
            tf.summary.scalar('triplet_loss', loss)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))

    train_op = tf.train.AdamOptimizer(FLAGS.fixed_lr).minimize(loss)
    return loss, train_op


def build_triplet_loss_2(logits, labels, margin, squared=False, triplet_strategy='batch_hard'):
    y_conv = tf.reshape(logits, [-1, FLAGS.attribute_label_cnt])

    # Define triplet loss
    with tf.name_scope('triplet_loss'):
        if triplet_strategy == "batch_all":
            loss, fraction = batch_all_triplet_loss(labels=labels, embeddings=y_conv, margin=margin, squared=squared)
            tf.summary.scalar('triplet_loss', loss)
        elif triplet_strategy == "batch_hard":
            loss = batch_hard_triplet_loss(labels=labels, embeddings=y_conv, margin=margin, squared=squared)
            tf.summary.scalar('triplet_loss', loss)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))
    with tf.variable_scope('train_triplet_loss'):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(FLAGS.learning_rate_triplet,
                                        global_step=global_step,
                                        decay_rate=FLAGS.lr_decay_rate_triplet,
                                        decay_steps=FLAGS.lr_decay_step_triplet)
        tf.summary.scalar('learning_rate_triplet', lr)

        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                   global_step=global_step,
                                                   learning_rate=lr,
                                                   optimizer=FLAGS.triplet_optimizer)
        return loss, train_op


def build_triplet_loss_3(logits, labels, margin, squared=False, triplet_strategy='batch_hard', freeze=False, variable_to_train=None):
    y_conv = tf.reshape(logits, [-1, FLAGS.attribute_label_cnt])

    # Define triplet loss
    with tf.name_scope('triplet_loss'):
        if triplet_strategy == "batch_all":
            loss, fraction = batch_all_triplet_loss(labels=labels, embeddings=y_conv, margin=margin, squared=squared)
            tf.summary.scalar('triplet_loss', loss)
        elif triplet_strategy == "batch_hard":
            loss = batch_hard_triplet_loss(labels=labels, embeddings=y_conv, margin=margin, squared=squared)
            tf.summary.scalar('triplet_loss', loss)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))
    with tf.variable_scope('train_triplet_loss'):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(FLAGS.learning_rate_triplet,
                                        global_step=global_step,
                                        decay_rate=FLAGS.lr_decay_rate_triplet,
                                        decay_steps=FLAGS.lr_decay_step_triplet)
        tf.summary.scalar('learning_rate_triplet', lr)

        if freeze:
            optimizer_2 = tf.train.AdamOptimizer(lr)
            train_op = tf.contrib.slim.learning.create_train_op(loss, optimizer_2, variables_to_train=variable_to_train)
        else:
            train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                       global_step=global_step,
                                                       learning_rate=lr,
                                                       optimizer=FLAGS.triplet_optimizer)
        return loss, train_op


def build_multi_loss(logits, attr_labels, num_labels, margin, squared=False, triplet_strategy='batch_hard'):
    y_conv = tf.reshape(logits, [-1, 2 * FLAGS.attribute_label_cnt])
    # y_conv = tf.nn.sigmoid(y_conv)
    y_conv_square = y_conv[:, 0:FLAGS.attribute_label_cnt]
    y_conv_triplet = y_conv[:, FLAGS.attribute_label_cnt:2 * FLAGS.attribute_label_cnt]

    print(y_conv, y_conv_square, y_conv_triplet)
    # Define square loss
    with tf.name_scope('square_loss'):
        square_loss = tf.reduce_mean(tf.square(y_conv_square - attr_labels))
        tf.summary.scalar('square_loss', square_loss)

    # Define triplet loss
    with tf.name_scope('triplet_loss'):
        if triplet_strategy == "batch_all":
            triplet_loss, fraction = batch_all_triplet_loss(labels=num_labels,
                                                            embeddings=y_conv_triplet,
                                                            margin=margin,
                                                            squared=squared)
            tf.summary.scalar('triplet_loss', triplet_loss)
        elif triplet_strategy == "batch_hard":
            triplet_loss = batch_hard_triplet_loss(labels=num_labels,
                                                   embeddings=y_conv_triplet,
                                                   margin=margin,
                                                   squared=squared)
            tf.summary.scalar('triplet_loss', triplet_loss)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))

    with tf.name_scope('multi_loss'):
        multi_loss = square_loss + triplet_loss
        tf.summary.scalar('multi_loss', multi_loss)

    with tf.variable_scope('train_multi_loss'):
        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step=global_step,
                                        decay_rate=FLAGS.lr_decay_rate,
                                        decay_steps=FLAGS.lr_decay_step)
        tf.summary.scalar('learning_rate_multi', lr)

        train_op = tf.contrib.layers.optimize_loss(loss=multi_loss,
                                                   global_step=global_step,
                                                   learning_rate=lr,
                                                   optimizer=FLAGS.triplet_optimizer)
        return multi_loss, train_op


def build_multi_loss_2(logits, gt_onehot_labels, whole_attr_labels, num_labels, margin, squared=False, triplet_strategy='batch_hard', optimizer='Adam'):
    y_conv = tf.reshape(logits, [-1, 2 * FLAGS.attribute_label_cnt])

    y_conv_softmax = y_conv[:, 0:FLAGS.attribute_label_cnt]
    y_conv_triplet = y_conv[:, FLAGS.attribute_label_cnt:2 * FLAGS.attribute_label_cnt]

    print(y_conv, y_conv_softmax, y_conv_triplet)
    # build softmax loss
    # (16,30) x (230,30).T output (16, 230), compatibility score
    whole_inner_product = tf.matmul(y_conv_softmax, tf.transpose(whole_attr_labels))

    with tf.name_scope('softmax_loss_with_score'):
        softmax_loss_with_score = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=whole_inner_product, labels=gt_onehot_labels))
        tf.summary.scalar('softmax_loss_with_score', softmax_loss_with_score)

    # Define triplet loss
    with tf.name_scope('triplet_loss'):
        if triplet_strategy == "batch_all":
            triplet_loss, fraction = batch_all_triplet_loss(labels=num_labels,
                                                            embeddings=y_conv_triplet,
                                                            margin=margin,
                                                            squared=squared)
            tf.summary.scalar('triplet_loss', triplet_loss)
        elif triplet_strategy == "batch_hard":
            triplet_loss = batch_hard_triplet_loss(labels=num_labels,
                                                   embeddings=y_conv_triplet,
                                                   margin=margin,
                                                   squared=squared)
            tf.summary.scalar('triplet_loss', triplet_loss)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))

    with tf.name_scope('multi_loss'):
        multi_loss = softmax_loss_with_score + triplet_loss
        tf.summary.scalar('multi_loss', multi_loss)

    with tf.variable_scope('train_multi_loss'):
        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step=global_step,
                                        decay_rate=FLAGS.lr_decay_rate,
                                        decay_steps=FLAGS.lr_decay_step)
        tf.summary.scalar('learning_rate_multi', lr)

        train_op = tf.contrib.layers.optimize_loss(loss=multi_loss,
                                                   global_step=global_step,
                                                   learning_rate=lr,
                                                   optimizer=optimizer)
        return multi_loss, train_op


def build_multi_loss_3(logits, gt_onehot_labels, whole_attr_labels, num_labels, margin, squared=False,
                       triplet_strategy='batch_hard', optimizer='Adam', freeze=False, variable_to_train=None):
    y_conv = tf.reshape(logits, [-1, 2 * FLAGS.attribute_label_cnt])

    y_conv_softmax = y_conv[:, 0:FLAGS.attribute_label_cnt]
    y_conv_triplet = y_conv[:, FLAGS.attribute_label_cnt:2 * FLAGS.attribute_label_cnt]

    print(y_conv, y_conv_softmax, y_conv_triplet)

    # build softmax loss
    # (16,30) x (230,30).T output (16, 230), compatibility score
    whole_inner_product = tf.matmul(y_conv_softmax, tf.transpose(whole_attr_labels))

    with tf.name_scope('softmax_loss_with_score'):
        softmax_loss_with_score = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=whole_inner_product, labels=gt_onehot_labels))
        tf.summary.scalar('softmax_loss_with_score', softmax_loss_with_score)

    # Define triplet loss
    with tf.name_scope('triplet_loss'):
        if triplet_strategy == "batch_all":
            triplet_loss, fraction = batch_all_triplet_loss(labels=num_labels,
                                                            embeddings=y_conv_triplet,
                                                            margin=margin,
                                                            squared=squared)
            tf.summary.scalar('triplet_loss', triplet_loss)
        elif triplet_strategy == "batch_hard":
            triplet_loss = batch_hard_triplet_loss(labels=num_labels,
                                                   embeddings=y_conv_triplet,
                                                   margin=margin,
                                                   squared=squared)
            tf.summary.scalar('triplet_loss', triplet_loss)
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(triplet_strategy))

    with tf.name_scope('multi_loss'):
        multi_loss = softmax_loss_with_score + triplet_loss
        tf.summary.scalar('multi_loss', multi_loss)

    with tf.variable_scope('train_multi_loss'):
        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step=global_step,
                                        decay_rate=FLAGS.lr_decay_rate,
                                        decay_steps=FLAGS.lr_decay_step)
        tf.summary.scalar('learning_rate_multi', lr)

        if freeze:
            optimizer_2 = tf.train.AdamOptimizer(lr)
            train_op = tf.contrib.slim.learning.create_train_op(multi_loss, optimizer_2,
                                                                variables_to_train=variable_to_train)
        else:
            train_op = tf.contrib.layers.optimize_loss(loss=multi_loss,
                                                       global_step=global_step,
                                                       learning_rate=lr,
                                                       optimizer=optimizer)
        return multi_loss, train_op