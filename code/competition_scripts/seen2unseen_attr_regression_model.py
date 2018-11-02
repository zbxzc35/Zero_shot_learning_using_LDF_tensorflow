# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：the model of regression, to model the relationship between the given seen attr to useen attr
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


import tensorflow as tf

from config import FLAGS


def get_weight_with_regularizer(shape, lambda1):
	var = tf.get_variable('weight', shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
	return var


def regression_model(input_placeholder, num_seen, num_useen):
	w = get_weight_with_regularizer([num_useen, num_seen], FLAGS.lambda1)
	pred_y = tf.matmul(w, input_placeholder)
	# print('pred_y', pred_y.shape)
	return pred_y


def regression_loss(pred_y, output_placeholder):
	l2_loss = tf.reduce_sum(tf.square(output_placeholder - pred_y))
	tf.summary.scalar('l2_loss', l2_loss)
	tf.add_to_collection('losses', l2_loss)

	l2_loss_with_regularizer = tf.add_n(tf.get_collection('losses'))
	tf.summary.scalar('l2_loss_with_regularizer', l2_loss_with_regularizer)

	train_l2_loss_op = tf.train.AdamOptimizer(FLAGS.fc_lr).minimize(l2_loss)
	train_l2_loss_with_regularizer_op = tf.train.AdamOptimizer(FLAGS.fc_lr).minimize(l2_loss_with_regularizer)
	return l2_loss, l2_loss_with_regularizer, train_l2_loss_op, train_l2_loss_with_regularizer_op