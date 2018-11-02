# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：train combining softmax loss and triplet loss according to LDF
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
slim = tf.contrib.slim

import loss as model
from config import FLAGS
from data_generator import DataGenerator
from data_generator import *
from parse_raw_data import *


def train_multi():
	'''
	Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

	pretrained_model_path_suffix = os.path.join(
		FLAGS.network_def + '_' + FLAGS.version + '_' + 'train_multi' + '_imagesize_' + str(
			FLAGS.img_size) + '_batchsize_' + str(FLAGS.batch_size) + '_experiment_' + FLAGS.experiment_id)
	pretrained_model_save_dir = os.path.join('../../data/results_multi/model_weights', pretrained_model_path_suffix)

	model_path_suffix = os.path.join(
		FLAGS.network_def + '_' + FLAGS.version + '_' + 'train_multi' + '_imagesize_' + str(
			FLAGS.img_size) + '_batchsize_' + str(FLAGS.batch_size) + '_experiment_' + FLAGS.experiment_id)
	train_log_save_dir = os.path.join('../../data/results_multi/logs', model_path_suffix,
	                                  'train')
	test_log_save_dir = os.path.join('../../data/results_multi/logs', model_path_suffix,
	                                 'val')
	model_save_dir = os.path.join('../../data/results_multi/model_weights', model_path_suffix)

	os.system('mkdir -p {}'.format(model_save_dir))
	os.system('mkdir -p {}'.format(train_log_save_dir))
	os.system('mkdir -p {}'.format(test_log_save_dir))

	'''
	Step 2: Create dataset and data generator
	'''
	print('CREATE DIFFERENT DATASETS')
	dataset = DataGenerator(FLAGS.attrs_per_class_dir, FLAGS.img_dir, FLAGS.train_file)
	dataset.generate_set(rand=True, validationRate=0.0)

	# train setp configuration
	train_size = dataset.count_train()
	training_iters_per_epoch = int(train_size / FLAGS.batch_size)
	print("train size: %d, training_iters_per_epoch: %d" % (train_size, training_iters_per_epoch))

	generator = dataset.generator(batchSize=FLAGS.batch_size, norm=FLAGS.normalize, sample='train')
	generator_eval = dataset.generator(batchSize=FLAGS.batch_size, norm=FLAGS.normalize, sample='valid')

	_, whole_attr_np, _ = parse_repre_label2one_hot_map(FLAGS.attrs_per_class_dir)
	# print(whole_attr_np)

	image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])  # [batch, 224, 224, 3]
	whole_label_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.num_class, FLAGS.attribute_label_cnt])  # [230, 30]
	gt_onehot_label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.num_class])  # [batch, 230]
	num_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])  # [batch]
	is_training = tf.placeholder(dtype=tf.bool)

	'''
	Step 3: Build network graph
	'''
	feature, endpoints = resnet_v2.resnet_v2_50(image_placeholder, num_classes=None, reuse=False, is_training=is_training)

	'''
	Step 4: Define variables to restore if have trained convnet parameters
	'''
	checkpoint_exclude_scope = ''
	exclusions = []
	if checkpoint_exclude_scope:
		exclusions = [scope.strip() for scope in checkpoint_exclude_scope.split(',')]
	print('exclusions variables: ', exclusions)
	variable_to_restore = []
	for var in slim.get_model_variables():
		excluded = False
		for exclusion in exclusions:
			if var.op.name.startswith(exclusion):
				excluded = True
		if not excluded:
			variable_to_restore.append(var)
	print('variable_to_restore', variable_to_restore)

	feature = tf.squeeze(feature, axis=[1, 2])
	print('feature shape:', feature)
	feature = slim.dropout(feature, keep_prob=0.5)
	logits = slim.fully_connected(feature, num_outputs=2 * FLAGS.attribute_label_cnt, activation_fn=None)
	print('logits shape', logits)

	total_variables = slim.get_model_variables()
	print('total_variables', total_variables)

	variable_to_train_if_freeze = [var for var in total_variables if var not in variable_to_restore]  # only train fc
	print('variable to train if freeze', variable_to_train_if_freeze)

	'''
	Step 5: Define multi loss according to LDF
	'''
	freeze = False
	if freeze:
		loss, train = model.build_multi_loss_3(logits, gt_onehot_label_placeholder, whole_label_placeholder,
		                                       num_label_placeholder, FLAGS.margin, FLAGS.squared,
		                                       FLAGS.triplet_strategy, optimizer='Adam', freeze=freeze,
		                                       variable_to_train=variable_to_train_if_freeze)
	else:
		loss, train = model.build_multi_loss_3(logits, gt_onehot_label_placeholder, whole_label_placeholder,
		                                       num_label_placeholder, FLAGS.margin, FLAGS.squared,
		                                       FLAGS.triplet_strategy, optimizer='Adam')

	'''
	Step 6: Training
	'''
	total_start_time = time.time()
	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
		# Create tensorboard
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)
		validation_writer = tf.summary.FileWriter(test_log_save_dir, sess.graph)

		# Create model saver
		saver_restore = tf.train.Saver(var_list=variable_to_restore)
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		# Restore pretrained weights
		if FLAGS.pretrained_model:
			pretrained_model = model_save_dir
			print('load checkpoint of:', pretrained_model)
			checkpoint = tf.train.get_checkpoint_state(pretrained_model)
			# 获取最新保存的模型检查点文件
			ckpt = checkpoint.model_checkpoint_path
			saver.restore(sess, ckpt)
			# check weights
			for variable in tf.trainable_variables():
				with tf.variable_scope('', reuse=True):
					var = tf.get_variable(variable.name.split(':0')[0])
					print(variable.name, np.mean(sess.run(var)))

		# Start training
		global_step = 0
		for epoch in range(152, FLAGS.training_epoch):
			for step in range(training_iters_per_epoch):
				# Train start
				image_data, attr_labels, num_labels, onehot_labels = next(generator)
				batch_start_time = time.time()
				global_step = step + epoch * (training_iters_per_epoch)

				summary, loss_result, _ = \
					sess.run([merged, loss, train], feed_dict={image_placeholder: image_data,
					                                           whole_label_placeholder: whole_attr_np,
					                                           num_label_placeholder: num_labels,
					                                           gt_onehot_label_placeholder: onehot_labels,
					                                           is_training:True})

				train_writer.add_summary(summary, global_step)

				if step % 10 == 0:
					print('[%s][training][epoch %d, step %d / %d exec %.2f seconds]  loss : %3.10f' %
					      (time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, step, training_iters_per_epoch, (time.time() - batch_start_time), loss_result))

			# Save models for one epoch
			saver.save(sess=sess, save_path=model_save_dir + '/' + FLAGS.network_def.split('.py')[0], global_step=(global_step + 1))
			print('\nModel checkpoint saved for one epoch...\n')

		# Save models for total training process
		saver.save(sess=sess, save_path=model_save_dir + '/' + FLAGS.network_def.split('.py')[0], global_step=(global_step + 1))
		print('\nModel checkpoint saved for total train...\n')

	print('Training done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))
	train_writer.close()
	sess.close()


if __name__ == '__main__':
	train_multi()