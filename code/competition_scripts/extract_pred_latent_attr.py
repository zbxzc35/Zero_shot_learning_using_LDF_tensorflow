# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：use trained LDF model to extract predicted latent attr
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
slim = tf.contrib.slim

from config import FLAGS
from data_generator import *
from parse_raw_data import *


def extract_pred_latent_attr():
	'''
		Step 1: Create dirs for saving models and logs
	'''
	print('Start extract predicted latent attr')
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
	model_path_suffix = os.path.join(
		FLAGS.network_def + '_' + FLAGS.version + '_' + 'train_multi' + '_imagesize_' + str(
			FLAGS.img_size) +
		'_batchsize_' + str(FLAGS.batch_size) + '_experiment_' + FLAGS.experiment_id)
	model_save_dir = os.path.join('../../data/results_multi/model_weights', model_path_suffix)

	print('Extract pred attr of train set: ' + model_path_suffix + ' ...')
	la_save_dir_train = os.path.join('../../data/results_extract_la' + '/train', model_path_suffix)
	la_save_dir_test = os.path.join('../../data/results_extract_la' + '/test', model_path_suffix)

	os.system('mkdir -p {}'.format(la_save_dir_train))
	os.system('mkdir -p {}'.format(la_save_dir_test))

	'''
	Step 2: Create dataset and data generator
	'''
	test_set_train = []
	with open(FLAGS.train_file, 'r') as f:
		for line in f.readlines():
			image_name = line.split('	')[0]
			test_set_train.append(image_name)
	print('READING LABELS OF TRAIN DATA')
	print('Train total num:', len(test_set_train))
	test_size_train = len(test_set_train)

	test_set_test = parse_test_image_list(FLAGS.test_file)
	print('Test total num:', len(test_set_test))
	test_size_test = len(test_set_test)

	'''
	Step 3: Build network graph
	'''
	_, whole_attr_np, _ = parse_repre_label2one_hot_map(FLAGS.attrs_per_class_dir)
	# print(whole_attr_np)

	with tf.Graph().as_default() as g3:
		image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_width,
																	FLAGS.img_depth])  # [batch, 224, 224, 3]
		is_training = tf.placeholder(dtype=tf.bool)

		feature, endpoints = resnet_v2.resnet_v2_50(image_placeholder, num_classes=None, reuse=False,
													is_training=is_training)
		feature = tf.squeeze(feature, axis=[1, 2])
		print('feature shape:', feature)
		feature = slim.dropout(feature, keep_prob=1)
		final_logits = slim.fully_connected(feature, num_outputs=2 * FLAGS.attribute_label_cnt, activation_fn=None)
		print('logits shape', final_logits)

	'''
	Step 4: Testing
	'''
	total_start_time = time.time()

	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True), graph=g3) as sess:
		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		if True:
			# Restore pretrained weights
			pretrained_model = model_save_dir
			print('load checkpoint of ', pretrained_model)
			checkpoint = tf.train.get_checkpoint_state(pretrained_model)
			ckpt = checkpoint.model_checkpoint_path  # 获取最新保存的模型检查点文件
			saver.restore(sess, ckpt)
			for variable in tf.trainable_variables():  # check weights
				with tf.variable_scope('', reuse=True):
					var = tf.get_variable(variable.name.split(':0')[0])
					print(variable.name, np.mean(sess.run(var)))

		# Extract train la start
		step = 0
		train_la_dict = {}
		while True:
			if step < test_size_train:
				image_name = test_set_train[step: step + FLAGS.batch_size_test]
				print('IMAGE_NAME', image_name)
				step = step + FLAGS.batch_size_test
				image_num = len(image_name)
				print('image num', image_num)

				image_data = np.zeros((image_num, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth), dtype=np.float32)
				for i in range(image_num):
					img = open_img(is_train=True, name=image_name[i], size=FLAGS.img_size,
								   color=FLAGS.img_type)

					if FLAGS.normalize:
						image_data[i, :, :, :] = img.astype(np.float32) / 255.0
					else:
						image_data[i, :, :, :] = img.astype(np.float32)

				batch_start_time = time.time()

				pred_logits = sess.run([final_logits], feed_dict={image_placeholder: image_data, is_training: False})
				pred_logits = np.array(pred_logits).squeeze()

				for i in range(image_num):
					train_la_dict[image_name[i]] = pred_logits[i]
				print('[%s][testing %d][step %d / %d exec %.2f seconds]'
				      % (time.strftime("%Y-%m-%d %H:%M:%S"), image_num, step, test_size_train, (time.time() - batch_start_time)))
			else:
				break
		print('train_la_dict: ', len(train_la_dict))
		np.savez(os.path.join(la_save_dir_train, 'train_la.npz'), dict=train_la_dict)
		train_la_dict_2 = np.load(os.path.join(la_save_dir_train, 'train_la.npz'))['dict'][()]
		print(len(train_la_dict_2), train_la_dict_2['7c382f330bd76982761f1a9191e9db0e.jpeg'])
		print('Extract train set done.')
		print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))

		# Extract test la start
		step = 0
		test_la_dict = {}
		while True:
			if step < test_size_test:
				image_name = test_set_test[step: step + FLAGS.batch_size_test]
				print('IMAGE_NAME', image_name)
				step = step + FLAGS.batch_size_test
				image_num = len(image_name)
				print('image num', image_num)

				image_data = np.zeros((image_num, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth), dtype=np.float32)
				for i in range(image_num):
					img = open_img(is_train=False, name=image_name[i], size=FLAGS.img_size,
								   color=FLAGS.img_type)

					if FLAGS.normalize:
						image_data[i, :, :, :] = img.astype(np.float32) / 255.0
					else:
						image_data[i, :, :, :] = img.astype(np.float32)

				batch_start_time = time.time()

				pred_logits = sess.run([final_logits], feed_dict={image_placeholder: image_data, is_training: False})
				pred_logits = np.array(pred_logits).squeeze()

				for i in range(image_num):
					test_la_dict[image_name[i]] = pred_logits[i]

				print('[%s][testing %d][step %d / %d exec %.2f seconds]'
				      % (time.strftime("%Y-%m-%d %H:%M:%S"), image_num, step, test_size_test,
				         (time.time() - batch_start_time)))
			else:
				break

		print('test_la_dict: ', len(test_la_dict))
		np.savez(os.path.join(la_save_dir_test, 'test_la.npz'), dict=test_la_dict)
		test_la_dict_2 = np.load(os.path.join(la_save_dir_test, 'test_la.npz'))['dict'][()]
		print(len(test_la_dict_2), test_la_dict_2['0003ae092034aa69da9782b2a3b4a15a.jpg'])
		print('Extract test set done.')
		print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))

		sess.close()

if __name__ == '__main__':
	extract_pred_latent_attr()