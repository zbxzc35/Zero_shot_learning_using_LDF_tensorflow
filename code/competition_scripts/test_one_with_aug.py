# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：test process using one image with augmentation
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


def test_one_with_aug_multi():
	'''
	Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

	pretrained_model_path_suffix = os.path.join(
		FLAGS.network_def + '_' + FLAGS.version + '_' + 'train_multi' + '_imagesize_' + str(
			FLAGS.img_size) +
		'_batchsize_' + str(FLAGS.batch_size) + '_experiment_' + FLAGS.experiment_id)
	pretrained_model_save_dir = os.path.join('../../data/results_multi/model_weights',
	                                         pretrained_model_path_suffix)

	print('Test_one_with_aug_multi: ' + pretrained_model_save_dir + ' ...')

	test_save_dir = os.path.join('../../submit/results_multi/test_B_with_aug', pretrained_model_path_suffix)
	os.system('mkdir -p {}'.format(test_save_dir))

	'''
	Step 2: Create dataset and data generator
	'''
	test_set = parse_test_image_list(FLAGS.test_file)

	# test setp configuration
	test_size = len(test_set)

	image_placeholder = tf.placeholder(dtype=tf.float32,
	                                   shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
	is_training = tf.placeholder(dtype=tf.bool)

	'''
	Step 3: Build network graph
	'''
	# logits = model.inference(image_placeholder, FLAGS.num_residual_blocks, reuse=False)
	feature, endpoints = resnet_v2.resnet_v2_50(image_placeholder, num_classes=None, reuse=False, is_training=is_training)
	feature = tf.squeeze(feature, axis=[1, 2])
	print('feature shape:', feature)
	feature = slim.dropout(feature, keep_prob=1)
	final_logits = slim.fully_connected(feature, num_outputs=2*FLAGS.attribute_label_cnt, activation_fn=None)
	print('logits shape', final_logits)

	'''
	Step 4: Testing
	'''
	total_start_time = time.time()

	represent_label2attribute_vec_map = parse_attribute_per_class(FLAGS.attrs_per_class_dir)
	print('represent_label2attribute_vec_map: ', len(represent_label2attribute_vec_map))

	repre_label_list = []
	attr_vec_list = []
	for repre_label in represent_label2attribute_vec_map.keys():
		# print('REPER_LABEL', repre_label)
		repre_label_list.append(repre_label)
		attr_vec_list.append(represent_label2attribute_vec_map[repre_label])
	print('attribute_vec2represent_label_map: ', len(repre_label_list), len(attr_vec_list))

	whole_class_repre_list, whole_attr_np, _ = parse_repre_label2one_hot_map(FLAGS.attrs_per_class_dir)
	# ####################### use train set to valid
	train_image2represent_label_map = parse_train_image2represent_label_map(FLAGS.train_file)
	print('train file', len(train_image2represent_label_map))

	gt_attr_save_dir = os.path.join('../../data/results_gt_attr_with_latent', pretrained_model_path_suffix)
	gt_la_attr = np.load(os.path.join(gt_attr_save_dir, 'gt_la.npz'))['list']
	print('gt_la_attr:', gt_la_attr, gt_la_attr.shape)

	repre_label2true_label_map = parse_represent_label2true_label_map(FLAGS.label_list)
	word_embedding_per_class = parse_word_embedding_per_class(FLAGS.class_wordembeddings)

	whole_word_list = []
	for i in range(len(whole_class_repre_list)):
		true_label = repre_label2true_label_map[whole_class_repre_list[i]]
		word = word_embedding_per_class[true_label]
		whole_word_list.append(word)

	whole_word_np = np.array(whole_word_list, dtype=np.float32)
	print('whole_word_np', whole_word_np.shape, whole_word_np)

	gt_attr = np.concatenate((whole_attr_np[:, 0:FLAGS.attribute_label_cnt], gt_la_attr), axis=1)
	print('gt_attr', gt_attr, gt_attr.shape)

	total_class_set_list, _, _ = parse_repre_label2one_hot_map(
		FLAGS.attrs_per_class_dir)
	print('Total class set', total_class_set_list, len(total_class_set_list))

	train_table = []
	train_class_table = []
	with open(FLAGS.train_file, 'r') as f:
		for line in f.readlines():
			image_name = line.split('	')[0]
			class_repre = line.split('	')[1].replace('\n', '')
			train_table.append(image_name)
			if class_repre not in train_class_table:
				train_class_table.append(class_repre)
	print('READING LABELS OF TRAIN DATA')
	print('Total num:', len(train_table))

	train_class_set_list = [item for item in total_class_set_list if item in train_class_table]
	print('Train class set', train_class_set_list, len(train_class_set_list))

	useen_class_set_list = [item for item in total_class_set_list if item not in train_class_table]
	print('useen_class_set_list', useen_class_set_list, len(useen_class_set_list))

	train_class_index_list = []
	useen_class_index_list = []
	for i in range(len(total_class_set_list)):
		if total_class_set_list[i] in train_class_set_list:
			train_class_index_list.append(i)
		else:
			useen_class_index_list.append(i)
	print('train_class_index_list', train_class_index_list, len(train_class_index_list))
	print('useen_class_index_list', useen_class_index_list, len(useen_class_index_list))

	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		if True:
			# Restore pretrained weights
			pretrained_model = pretrained_model_save_dir
			checkpoint = tf.train.get_checkpoint_state(pretrained_model)
			ckpt = checkpoint.model_checkpoint_path  # 获取最新保存的模型检查点文件
			saver.restore(sess, ckpt)
			for variable in tf.trainable_variables():  # check weights
				with tf.variable_scope('', reuse=True):
					var = tf.get_variable(variable.name.split(':0')[0])
					print(variable.name, np.mean(sess.run(var)))

		# Test start
		step = 0
		pred_labels_total = []
		while True:
			if step < test_size:
				image_name = test_set[step]
				step = step + 1

				image_data = aug_test_image(is_train=False, name=image_name, aug_num=FLAGS.aug_num)

				batch_start_time = time.time()

				pred_logits = sess.run([final_logits], feed_dict={image_placeholder: image_data, is_training: False})
				pred_logits = np.array(pred_logits).squeeze()

				scores = np.matmul(pred_logits, gt_attr.T)
				print('scores_shape', scores.shape)

				scores_useen = np.zeros((FLAGS.aug_num, len(useen_class_index_list)), dtype=np.float32)
				for i in range(len(useen_class_index_list)):
					scores_useen[:, i] = scores[:, useen_class_index_list[i]]
				print('scores_useen', scores_useen)

				max_scroes_indexes = np.argmax(scores_useen, axis=1)
				print('max_scroes_indexes', max_scroes_indexes)

				pred_class_index = []
				for i in range(max_scroes_indexes.shape[0]):
					pred_class_index.append(useen_class_index_list[max_scroes_indexes[i]])
				print('pred_class_index', pred_class_index)

				pred_repre_labels = []
				for i in range(len(pred_class_index)):
					pred_repre_labels.append(whole_class_repre_list[pred_class_index[i]])

				pred_label_set = list(set(pred_repre_labels))
				print('pred_label_set: ', pred_label_set)
				pred_label_set_num = len(pred_label_set)
				pred_label_set_count = np.zeros(pred_label_set_num, dtype=np.int32)
				for pred in pred_repre_labels:
					for j in range(pred_label_set_num):
						if pred == pred_label_set[j]:
							pred_label_set_count[j] += 1

				max_index = int(np.argmax(pred_label_set_count))
				pred_label_after_vote = pred_label_set[max_index]
				print('pred_label_after_vote', pred_label_after_vote)

				pred_labels_total.append(pred_label_after_vote)

				print('[%s][testing %d][step %d / %d exec %.2f seconds]' %
				      (time.strftime("%Y-%m-%d %H:%M:%S"), 1, step, test_size, (time.time() - batch_start_time)))
			else:
				break

	print('Testing done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))

	# write to submit.txt
	with open(test_save_dir + '/' + 'submit_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S")), 'w') as f:
		for i in range(len(test_set)):
			# print('LINES', i)
			f.writelines([test_set[i] + '\t' + pred_labels_total[i] + '\n'])
		f.close()


if __name__ == '__main__':
	test_one_with_aug_multi()