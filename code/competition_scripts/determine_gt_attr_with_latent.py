# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：use trained seen to useen attr regression model
#  				 to determine the ground truth label of latent attributes in useen dataset
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》

import tensorflow as tf
import os
import time
import numpy as np

from config import FLAGS

from parse_raw_data import *
from data_generator import *
from seen2unseen_attr_regression_model import *


def determine_gt_attr_with_latent():
	'''
		Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
	model_path_suffix_la = os.path.join(
		FLAGS.network_def + '_' + FLAGS.version + '_' + 'train_multi' + '_imagesize_' + str(
			FLAGS.img_size) + '_batchsize_' + str(FLAGS.batch_size) + '_experiment_' + FLAGS.experiment_id)
	la_save_dir_train = os.path.join('../../data/results_extract_la' + '/train', model_path_suffix_la)
	la_save_dir_test = os.path.join('../../data/results_extract_la' + '/test', model_path_suffix_la)

	print('load pred attr of train set: ' + la_save_dir_train)
	print('load pred attr of test set: ' + la_save_dir_test)

	pretrained_model_path_suffix = os.path.join(
		FLAGS.network_def + '_' + FLAGS.version + '_' + 'train_multi' + '_imagesize_' + str(
			FLAGS.img_size) + '_batchsize_' + str(FLAGS.batch_size) + '_experiment_' + FLAGS.experiment_id)
	pretrained_model_save_dir = os.path.join('../../data/results_regression/model_weights', pretrained_model_path_suffix)

	gt_attr_save_dir = os.path.join('../../data/results_gt_attr_with_latent', pretrained_model_path_suffix)
	os.system('mkdir -p {}'.format(gt_attr_save_dir))

	train_la_dict = np.load(os.path.join(la_save_dir_train, 'train_la.npz'))['dict'][()]
	print(len(train_la_dict), train_la_dict['7c382f330bd76982761f1a9191e9db0e.jpeg'])

	test_la_dict = np.load(os.path.join(la_save_dir_test, 'test_la.npz'))['dict'][()]
	print(len(test_la_dict), test_la_dict['9a5de40925cd27783c7e229b128d3768.jpg'])

	'''
		Step 2: Create dataset and data generator
	'''
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
	# label_file = pd.read_csv(self.train_file)
	print('READING LABELS OF TRAIN DATA')
	print('Total num:', len(train_table))

	train_class_set_list = [item for item in total_class_set_list if item in train_class_table]
	print('Train class set', train_class_set_list, len(train_class_set_list))

	# obtain image name and class label in train.txt
	train_image2represent_label_map = \
		parse_train_image2represent_label_map(FLAGS.train_file)

	# obtain class label and attribute label in attrs_per_class.txt
	represent_label2attribute_vec_map = \
		parse_attribute_per_class(FLAGS.attrs_per_class_dir)

	useen_class_set_list = [item for item in total_class_set_list if item not in train_class_table]
	print('useen_class_set_list', useen_class_set_list, len(useen_class_set_list))

	'''
	Step 3: Build network graph
	'''
	with tf.Graph().as_default() as g2:
		input_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(train_class_set_list), FLAGS.attribute_label_cnt])
		pred_y = regression_model(input_placeholder, len(train_class_set_list), len(useen_class_set_list))

	'''
		Step 4: Training
	'''

	total_start_time = time.time()
	total_gt_la_attr_dict = {}
	gt_la_attr_train = np.zeros((len(train_class_set_list), FLAGS.attribute_label_cnt), dtype=np.float32)
	for i in range(len(train_class_set_list)):
		temp_la_list = []
		for image_name in train_table:
			if train_image2represent_label_map[image_name] == train_class_set_list[i]:
				temp_la_list.append(train_la_dict[image_name][FLAGS.attribute_label_cnt:2 * FLAGS.attribute_label_cnt])
		temp_la_np = np.array(temp_la_list)
		# print('temp_la_np', temp_la_np, temp_la_np.shape)
		gt_la_attr_train[i, :] = np.mean(temp_la_np, axis=0)
		total_gt_la_attr_dict[train_class_set_list[i]] = gt_la_attr_train[i, :]
		print(train_class_set_list[i], 'gt_la_attr_train[i, :]', gt_la_attr_train[i, :])
	print('gt_la_attr_train_dict', total_gt_la_attr_dict, len(total_gt_la_attr_dict))

	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True), graph=g2) as sess:

		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		if True:
			# Restore pretrained weights
			pretrained_model = pretrained_model_save_dir
			print('loading checkpoint of ', pretrained_model)
			checkpoint = tf.train.get_checkpoint_state(pretrained_model)
			ckpt = checkpoint.model_checkpoint_path  # 获取最新保存的模型检查点文件
			saver.restore(sess, ckpt)
			for variable in tf.trainable_variables():  # check weights
				with tf.variable_scope('', reuse=True):
					var = tf.get_variable(variable.name.split(':0')[0])
					print(variable.name, np.mean(sess.run(var)))

		# Test start
		gt_la_attr_unseen = sess.run(pred_y, feed_dict={input_placeholder: gt_la_attr_train})
		print('gt_la_attr_unseen', gt_la_attr_unseen.shape, gt_la_attr_unseen)
		sess.close()

	print('Testing done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))

	for i in range(len(useen_class_set_list)):
		total_gt_la_attr_dict[useen_class_set_list[i]] = gt_la_attr_unseen[i, :]
	print('gt_la_attr_unseen_dict', total_gt_la_attr_dict, len(total_gt_la_attr_dict))

	total_repre_list_with_latent = []
	for i in range(len(total_class_set_list)):
		total_repre_list_with_latent.append(total_gt_la_attr_dict[total_class_set_list[i]])
	print('total_repre_list_with_latent', total_repre_list_with_latent, len(total_repre_list_with_latent))

	np.savez(os.path.join(gt_attr_save_dir, 'gt_la.npz'), list=total_repre_list_with_latent)


if __name__ == '__main__':
	determine_gt_attr_with_latent()
