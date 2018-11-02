# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：train regression to model the relationship between the given seen attr to useen attr
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


import tensorflow as tf
import os
import time
import numpy as np

from config import FLAGS

from data_generator import *
from data_generator import *
from seen2unseen_attr_regression_model import *


def train_seen_to_useen_attr_regression():
	'''
		Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

	model_path_suffix_save = os.path.join(
		FLAGS.network_def + '_' + FLAGS.version + '_' + 'train_multi' + '_imagesize_' + str(
			FLAGS.img_size) + '_batchsize_' + str(FLAGS.batch_size) + '_experiment_' + FLAGS.experiment_id)
	model_save_dir = os.path.join('../../data/results_regression/model_weights', model_path_suffix_save)
	train_log_save_dir = os.path.join('../../data/results_regression/logs', model_path_suffix_save, 'train')
	os.system('mkdir -p {}'.format(model_save_dir))
	os.system('mkdir -p {}'.format(train_log_save_dir))

	'''
		Step 2: Create dataset
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

	input_data = np.zeros((len(train_class_set_list), FLAGS.attribute_label_cnt), dtype=np.float32)
	for i in range(len(train_class_set_list)):
		input_data[i, :] = np.array(represent_label2attribute_vec_map[train_class_set_list[i]])[0:FLAGS.attribute_label_cnt]
	print('input_data:', input_data[-1])

	useen_class_set_list = [item for item in total_class_set_list if item not in train_class_table]
	print('useen_class_set_list', useen_class_set_list, len(useen_class_set_list))

	output_data = np.zeros([len(useen_class_set_list), FLAGS.attribute_label_cnt], dtype=np.float32)
	for i in range(len(useen_class_set_list)):
		output_data[i, :] = np.array(represent_label2attribute_vec_map[useen_class_set_list[i]])[0:FLAGS.attribute_label_cnt]
	print('ouput_data:', output_data[-1])

	'''
		Step 3: Build network graph
	'''
	with tf.Graph().as_default() as g1:
		input_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(train_class_set_list), FLAGS.attribute_label_cnt])
		output_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(useen_class_set_list), FLAGS.attribute_label_cnt])
		pred_y = regression_model(input_placeholder, len(train_class_set_list), len(useen_class_set_list))
		l2_loss, l2_loss_with_regularizer, train_l2_loss_op, train_l2_loss_with_regularizer_op = regression_loss(pred_y, output_placeholder)

	'''
		Step 4: Training
	'''
	total_start_time = time.time()

	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True), graph=g1) as sess:
		# Create tensorboard
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)

		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		global_step = 0
		# Train start
		for epoch in range(FLAGS.regression_training_epoch):
			for step in range(len(useen_class_set_list)):
				batch_start_time = time.time()
				global_step = step + epoch * len(useen_class_set_list)
				summary, result_l2_loss_with_regularizer, _ = sess.run([merged, l2_loss_with_regularizer, train_l2_loss_with_regularizer_op],
				 feed_dict={input_placeholder:input_data, output_placeholder:output_data})
				train_writer.add_summary(summary, global_step)

				print('[%s][training][epoch %d / %d, step %d / %d, exec %.2f seconds]  loss : %3.10f' %
					  (time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, FLAGS.regression_training_epoch, step + 1, len(useen_class_set_list),
					   (time.time() - batch_start_time), result_l2_loss_with_regularizer))

		saver.save(sess=sess, save_path=model_save_dir + '/' + FLAGS.network_def.split('.py')[0],
		           global_step=(global_step + 1))
		print('\nModel checkpoint saved for total train...\n')

	print('Training useen to useen regression done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))
	train_writer.close()
	sess.close()


if __name__ == '__main__':
	train_seen_to_useen_attr_regression()
