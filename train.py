import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import torch
import emd_module as emd
import socket
import importlib
import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import data_utils
import faulthandler

faulthandler.enable()
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='ccdgn', help='Model name: ccdgn')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--num_drop', type=int, default=10, help='num of points to drop each step')
parser.add_argument('--num_steps', type=int, default=10, help='num of steps to drop each step')
parser.add_argument('--drop_neg', action='store_true', help='drop negative points')
parser.add_argument('--power', type=int, default=1, help='x: -dL/dr*r^x')
parser.add_argument('--points', type=int, default=512, help='attack changes points')
FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(0)
tf.set_random_seed(0)
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40
TH = 0.5
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
	os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles( \
	os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


class SphereAttack():
	def __init__(self, num_drop, num_steps, pointclouds_pl, labels_pl,
				 is_training_pl, pred, classify_loss):
		self.a = num_drop  # how many points to remove
		self.k = num_steps

		self.is_training = False

		# The number of points is not specified
		self.pointclouds_pl, self.labels_pl = pointclouds_pl, labels_pl
		self.is_training_pl = is_training_pl
		# simple model
		self.pred = pred
		self.classify_loss = classify_loss
		label_mask = tf.one_hot(self.labels_pl,
								40,
								on_value=1.0,
								off_value=0.0,
								dtype=tf.float32)
		correct_logit = tf.reduce_sum(label_mask * self.pred, axis=1)
		wrong_logit = tf.reduce_max((1 - label_mask) * self.pred - 1e4 * label_mask, axis=1)
		self.loss_cw = -tf.nn.relu(correct_logit - wrong_logit + 50)
		self.grad_cw = tf.gradients(self.loss_cw, self.pointclouds_pl)[0]
		self.grad = tf.gradients(self.classify_loss, self.pointclouds_pl)[0]

	def drop_points(self, pointclouds_pl, labels_pl, sess, is_pert=False):
		step = 0.005
		eps = 0.02
		pointclouds_pl_adv = pointclouds_pl.copy()
		opl = pointclouds_pl.copy()
		point_index_set = []
		if is_pert == False:
			for i in range(self.k):
				grad = sess.run(self.grad, feed_dict={self.pointclouds_pl: pointclouds_pl_adv,
													  self.labels_pl: labels_pl,
													  self.is_training_pl: self.is_training})
				#print(grad)
				#grad = _
				sphere_core = np.median(pointclouds_pl_adv, axis=1, keepdims=True)

				sphere_r = np.sqrt(np.sum(np.square(pointclouds_pl_adv - sphere_core), axis=2))  ## BxN

				sphere_axis = pointclouds_pl_adv - sphere_core  ## BxNx3

				if FLAGS.drop_neg:
					sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
											 np.power(sphere_r, FLAGS.power))
				else:
					sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2),
											  np.power(sphere_r, FLAGS.power))

				drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1] - self.a, axis=1)[:, -self.a:]

				point_index_set.append([])
				tmp = np.zeros((pointclouds_pl_adv.shape[0], pointclouds_pl_adv.shape[1] - self.a, 3), dtype=float)
				for j in range(pointclouds_pl.shape[0]):
					tmp[j] = np.delete(pointclouds_pl_adv[j], drop_indice[j], axis=0)  # along N points to delete

				pointclouds_pl_adv = tmp.copy()
			return pointclouds_pl_adv, np.array(point_index_set).reshape(sphere_r.shape[0], -1)
		else:
			tmp = opl.copy()
			for i in range(self.k):
				grad = sess.run(self.grad, feed_dict={self.pointclouds_pl: pointclouds_pl_adv,
													  self.labels_pl: labels_pl,
													  self.is_training_pl: self.is_training})

				tmp = tmp + np.sign(grad) * step
				tmp = np.clip(tmp, opl - eps, opl + eps)
				pointclouds_pl_adv = tmp.copy()
			return pointclouds_pl_adv

def log_string(out_str):
	LOG_FOUT.write(out_str + '\n')
	LOG_FOUT.flush()
	print(out_str)

def get_learning_rate(batch):
	learning_rate = tf.train.exponential_decay(
		BASE_LEARNING_RATE,  # Base learning rate.
		batch * BATCH_SIZE,  # Current index into the dataset.
		DECAY_STEP,  # Decay step.
		DECAY_RATE,  # Decay rate.
		staircase=True)
	learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
	return learning_rate

def get_bn_decay(batch):
	bn_momentum = tf.train.exponential_decay(
		BN_INIT_DECAY,
		batch * BATCH_SIZE,
		BN_DECAY_DECAY_STEP,
		BN_DECAY_DECAY_RATE,
		staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def train():
	with tf.Graph().as_default():
		with tf.device('/gpu:' + str(GPU_INDEX)):
			pointclouds_pl, labels_pl_1 = MODEL.placeholder_inputs(BATCH_SIZE, None)
			pointclouds_pl_adv = tf.placeholder(tf.float32, shape=(BATCH_SIZE, None, 3))
			labels_pl_2 = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
			labels_pl_3 = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
			is_training_pl = tf.placeholder(tf.bool, shape=())
			lam = tf.placeholder(tf.float32, shape=())


			# Note the global_step=batch parameter to minimize.
			# That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
			batch = tf.Variable(0)
			bn_decay = get_bn_decay(batch)
			tf.summary.scalar('bn_decay', bn_decay)

			# Get model and loss
			# def train_eval():
			pred, end_points, hx = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
			pred1, end_points1, hx1 = MODEL.get_model(pointclouds_pl_adv, is_training_pl, bn_decay=bn_decay)

			loss = MODEL.get_loss(pred, labels_pl_1, end_points)
			loss = tf.reduce_mean(loss)
			loss_adv = MODEL.get_loss(pred1, labels_pl_2, end_points) * (1 - lam) + \
					   MODEL.get_loss(pred1, labels_pl_3, end_points) * lam
			loss_adv = tf.reduce_mean(loss_adv)
			loss_v2 = MODEL.get_loss_v2(pred, labels_pl_1)
			attacker = SphereAttack(FLAGS.num_drop, FLAGS.num_steps, pointclouds_pl, labels_pl_1,
									is_training_pl, pred, loss_v2)
			tf.summary.scalar('loss', loss)
			correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl_1))
			accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
			tf.summary.scalar('accuracy', accuracy)
			learning_rate = get_learning_rate(batch)
			tf.summary.scalar('learning_rate', learning_rate)
			if OPTIMIZER == 'momentum':
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
			elif OPTIMIZER == 'adam':
				optimizer = tf.train.AdamOptimizer(learning_rate)
			train_op = optimizer.minimize((loss + loss_adv) / 2.0, global_step=batch)

			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()

		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)

		# Add summary writers
		# merged = tf.merge_all_summaries()
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
											 sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

		# Init variables
		init = tf.global_variables_initializer()
		# To fix the bug introduced in TF 0.12.1 as in
		# http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
		# sess.run(init)
		sess.run(init, {is_training_pl: True})

		ops = {'pointclouds_pl': pointclouds_pl,
			   'pointclouds_pl_adv': pointclouds_pl_adv,
			   'labels_pl_1': labels_pl_1,
			   'labels_pl_2': labels_pl_2,
			   'labels_pl_3': labels_pl_3,
			   'lam': lam,
			   'is_training_pl': is_training_pl,
			   'pred': pred,
			   'pred1': pred1,
			   'loss': loss,
			   'train_op': train_op,
			   'merged': merged,
			   'step': batch,
			   'acc': accuracy,
			   'hx': hx}

		for epoch in range(MAX_EPOCH):
			log_string('**** EPOCH %03d ****' % (epoch))
			sys.stdout.flush()
			# if epoch <= 10:
			#    train_one_epoch(sess, ops, train_writer)
			# else:
			train_one_epoch_adap(sess, ops, train_writer, attacker)
			eval_one_epoch(sess, ops, test_writer)

			# Save the variables to disk.
			if epoch % 10 == 0:
				save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
				log_string("Model saved in file: %s" % save_path)

def mixup_sampler(data, label):
	PointcloudScaleAndTranslate = data_utils.PointcloudScaleAndTranslate()
	beta = 1
	data_torch = torch.from_numpy(data).cuda()
	data_torch = PointcloudScaleAndTranslate(data_torch)
	lam = np.random.beta(beta, beta)
	B = data_torch.size()[0]
	rand_index = torch.randperm(B)
	target_a = label
	target_b = label[rand_index]
	point_a = data_torch.cuda()
	point_b = data_torch[rand_index].cuda()
	point_c = data_torch[rand_index].cuda()
	remd = emd.emdModule()
	remd = remd.cuda()
	dis, ind = remd(point_a, point_b, 0.005, 300)
	for idx in range(B):
		point_c[idx, :, :] = point_c[idx, ind[idx].long(), :]

	int_lam = int(NUM_POINT * lam)
	int_lam = max(1, int_lam)

	random_point = torch.from_numpy(np.random.choice(NUM_POINT, B, replace=False, p=None))
	# kNN
	ind1 = torch.tensor(range(B))
	query = point_a[ind1, random_point].view(B, 1, 3)
	dist = torch.sqrt(torch.sum((point_a - query.repeat(1, NUM_POINT, 1)) ** 2, 2))
	idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
	for i2 in range(B):
		data_torch[i2, idxs[i2], :] = point_c[i2, idxs[i2], :]
	# adjust lambda to exactly match point ratio
	lam = int_lam * 1.0 / NUM_POINT
	data = data_torch.cpu().detach().numpy()

	added_points = np.zeros((data.shape[0], FLAGS.points, 3), dtype=data.dtype)
	for i in range(data.shape[0]):
		min1 = np.min(data[i, :, 0])
		max1 = np.max(data[i, :, 0])
		min2 = np.min(data[i, :, 1])
		max2 = np.max(data[i, :, 1])
		min3 = np.min(data[i, :, 2])
		max3 = np.max(data[i, :, 2])
		added_points[i, :, 0] = np.random.uniform(min1, max1, (FLAGS.points))
		added_points[i, :, 1] = np.random.uniform(min2, max2, (FLAGS.points))
		added_points[i, :, 2] = np.random.uniform(min3, max3, (FLAGS.points))
	jittered_data_adv_new = np.concatenate((data, added_points), axis=1)
	return data, jittered_data_adv_new, lam, target_a, target_b

def adv_generator(data, label, attacker, sess):
	sample_number = NUM_POINT + FLAGS.points
	rotated_data = provider.rotate_point_cloud(data)
	jittered_data = provider.jitter_point_cloud(rotated_data)

	jittered_data_adv, gradient_index = attacker.drop_points(jittered_data,
															 label, sess, False)
	jittered_data_adv_drop = jittered_data_adv.copy()

	added_points = np.zeros((jittered_data_adv.shape[0], FLAGS.points, 3), dtype=jittered_data_adv.dtype)
	for i in range(jittered_data_adv.shape[0]):
		min1 = np.min(jittered_data_adv[i, :, 0])
		max1 = np.max(jittered_data_adv[i, :, 0])
		min2 = np.min(jittered_data_adv[i, :, 1])
		max2 = np.max(jittered_data_adv[i, :, 1])
		min3 = np.min(jittered_data_adv[i, :, 2])
		max3 = np.max(jittered_data_adv[i, :, 2])
		added_points[i, :, 0] = np.random.uniform(min1, max1, (FLAGS.points))
		added_points[i, :, 1] = np.random.uniform(min2, max2, (FLAGS.points))
		added_points[i, :, 2] = np.random.uniform(min3, max3, (FLAGS.points))
	jittered_data_adv_add = np.concatenate((jittered_data_adv, added_points), axis=1)
	jittered_data_adv_perturb = attacker.drop_points(jittered_data,
													 label, sess, True)
	#added_points_drop = np.zeros((jittered_data_adv_drop.shape[0], sample_number - jittered_data_adv_drop.shape[1], 3))
	#added_points_add = np.zeros((jittered_data_adv_add.shape[0], sample_number - jittered_data_adv_add.shape[1], 3))
	#added_points_perturb = np.zeros((jittered_data_adv_drop.shape[0], sample_number - jittered_data_adv_perturb.shape[1], 3))
	#jittered_data_adv_drop = np.concatenate((jittered_data_adv_drop, added_points_drop), axis=1)
	#jittered_data_adv_add = np.concatenate((jittered_data_adv_add, added_points_add), axis=1)
	#jittered_data_adv_perturb = np.concatenate((jittered_data_adv_perturb, added_points_perturb), axis=1)
	return jittered_data_adv_drop, jittered_data_adv_add, jittered_data_adv_perturb

def train_one_epoch_adap(sess, ops, train_writer, attacker):
	""" ops: dict mapping from string to tf ops """
	is_training = True

	# Shuffle train files
	train_file_idxs = np.arange(0, len(TRAIN_FILES))
	np.random.shuffle(train_file_idxs)

	for fn in range(len(TRAIN_FILES)):
		log_string('----' + str(fn) + '-----')
		current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
		current_data = current_data[:, 0:NUM_POINT, :]
		current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
		current_label = np.squeeze(current_label)

		file_size = current_data.shape[0]
		num_batches = file_size // BATCH_SIZE

		total_correct = 0
		total_seen = 0
		loss_sum = 0
		correct_normal_sum, correct_drop_sum, correct_perturb_sum = 0.0, 0.0, 0.0
		for batch_idx in range(num_batches):
			start_idx = batch_idx * BATCH_SIZE
			end_idx = (batch_idx + 1) * BATCH_SIZE

			# Augment batched point clouds by rotation and jittering
			data = current_data[start_idx:end_idx, :, :]
			label = current_label[start_idx:end_idx]
			data_, jittered_data_mix, lam, target_a, target_b = mixup_sampler(data, label)
			jittered_data_adv_drop, jittered_data_adv_add, jittered_data_adv_perturb = adv_generator(data, label,
																									 attacker, sess)
			feed_dict_normal = {ops['pointclouds_pl']: data,
								ops['pointclouds_pl_adv']: data,
								ops['labels_pl_1']: label,
								ops['labels_pl_2']: label,
								ops['labels_pl_3']: label,
								ops['lam']: 1.0,
								ops['is_training_pl']: is_training, }

			feed_dict_drop = {ops['pointclouds_pl']: jittered_data_adv_drop,
							  ops['pointclouds_pl_adv']: jittered_data_adv_drop,
							  ops['labels_pl_1']: label,
							  ops['labels_pl_2']: label,
							  ops['labels_pl_3']: label,
							  ops['lam']: 1.0,
							  ops['is_training_pl']: is_training, }

			feed_dict_perturb = {ops['pointclouds_pl']: jittered_data_adv_perturb,
								 ops['pointclouds_pl_adv']: jittered_data_adv_perturb,
								 ops['labels_pl_1']: label,
								 ops['labels_pl_2']: label,
								 ops['labels_pl_3']: label,
								 ops['lam']: 1.0,
								 ops['is_training_pl']: is_training, }
			pred_normal = sess.run(ops['pred'], feed_dict=feed_dict_normal)
			pred_drop = sess.run(ops['pred'], feed_dict=feed_dict_drop)
			pred_perturb = sess.run(ops['pred'], feed_dict=feed_dict_perturb)
			pred_normal = np.argmax(pred_normal, 1)
			pred_drop = np.argmax(pred_drop, 1)
			pred_perturb = np.argmax(pred_perturb, 1)
			correct_normal = np.sum(pred_normal == current_label[start_idx:end_idx])/(end_idx-start_idx)
			correct_drop = np.sum(pred_drop == current_label[start_idx:end_idx])/(end_idx-start_idx)
			correct_perturb = np.sum(pred_perturb == current_label[start_idx:end_idx])/(end_idx-start_idx)
			correct_normal_sum += correct_normal
			correct_drop_sum += correct_drop
			correct_perturb_sum += correct_perturb
			cl = [correct_drop_sum / (batch_idx+1), correct_perturb_sum / (batch_idx+1)]
			min_value = min(cl)
			max_value = max(cl)
			print('min acc:',min_value,'min attack:', cl.index(min_value),'max acc:',max_value,'max attack:', cl.index(max_value))
			if min_value > TH * correct_normal_sum / (batch_idx+1):
				feed_dict = {ops['pointclouds_pl']: data,
							 ops['pointclouds_pl_adv']: jittered_data_mix,
							 ops['labels_pl_1']: label,
							 ops['labels_pl_2']: target_a,
							 ops['labels_pl_3']: target_b,
							 ops['lam']: lam,
							 ops['is_training_pl']: is_training, }
				summary, step, _, loss_val, pred_val, hx = sess.run([ops['merged'], ops['step'],
																	 ops['train_op'], ops['loss'], ops['pred'], ops['hx']],
																	feed_dict=feed_dict)
			else:
				feed_dict = {ops['pointclouds_pl']: data,
							 ops['pointclouds_pl_adv']: jittered_data_adv_drop,
							 ops['labels_pl_1']: label,
							 ops['labels_pl_2']: label,
							 ops['labels_pl_3']: label,
							 ops['lam']: lam,
							 ops['is_training_pl']: is_training, }

				summary, step, _, loss_val, pred_val, hx = sess.run([ops['merged'], ops['step'],
																	 ops['train_op'], ops['loss'], ops['pred'], ops['hx']],
																	feed_dict=feed_dict)
				feed_dict = {ops['pointclouds_pl']: data,
							 ops['pointclouds_pl_adv']: jittered_data_adv_perturb,
							 ops['labels_pl_1']: label,
							 ops['labels_pl_2']: label,
							 ops['labels_pl_3']: label,
							 ops['lam']: lam,
							 ops['is_training_pl']: is_training, }
				summary, step, _, loss_val, pred_val, hx = sess.run([ops['merged'], ops['step'],
																	 ops['train_op'], ops['loss'], ops['pred'], ops['hx']],
																	feed_dict=feed_dict)

			train_writer.add_summary(summary, step)
			pred_val = np.argmax(pred_val, 1)
			correct = np.sum(pred_val == current_label[start_idx:end_idx])
			total_correct += correct
			total_seen += BATCH_SIZE
			loss_sum += loss_val
			pred1 = sess.run(ops['pred1'], feed_dict=feed_dict)
		log_string('mean loss: %f' % (loss_sum / float(num_batches)))
		log_string('accuracy: %f' % (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, test_writer):
	""" ops: dict mapping from string to tf ops """
	bs = BATCH_SIZE
	is_training = False
	total_correct = 0
	total_seen = 0
	loss_sum = 0
	total_seen_class = [0 for _ in range(NUM_CLASSES)]
	total_correct_class = [0 for _ in range(NUM_CLASSES)]

	for fn in range(len(TEST_FILES)):
		log_string('----' + str(fn) + '-----')
		current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
		current_data = current_data[:, 0:NUM_POINT, :]
		current_label = np.squeeze(current_label)

		file_size = current_data.shape[0]
		num_batches = file_size // bs

		for batch_idx in range(num_batches):
			start_idx = batch_idx * bs
			end_idx = (batch_idx + 1) * bs
			d = current_data[start_idx:end_idx, :, :]
			l = current_label[start_idx:end_idx]
			feed_dict = {ops['pointclouds_pl']: d,
						 ops['pointclouds_pl_adv']: d,
						 ops['labels_pl_1']: l,
						 ops['labels_pl_2']: l,
						 ops['labels_pl_3']: l,
						 ops['lam']: 1.0,
						 ops['is_training_pl']: is_training}
			step, loss_val, pred_val, acc = sess.run([ops['step'],
													  ops['loss'], ops['pred'], ops['acc']], feed_dict=feed_dict)
			# print(state)
			# print(loss_val, acc)
			pred_val = np.argmax(pred_val, 1)
			correct = np.sum(pred_val == l)
			total_correct += correct
			total_seen += bs
			loss_sum += (loss_val * bs)
			for i in range(start_idx, end_idx):
				l = current_label[i]
				total_seen_class[l] += 1
				total_correct_class[l] += (pred_val[i - start_idx] == l)

	log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
	log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
	log_string('eval avg class acc: %f' % (
		np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


if __name__ == "__main__":
	train()
	LOG_FOUT.close()
