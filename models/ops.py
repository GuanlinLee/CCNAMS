import os,sys
import tensorflow as tf
import tensorflow.contrib as tf_contrib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
from tf_sampling import batch_gather_point
# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None
weight_regularizer_fully = None

def downsample(x, k, point_neighbor, time, is_training, bn_decay=None):

	point_cloud_neighbors = tf_util.conv2d(point_neighbor, x.get_shape().as_list()[-1], [1, 4],
					   padding='SAME', stride=[1, 1],
					   bn=True, is_training=is_training,
					   scope='f_conv'+time, bn_decay=bn_decay)  # [bs, n, k, c']
	point_cloud_neighbors = tf.reduce_mean(point_cloud_neighbors, axis=-2, keep_dims=True)  # [bs, n, 1, c']
	new_x = x + point_cloud_neighbors
	return new_x


def attention(x, ch, is_training, sn=True, bn_decay=None, time= '1'):
	x = x
	batch_size = x.get_shape().as_list()[0]
	height, width, num_channels = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
	f = tf_util.conv2d(x, ch //8, [1, 1],
					   padding='VALID', stride=[1, 1],
					   bn=True, is_training=is_training,
					   scope='f_conv'+time, bn_decay=bn_decay)  # [bs, h, w, c']

	g = tf_util.conv2d(x, ch //8, [1, 1],
					   padding='VALID', stride=[1, 1],
					   bn=True, is_training=is_training,
					   scope='g_conv'+time, bn_decay=bn_decay)  # [bs, h, w, c']

	h = tf_util.conv2d(x, ch //2, [1, 1],
					   padding='VALID', stride=[1, 1],
					   bn=True, is_training=is_training,
					   scope='h_conv'+time, bn_decay=bn_decay)  # [bs, h, w, c]

	s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
	beta = tf.nn.softmax(s)  # attention map
	o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
	gamma = tf.get_variable("gamma"+time, [1], initializer=tf.constant_initializer(0.0))
	o = tf.reshape(o, shape=[batch_size, -1, 1, ch // 2])  # [bs, h, w, C]
	print(o.get_shape().as_list())
	o = tf_util.conv2d(o, ch, [1, 1],
					   padding='VALID', stride=[1, 1],
					   bn=True, is_training=is_training,
					   scope='atten_conv'+time, bn_decay=bn_decay)
	x = gamma * o + x
	return x

def hw_flatten(x) :
	return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def max_pooling(x) :
	return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
