import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net
import ops

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20
    k_up = k
    #print(point_cloud)
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    #print(adj_matrix)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    #print(nn_idx)
    edge_feature, point_neighbor0 = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    with tf.variable_scope('v', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)
        end_points['transform'] = transform
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature, point_neighbor1 = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='dgcnn1', bn_decay=bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net = ops.downsample(net, k_up, point_neighbor0, '1', is_training, bn_decay)
        net1 = net
        adj_matrix = tf_util.pairwise_distance(net)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        edge_feature, point_neighbor2 = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='dgcnn2', bn_decay=bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net = ops.downsample(net, k_up, point_neighbor1, '2', is_training, bn_decay)
        net2 = net

        adj_matrix = tf_util.pairwise_distance(net)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature, point_neighbor3 = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

        net = tf_util.conv2d(edge_feature, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='dgcnn3', bn_decay=bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net = ops.downsample(net, k_up, point_neighbor2, '3', is_training, bn_decay)
        net3 = net

        adj_matrix = tf_util.pairwise_distance(net)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature, point_neighbor4 = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

        net = tf_util.conv2d(edge_feature, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='dgcnn4', bn_decay=bn_decay)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        net = ops.downsample(net, k_up, point_neighbor3, '4', is_training, bn_decay)
        net4 = net

        hx = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='agg', bn_decay=bn_decay)
        end_points['pre_max'] = hx
        net = tf.reduce_max(hx, axis=1, keep_dims=True)
        end_points['post_max'] = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                              scope='dp2')
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, hx

def get_loss_v2(pred, label, end_points=None):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = loss
    return classify_loss

def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
		label: B, """
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = loss
    return classify_loss

def get_adv_loss(unscaled_logits, targets, kappa=0):
    with tf.variable_scope('adv_loss'):
        unscaled_logits_shape = tf.shape(unscaled_logits)

        B = unscaled_logits_shape[0]
        K = unscaled_logits_shape[1]
        ybar = tf.nn.softmax(unscaled_logits, name='ybar')
        y = tf.argmin(ybar, axis=1, output_type=tf.int32)
        ydim = unscaled_logits.get_shape().as_list()[1]
        tlab = tf.one_hot(y, ydim, on_value=1., off_value=0.)
        # encourage to classify to a wrong category
        # tlab=tf.one_hot(targets,depth=K,on_value=1.,off_value=0.)
        # tlab=tf.expand_dims(tlab,0)
        # tlab=tf.tile(tlab,[B,1])
        real = tf.reduce_sum((tlab) * unscaled_logits, 1)
        other = tf.reduce_max((1 - tlab) * unscaled_logits -
                              (tlab * 10000), 1)
        loss1 = tf.maximum(np.asarray(0., dtype=np.dtype('float32')), other - real + kappa)
        return tf.reduce_mean(loss1)


def get_critical_points(sess, ops, data, BATCH_SIZE, NUM_ADD, NUM_POINT=1024):
    ####################################################
    ### get the critical point of the given point clouds
    ### data shape: BATCH_SIZE*NUM_POINT*3
    ### return : BATCH_SIZE*NUM_ADD*3
    #####################################################
    sess.run(tf.assign(ops['pert'], tf.zeros([BATCH_SIZE, NUM_ADD, 3])))
    is_training = False

    # to make sure init_points is in shape of BATCH_SIZE*NUM_ADD*3 so that it can be fed to initial_point_pl
    if NUM_ADD > NUM_POINT:
        init_points = np.tile(data[:, :2, :], [1, NUM_ADD / 2, 1])  ## due to the max pooling operation of PointNet,
        ## duplicated points would not affect the global feature vector
    else:
        init_points = data[:, :NUM_ADD, :]
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['is_training_pl']: is_training,
                 ops['initial_point_pl']: init_points}
    pre_max_val, post_max_val = sess.run([ops['pre_max'], ops['post_max']], feed_dict=feed_dict)
    pre_max_val = pre_max_val[:, :NUM_POINT, ...]
    pre_max_val = np.reshape(pre_max_val,
                             [BATCH_SIZE, NUM_POINT, 1024])  # 1024 is the dimension of PointNet's global feature vector

    critical_points = []
    for i in range(len(pre_max_val)):
        # get the most important critical points if NUM_ADD < number of critical points
        # the importance is demtermined by counting how many elements in the global featrue vector is
        # contributed by one specific point
        idx, counts = np.unique(np.argmax(pre_max_val[i], axis=0), return_counts=True)
        idx_idx = np.argsort(counts)
        if len(counts) >= NUM_ADD:
            points = data[i][idx[idx_idx[-NUM_ADD:]]]
        else:
            points = data[i][idx]
            tmp_num = NUM_ADD - len(counts)
            while (tmp_num > len(counts)):
                points = np.concatenate([points, data[i][idx]])
                tmp_num -= len(counts)
            points = np.concatenate([points, data[i][-tmp_num:]])

        critical_points.append(points)
    critical_points = np.stack(critical_points)
    return critical_points

def get_critical_points_aia(sess, ops, data, BATCH_SIZE, NUM_ADD, NUM_POINT):
    ####################################################
    ### get the critical point of the given point clouds
    ### data shape: BATCH_SIZE*NUM_POINT*3
    ### return : BATCH_SIZE*NUM_ADD*3
    #####################################################
    sess.run(tf.assign(ops['pert'], tf.zeros([BATCH_SIZE, NUM_ADD, 3])))
    is_training = False
    NUM_POINT = data.shape[1]
    # to make sure init_points is in shape of BATCH_SIZE*NUM_ADD*3 so that it can be fed to initial_point_pl
    if NUM_ADD > NUM_POINT:
        init_points = np.tile(data[:, :2, :], [1, NUM_ADD // 2, 1])  ## due to the max pooling operation of PointNet,
        ## duplicated points would not affect the global feature vector
    else:
        init_points = data[:, :NUM_ADD, :]
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['pointclouds_pl_resample']: data,
                 ops['is_training_pl']: is_training,
                 ops['initial_point_pl']: init_points}
    pre_max_val, post_max_val = sess.run([ops['pre_max'], ops['post_max']], feed_dict=feed_dict)
    pre_max_val = pre_max_val[:, :NUM_POINT, ...]
    pre_max_val = np.reshape(pre_max_val,
                             [BATCH_SIZE, NUM_POINT, 1024])  # 1024 is the dimension of PointNet's global feature vector

    critical_points = []
    for i in range(len(pre_max_val)):
        # get the most important critical points if NUM_ADD < number of critical points
        # the importance is demtermined by counting how many elements in the global featrue vector is
        # contributed by one specific point
        idx, counts = np.unique(np.argmax(pre_max_val[i], axis=0), return_counts=True)
        idx_idx = np.argsort(counts)
        if len(counts) >= NUM_ADD:
            points = data[i][idx[idx_idx[-NUM_ADD:]]]
        else:
            points = data[i][idx]
            tmp_num = NUM_ADD - len(counts)
            while (tmp_num > len(counts)):
                points = np.concatenate([points, data[i][idx]])
                tmp_num -= len(counts)
            points = np.concatenate([points, data[i][-tmp_num:]])
        critical_points.append(points)
    critical_points = np.stack(critical_points)
    return critical_points
if __name__=='__main__':
    batch_size = 2
    num_pt = 124
    pos_dim = 3

    input_feed = np.random.rand(batch_size, num_pt, pos_dim)
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed>=0.5] = 1
    label_feed[label_feed<0.5] = 0
    label_feed = label_feed.astype(np.int32)

    # # np.save('./debug/input_feed.npy', input_feed)
    # input_feed = np.load('./debug/input_feed.npy')
    # print input_feed

    with tf.Graph().as_default():
        input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
        pos, ftr = get_model(input_pl, tf.constant(True))
        # loss = get_loss(logits, label_pl, None)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {input_pl: input_feed, label_pl: label_feed}
            res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)












