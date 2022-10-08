import numpy as np
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
#from keras import backend as K


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0.000001, 0.000001
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0.000001, 0.000001
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc

def cal_gauc(pred, label, user_id):
    '''
    :param label: ground truth
    :param prob: predicted prob
    :param user_id: user index
    :return: gauc
    '''
    if(len(label) != len(user_id)):
        raise ValueError("impression id num should equal to the sample num,"\
                         "impression id num is {}".format(len(user_id)))
    group_truth = defaultdict(lambda: [])
    group_score = defaultdict(lambda: [])
    for idx, truth in enumerate(label):
        uid = user_id[idx]
        group_truth[uid].append(label[idx])
        group_score[uid].append(pred[idx])
    group_flag = defaultdict(lambda: False)
    for uid in set(user_id):
        truths = group_truth[uid]
        flag = False
        for i in range(len(truths)-1):
            if(truths[i] != truths[i+1]):
                flag = True
                break
        group_flag[uid] = flag
    total_auc = 0
    total_impression = 0
    auc_details = {}
    for uid in group_flag:
        if group_flag[uid] and len(group_truth[uid]) > 1:
            p_t_list = []
            for i in range(len(group_truth[uid])):
                p_t_list.append([group_score[uid][i], group_truth[uid][i]])
            total_auc += len(group_truth[uid]) * calc_auc(p_t_list)
            auc_details[uid] = [len(group_truth[uid]), calc_auc(p_t_list)]
            total_impression += len(group_truth[uid])
    group_auc = float(total_auc) / total_impression
    # group_auc = round(group_auc, 4)

    for key in auc_details:
        auc_details[key][0] = float(auc_details[key][0])/total_impression
    return group_auc, auc_details



def bn(data, is_training, center=True, scale=True, epsilon=1e-3, momentum=0.9, scope=None):
    # print "# %s # bn"%tf.get_default_graph().get_name_scope()
    out = tf.layers.batch_normalization(
        inputs=data,
        momentum=momentum,
        epsilon=epsilon,
        scale=scale,
        moving_variance_initializer=tf.zeros_initializer(),
        training=is_training,
        center=center,
        reuse=tf.AUTO_REUSE,
        name=scope
    )
    return out




def fc(data, units, act_type, is_training, use_bn=False, scope=None):
    units_in = int(data.shape[-1])
    assert units_in > 0, 'invalid input shape: %s'%data.shape

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weight_value = np.random.randn(units, units_in).astype(
            np.float32
        ) / np.sqrt(units_in)
        #print scope, "############", weight_value.shape, weight_value
        print "# %s/fc #"%tf.get_default_graph().get_name_scope()
        weight_value = np.transpose(weight_value)
        dout = tf.layers.dense(data, units, activation=None,
                               kernel_initializer=tf.initializers.constant(weight_value),
                               bias_initializer=tf.initializers.constant(0.1), name='fc')

        if use_bn and act_type != 'dice':
            dout = bn(dout, is_training, scope='bn')

        if act_type == 'sigmoid':
            out = tf.nn.sigmoid(dout)
        elif act_type == 'relu':
            out = tf.nn.relu(dout)
        elif act_type == 'prelu':
            alpha = tf.get_variable(
                'prelu_alpha',
                shape=(units,),
                initializer=tf.constant_initializer(-0.25),
            )
            out = tf.maximum(0.0, dout) + alpha * tf.minimum(0.0, dout)
        elif act_type == 'dice':
            out = bn(dout, is_training=is_training, center=False, scale=False, epsilon=1e-4, momentum=0.99, scope='dice_bn')
            logits = tf.nn.sigmoid(out)   # 1 / (1 + tf.exp(-out))
            dice_gamma = tf.get_variable(
                'dice_gamma',
                shape=(1, units),
                initializer=tf.constant_initializer(-0.25)
            )
            out = tf.multiply(dice_gamma, (1.0 - logits) * dout) + logits * dout
        elif not act_type:
            out = dout
        else:
            raise RuntimeError('unknown act_type %s' % act_type)
    return out
