import sys
sys.path.append('..')
sys.path.append('../..')

from .network import Network

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

import numpy as np
import yaml
import cv2
from utils.tf import recoverer

from utils.opencvhelper import MatcherWrapper
from utils.tf import recoverer

BASE_PATH = '/detector_master/'

class ASLFeatNet(Network):

    def close(self):
        self.sess.close()
        tf1.compat.v1.reset_default_graph()

    def construct_network(self):
        ph_imgs = tf1.placeholder(dtype=tf1.float32, shape=(None, None, None, 1), name='input')
        mean, variance = tf1.nn.moments(
            tf1.cast(ph_imgs, tf1.float32), axes=[1, 2], keep_dims=True)
        norm_input = tf1.nn.batch_normalization(ph_imgs, mean, variance, None, None, 1e-5)
        config_dict = {'det_config': self.netConfig['config']}
        self.network = ASLFeatNet({'data': norm_input}, is_training=False, resue=False, **config_dict)
        self.endpoints = self.network.endpoints
        
    def init(self):
        self.config_file_path = BASE_PATH + 'configs/our_dectetor_test.yaml'
        with open(self.config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.modelPath = BASE_PATH + config['model_path']
        self.netConfig = config['net']
        sess_config = tf1.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.construct_network()
        self.sess = tf1.compat.v1.Session(config=sess_config)
        recoverer(self.sess, self.modelPath)
        
        ###### loss calculation function #######

    def setup(self):
        self.tmp_feat_map = [None, None, None]
        det_config = self.extra_args['det_config']
        deform_desc = det_config['deform_desc']

        (self.feed('data')
         .conv_bn(3, 32, 1, name='conv0')
         .conv(3, 32, 1, biased=False, relu=False, name='conv1')
         .batch_normalization(relu=True, name='conv1/bn')
         .conv_bn(3, 64, 2, name='conv2')
         .conv(3, 64, 1, biased=False, relu=False, name='conv3')
         .batch_normalization(relu=True, name='conv3/bn')
         .conv_bn(3, 128, 2, name='conv4')
         .conv_bn(3, 128, 1, name='conv5'))

        if deform_desc > 0:
            if deform_desc == 1:
                deform_type = 'u'
            elif deform_desc == 2:
                deform_type = 'a'
            elif deform_desc == 3:
                deform_type = 'h'
            else:
                raise NotImplementedError

            (self.feed('conv5')
             .deform_conv_bn(3, 128, 1, deform_type=deform_type, name='conv6_0')
             .deform_conv_bn(3, 128, 1, deform_type=deform_type, name='conv6_1')
             .deform_conv(3, 128, 1, biased=False, relu=False, deform_type=deform_type, name='conv6'))
        else:
            (self.feed('conv5')
             .conv_bn(3, 128, 1, name='conv6_0')
             .conv_bn(3, 128, 1, name='conv6_1')
             .conv(3, 128, 1, biased=False, relu=False, name='conv6'))

        dense_feat_map = self.layers['conv6']

        comb_names = ['conv1', 'conv3', 'conv6']
        comb_weights = tf.constant([1, 2, 3], dtype=tf.float32)
        comb_weights /= tf.reduce_sum(comb_weights)
        scale = [3, 2, 1]

        comb_score_map = None

        ori_h = tf.shape(self.inputs['data'])[1]
        ori_w = tf.shape(self.inputs['data'])[2]

        for idx, tmp_name in enumerate(comb_names):
            tmp_feat_map = self.layers[tmp_name]
            prep_dense_feat_map = tmp_feat_map

            alpha, beta = self.peakiness_score(prep_dense_feat_map, ksize=3,
                                            need_norm=det_config['need_norm'],
                                            dilation=scale[idx], name=tmp_name)
            score_vol = alpha * beta
            score_map = tf.reduce_max(score_vol, axis=-1, keepdims=True)
            score_map = tf.image.resize(score_map, (ori_h, ori_w))
            tmp_comb_weights = comb_weights[idx] * score_map

            if comb_score_map is None:
                comb_score_map = tmp_comb_weights
            else:
                comb_score_map += tmp_comb_weights

        score_map = comb_score_map

        kpt_inds, kpt_score = self.extract_kpts(
            score_map, k=2500,
            score_thld=det_config['score_thld'], edge_thld=det_config['edge_thld'],
            nms_size=5, eof_size=det_config['eof_mask'])

        offsets = tf.squeeze(self.kpt_refinement(score_map), axis=-2)
        offsets = tf.gather_nd(offsets, kpt_inds, batch_dims=1)
        offsets = tf.clip_by_value(offsets, -0.5, 0.5)
        kpt_inds = tf.cast(kpt_inds, tf.float32) + offsets

        self.tmp_feat_map[0] = self.layers[comb_names[0]]
        self.tmp_feat_map[1] = self.layers[comb_names[1]]
        self.tmp_feat_map[2] = self.layers[comb_names[2]]

        #p = self.peakiness_score(self.tmp_feat_map[2])
        
        ones = tf1.ones([ori_h, ori_w], tf1.float32)
        indices = tf1.where(ones)
        indices = tf.expand_dims(indices, axis=0)
        indices = tf.cast(indices, tf.float32)

        self.endpoints['feature_maps'] = self.tmp_feat_map
        #self.endpoints['descs'] = tf.nn.l2_normalize(interpolate(indices/4, dense_feat_map), axis=-1, name='descs')
        self.endpoints['descs'] = tf.nn.l2_normalize(interpolate(
            indices / 4, dense_feat_map), axis=-1, name='descs')
        self.endpoints['keypoints'] = tf.stack([kpt_inds[:, :, 1], kpt_inds[:, :, 0]], axis=-1, name='kpts')
        self.endpoints['score_map'] = score_map
        self.endpoints['scores'] = tf.identity(kpt_score, name='scores')
        #self.endpoints['descs'] = tf.nn.l2_normalize(dense_feat_map, axis=-1, name='descs')

    def peakiness_score(self, inputs, ksize=3, need_norm=True, dilation=1, name='conv'):
        if need_norm:
            from tensorflow.python.training.moving_averages import assign_moving_average
            with tf1.variable_scope('tower', reuse=self.reuse):
                moving_instance_max = tf1.get_variable('%s/instance_max' % name, (),
                                                                initializer=tf1.constant_initializer(
                                                                    1),
                                                                trainable=False)
            decay = 0.99

            if self.training:
                instance_max = tf1.reduce_max(inputs)
                with tf1.control_dependencies([assign_moving_average(moving_instance_max, instance_max, decay)]):
                    inputs = inputs / moving_instance_max
            else:
                print(moving_instance_max)
                inputs = inputs / moving_instance_max

        pad_inputs = tf1.pad(inputs, [[0, 0], [dilation, dilation],
                                     [dilation, dilation], [0, 0]], mode='REFLECT')
        avg_inputs = tf1.nn.pool(pad_inputs, [ksize, ksize],
                                'AVG', 'VALID', dilation_rate=[dilation, dilation])
        alpha = tf1.math.softplus(inputs - avg_inputs)
        beta = tf1.math.softplus(inputs - tf1.reduce_mean(inputs, axis=-1, keepdims=True))
        return alpha, beta

    def d2net_score(self, inputs, ksize=3, need_norm=True, dilation=1, name='conv'):
        channel_wise_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
        beta = inputs / (channel_wise_max + 1e-6)

        if need_norm:
            from tensorflow.python.training.moving_averages import assign_moving_average
            with tf.compat.v1.variable_scope('tower', reuse=self.reuse):
                moving_instance_max = tf.compat.v1.get_variable('%s/instance_max' % name, (),
                                                                initializer=tf.constant_initializer(
                                                                    1),
                                                                trainable=False)
            decay = 0.99

            if self.training:
                instance_max = tf.reduce_max(inputs)
                with tf.control_dependencies([assign_moving_average(moving_instance_max, instance_max, decay)]):
                    exp_logit = tf.exp(inputs / moving_instance_max)
            else:
                exp_logit = tf.exp(inputs / moving_instance_max)
        else:
            exp_logit = tf.exp(inputs)

        pad_exp_logit = tf.pad(exp_logit, [[0, 0], [dilation, dilation],
                                           [dilation, dilation], [0, 0]], constant_values=1)
        sum_logit = tf.nn.pool(pad_exp_logit, [ksize, ksize],
                               'AVG', 'VALID', dilation_rate=[dilation, dilation]) * (ksize ** 2)
        alpha = exp_logit / (sum_logit + 1e-6)
        return alpha, beta

    def extract_kpts(self, score_map, k=256, score_thld=0, edge_thld=0, nms_size=3, eof_size=5):
        h = tf1.shape(score_map)[1]
        w = tf1.shape(score_map)[2]

        mask = score_map > score_thld
        if nms_size > 0:
            nms_mask = tf1.nn.max_pool(
                score_map, ksize=[1, nms_size, nms_size, 1], strides=[1, 1, 1, 1], padding='SAME')
            nms_mask = tf1.equal(score_map, nms_mask)
            mask = tf1.logical_and(nms_mask, mask)
        #if eof_size > 0:
        #    eof_mask = tf1.ones((1, h - 2 * eof_size, w - 2 * eof_size, 1), dtype=tf1.float32)
        #    eof_mask = tf1.pad(eof_mask, [[0, 0], [eof_size, eof_size],
        #                                 [eof_size, eof_size], [0, 0]])
        #    eof_mask = tf1.cast(eof_mask, tf.bool)
        #    mask = tf1.logical_and(eof_mask, mask)
        if edge_thld > 0:
            edge_mask = self.edge_mask(score_map, 1, dilation=1, edge_thld=edge_thld)
            mask = tf1.logical_and(edge_mask, mask)

        zeros = tf1.zeros([h, w], tf1.float32)
        ones = tf1.ones([h, w], tf1.float32)
        mask = tf1.reshape(mask, (h, w))
        mask_val = tf1.where(mask, ones, zeros)
        score_map = tf1.reshape(score_map, (h, w))
        score_map = tf1.math.multiply(score_map, mask_val)
        indices = tf1.where(mask)
        scores = tf1.gather_nd(score_map, indices)
        sample = tf1.argsort(scores, direction='DESCENDING')[0:k]

        indices = tf1.expand_dims(tf1.gather(indices, sample), axis=0)
        
        scores = tf1.expand_dims(tf1.gather(scores, sample), axis=0)
        
        #return indices, score_map
        return indices, scores

    def kpt_refinement(self, inputs):
        n_channel = inputs.get_shape()[-1]

        di_filter = tf.reshape(tf.constant([[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]), (3, 3, 1, 1))
        dj_filter = tf.reshape(tf.constant([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]), (3, 3, 1, 1))
        dii_filter = tf.reshape(tf.constant([[0, 1., 0], [0, -2., 0], [0, 1., 0]]), (3, 3, 1, 1))
        dij_filter = tf.reshape(
            0.25 * tf.constant([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]), (3, 3, 1, 1))
        djj_filter = tf.reshape(tf.constant([[0, 0, 0], [1., -2., 1.], [0, 0, 0]]), (3, 3, 1, 1))

        dii_filter = tf.tile(dii_filter, (1, 1, n_channel, 1))
        dii = tf.nn.depthwise_conv2d(inputs, filter=dii_filter, strides=[
                                     1, 1, 1, 1], padding='SAME')

        dij_filter = tf.tile(dij_filter, (1, 1, n_channel, 1))
        dij = tf.nn.depthwise_conv2d(inputs, filter=dij_filter, strides=[
                                     1, 1, 1, 1], padding='SAME')

        djj_filter = tf.tile(djj_filter, (1, 1, n_channel, 1))
        djj = tf.nn.depthwise_conv2d(inputs, filter=djj_filter, strides=[
                                     1, 1, 1, 1], padding='SAME')

        det = dii * djj - dij * dij

        inv_hess_00 = tf.math.divide_no_nan(djj, det)
        inv_hess_01 = tf.math.divide_no_nan(-dij, det)
        inv_hess_11 = tf.math.divide_no_nan(dii, det)

        di_filter = tf.tile(di_filter, (1, 1, n_channel, 1))
        di = tf.nn.depthwise_conv2d(inputs, filter=di_filter, strides=[1, 1, 1, 1], padding='SAME')

        dj_filter = tf.tile(dj_filter, (1, 1, n_channel, 1))
        dj = tf.nn.depthwise_conv2d(inputs, filter=dj_filter, strides=[1, 1, 1, 1], padding='SAME')

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)

        return tf.stack([step_i, step_j], axis=-1)

    def edge_mask(self, inputs, n_channel, dilation=1, edge_thld=5):
        # non-edge
        dii_filter = tf.reshape(tf.constant([[0, 1., 0], [0, -2., 0], [0, 1., 0]]), (3, 3, 1, 1))
        dij_filter = tf.reshape(
            0.25 * tf.constant([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]), (3, 3, 1, 1))
        djj_filter = tf.reshape(tf.constant([[0, 0, 0], [1., -2., 1.], [0, 0, 0]]), (3, 3, 1, 1))

        dii_filter = tf.tile(dii_filter, (1, 1, n_channel, 1))

        pad_inputs = tf.pad(inputs, [[0, 0], [dilation, dilation], [
                            dilation, dilation], [0, 0]], constant_values=0)

        dii = tf.nn.depthwise_conv2d(pad_inputs, filter=dii_filter, strides=[
                                     1, 1, 1, 1], padding='VALID', dilations=[dilation] * 2)

        dij_filter = tf.tile(dij_filter, (1, 1, n_channel, 1))
        dij = tf.nn.depthwise_conv2d(pad_inputs, filter=dij_filter, strides=[
                                     1, 1, 1, 1], padding='VALID', dilations=[dilation] * 2)

        djj_filter = tf.tile(djj_filter, (1, 1, n_channel, 1))
        djj = tf.nn.depthwise_conv2d(pad_inputs, filter=djj_filter, strides=[
                                     1, 1, 1, 1], padding='VALID', dilations=[dilation] * 2)

        det = dii * djj - dij * dij
        tr = dii + djj
        thld = (edge_thld + 1)**2 / edge_thld
        is_not_edge = tf.logical_and(tr * tr / det <= thld, det > 0)
        return is_not_edge

def interpolate(pos, inputs, batched=True, nd=True):
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]

    i = pos[:, :, 0]
    j = pos[:, :, 1]
    
    i_top_left = tf.clip_by_value(tf.cast(tf.math.floor(i), tf.int32), 0, h - 1)
    j_top_left = tf.clip_by_value(tf.cast(tf.math.floor(j), tf.int32), 0, w - 1)

    i_top_right = tf.clip_by_value(tf.cast(tf.math.floor(i), tf.int32), 0, h - 1)
    j_top_right = tf.clip_by_value(tf.cast(tf.math.ceil(j), tf.int32), 0, w - 1)

    i_bottom_left = tf.clip_by_value(tf.cast(tf.math.ceil(i), tf.int32), 0, h - 1)
    j_bottom_left = tf.clip_by_value(tf.cast(tf.math.floor(j), tf.int32), 0, w - 1)

    i_bottom_right = tf.clip_by_value(tf.cast(tf.math.ceil(i), tf.int32), 0, h - 1)
    j_bottom_right = tf.clip_by_value(tf.cast(tf.math.ceil(j), tf.int32), 0, w - 1)

    dist_i_top_left = i - tf.cast(i_top_left, tf.float32)
    dist_j_top_left = j - tf.cast(j_top_left, tf.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    if nd:
        w_top_left = w_top_left[..., None]
        w_top_right = w_top_right[..., None]
        w_bottom_left = w_bottom_left[..., None]
        w_bottom_right = w_bottom_right[..., None]

    interpolated_val = (
        w_top_left * tf.gather_nd(inputs, tf.stack([i_top_left, j_top_left], axis=-1), batch_dims=1) +
        w_top_right * tf.gather_nd(inputs, tf.stack([i_top_right, j_top_right], axis=-1), batch_dims=1) +
        w_bottom_left * tf.gather_nd(inputs, tf.stack([i_bottom_left, j_bottom_left], axis=-1), batch_dims=1) +
        w_bottom_right * tf.gather_nd(inputs, tf.stack([i_bottom_right, j_bottom_right], axis=-1), batch_dims=1)
    )

    if not batched:
        interpolated_val = tf.squeeze(interpolated_val, axis=0)
    return interpolated_val


