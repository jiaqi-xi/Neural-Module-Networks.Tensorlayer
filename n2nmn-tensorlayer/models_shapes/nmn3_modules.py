from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T
import tensorflow_fold as td
from models_shapes import nmn3_assembler
from models_shapes.nmn3_layers import fc_layer as fc, conv_layer as conv,\
     empty_safe_1x1_conv as _1x1_conv, empty_safe_conv as  _conv, _Conv2DGrad as _Conv2DGrad

class Modules:
    def __init__(self, image_feat_grid, word_vecs, num_choices):
        self.image_feat_grid = image_feat_grid
        self.word_vecs = word_vecs
        self.num_choices = num_choices

    def _1x1_(self, fctext, text_param, N, x,map_dim=500):
        text_param_mapped = fc(fctext, text_param, output_dim=map_dim)
        text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))
        eltwise_mult = tf.nn.l2_normalize(x * text_param_mapped, 3)
        att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)
        return att_grid

    # All the layers are wrapped with td.ScopedLayer
    def FindModule(self, time_idx, batch_idx, map_dim=500, scope='FindModule',
        reuse=None):

        image_feat_grid = tf.gather(self.image_feat_grid, batch_idx)
        text_param = tf.gather_nd(self.word_vecs, tf.stack([time_idx, batch_idx], axis=1))
        with tf.variable_scope(scope, reuse=reuse):
            N = tf.shape(time_idx)[0]
            image_feat_mapped = _1x1_conv('conv_image', image_feat_grid,
                                          output_dim=map_dim)

            att_grid = self._1x1_('fc_text', text_param, N, image_feat_mapped, map_dim)

        return att_grid

    def TransformModule(self, input_0, time_idx, batch_idx, kernel_size=3, map_dim=500, scope='TransformModule', reuse=None):

        text_param = tf.gather_nd(self.word_vecs, tf.stack([time_idx, batch_idx], axis=1))
        with tf.variable_scope(scope, reuse=reuse):
            att_shape = tf.shape(input_0)
            N = att_shape[0]
            att_maps = _conv('conv_maps', input_0, kernel_size=kernel_size,
                stride=1, output_dim=map_dim)

            att_grid = self._1x1_('text_fc', text_param, N, att_maps,map_dim)

        return att_grid

    def AndModule(self, input_0, input_1, time_idx, batch_idx,scope='AndModule', reuse=None):

        return tf.minimum(input_0, input_1)


    def AnswerModule(self, input_0, time_idx, batch_idx, scope='AnswerModule', reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            att_min = tf.reduce_min(input_0, axis=[1, 2])
            att_avg = tf.reduce_mean(input_0, axis=[1, 2])
            att_max = tf.reduce_max(input_0, axis=[1, 2])
            # att_reduced has shape [N, 3]
            att_reduced = tf.concat([att_min, att_avg, att_max], axis=1)
            scores = fc('fc_scores', att_reduced, output_dim=self.num_choices)

        return scores


    def do_recur(self):
        # This does Recursion of the FindModule & TransformModule & AndModule
        att_shape = self.image_feat_grid.get_shape().as_list()[1:-1] + [1]

        # Forward declaration of module recursion
        att_expr_decl = td.ForwardDeclaration(td.PyObjectType(), td.TensorType(att_shape))

        # FindModule
        fl_find = td.Record([('time_idx', td.Scalar(dtype='int32')),
                             ('batch_idx', td.Scalar(dtype='int32'))])
        fl_find = fl_find >> \
                  td.ScopedLayer(self.FindModule, name_or_scope='FindModule')

        # TransformModule
        fl_transform = td.Record([('input_0', att_expr_decl()),
                                  ('time_idx', td.Scalar('int32')),
                                  ('batch_idx', td.Scalar('int32'))])
        fl_transform = fl_transform >> \
                       td.ScopedLayer(self.TransformModule, name_or_scope='TransformModule')

        # AndModule
        fl_and = td.Record([('input_0', att_expr_decl()),
                            ('input_1', att_expr_decl()),
                            ('time_idx', td.Scalar('int32')),
                            ('batch_idx', td.Scalar('int32'))])
        fl_and = fl_and >> \
                 td.ScopedLayer(self.AndModule, name_or_scope='AndModule')

        recursion_result = td.OneOf(td.GetItem('module'),
                                    {'_Find': fl_find,
                                     '_Transform': fl_transform,
                                     '_And': fl_and})
        att_expr_decl.resolve_to(recursion_result)
        return recursion_result

    def get_output_scores(self, recursion_result):
        # For valid expressions, output scores for choice with AnswerModule
        predicted_scores = td.Record([('input_0', recursion_result),
                                      ('time_idx', td.Scalar('int32')),
                                      ('batch_idx', td.Scalar('int32'))])
        predicted_scores = predicted_scores >> \
                           td.ScopedLayer(self.AnswerModule, name_or_scope='AnswerModule')

        # For invalid expressions, define a dummy answer so that all answers have the same form
        dummy_scores = td.Void() >> td.FromTensor(np.zeros(self.num_choices, np.float32))
        output_scores = td.OneOf(td.GetItem('module'),
                                 {'_Answer': predicted_scores,
                                  nmn3_assembler.INVALID_EXPR: dummy_scores})
        return output_scores
