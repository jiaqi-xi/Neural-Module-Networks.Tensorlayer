from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tensorflow_fold as td
from tensorflow import convert_to_tensor as to_T

from models_shapes import nmn3_netgen_att
from models_shapes import nmn3_assembler
from models_shapes.shapes_convnet import shapes_convnet
from models_shapes.nmn3_modules import Modules
from util.cnn import fc_layer as fc, conv_layer as conv

class NMN3ModelAtt:
    def __init__(self, image_batch, text_seq_batch, seq_length_batch,T_decoder, 
        num_vocab_txt, embed_dim_txt, num_vocab_nmn,embed_dim_nmn, lstm_dim, 
        num_layers, EOS_idx, encoder_dropout, decoder_dropout, decoder_sampling,
        num_choices, gt_layout_batch=None,
        scope='neural_module_network', reuse=None):

        with tf.variable_scope(scope, reuse=reuse):

            # STEP 1: Get Visual feature by CNN
            with tf.variable_scope('image_feature_cnn'):
                self.image_feat_grid = shapes_convnet(image_batch)

            # STEP 2: Get module layout tokens by Seq2seq RNN
            with tf.variable_scope('layout_generation'):
                att_seq2seq = nmn3_netgen_att.AttentionSeq2Seq(text_seq_batch,
                    seq_length_batch, T_decoder, num_vocab_txt,embed_dim_txt, 
                    num_vocab_nmn, embed_dim_nmn, lstm_dim, num_layers, EOS_idx, 
                    encoder_dropout, decoder_dropout, decoder_sampling,  
                    gt_layout_batch)

                # Set the variables in att_seq2seq
                self.att_seq2seq = att_seq2seq
                self.predicted_tokens = att_seq2seq.predicted_tokens
                self.token_probs = att_seq2seq.token_probs
                self.neg_entropy = att_seq2seq.neg_entropy
                self.word_vecs = att_seq2seq.word_vecs
                self.atts = att_seq2seq.atts

                # Log probability of each generated sequence
                self.log_seq_prob = tf.reduce_sum(tf.log(self.token_probs), axis=0)

            # STEP 3: Build Neural Module Network by assembling different modules
            with tf.variable_scope('layout_execution'):
                self.modules = Modules(self.image_feat_grid, self.word_vecs, num_choices)
                
                # Recursion of the Find & Transform & And modules according to the layout
                # and get output scores for choice with AndModule                
                recursion_result = self.modules.do_recur()
                output_scores = self.modules.get_output_scores(recursion_result)

                # Compile and get the output scores
                self.compiler = td.Compiler.create(output_scores)
                self.scores = self.compiler.output_tensors[0]

            # Step 4: Regularization: Entropy + L2
            self.entropy_reg = tf.reduce_mean(self.neg_entropy)
            module_weights = [v for v in tf.trainable_variables()
                              if (scope in v.op.name and v.op.name.endswith('weights'))]
            self.l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in module_weights])
