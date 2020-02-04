from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import argparse
import tensorlayer as tl

# Initialize argparser
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', required=True)
parser.add_argument('--snapshot_name', required=True)
parser.add_argument('--test_split', required=True)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; 
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))
import json

from models_shapes.nmn3_assembler import Assembler
from models_shapes.nmn3_model import NMN3ModelAtt

# Share parameters
from params import *
# Module parameters
encoder_dropout = False
decoder_dropout = False
decoder_sampling = False

exp_name = args.exp_name
snapshot_name = args.snapshot_name
snapshot_file = './exp_shapes/tfmodel/%s/%s' % (exp_name, snapshot_name)
# Data files
image_sets = args.test_split.split(':')

save_dir = './exp_shapes/results/%s/%s.%s' % (exp_name, snapshot_name, '_'.join(image_sets))
save_file = save_dir + '.txt'
os.makedirs(save_dir, exist_ok=True)

# Preparation
[num_questions, training_images, num_batches, num_vocab_txt,
assembler, num_vocab_nmn, text_seq_array, seq_length_array,
gt_layout_array, image_array, vqa_label_array] = Pre(image_sets)

# Network inputs
text_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_batch = tf.placeholder(tf.float32, [None, H_im, W_im, 3])
expr_validity_batch = tf.placeholder(tf.bool, [None])

# The model
nmn3_model = NMN3ModelAtt(image_batch=image_batch, 
    text_seq_batch=text_seq_batch,
    seq_length_batch=seq_length_batch,
    T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim,
    num_layers=num_layers, EOS_idx=assembler.EOS_idx,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    decoder_sampling=decoder_sampling,
    num_choices=num_choices)

compiler = nmn3_model.compiler
scores = nmn3_model.scores

snapshot_saver = tf.train.Saver()

tl.layers.initialize_global_variables(sess)
snapshot_saver.restore(sess, snapshot_file)

answer_correct = 0
layout_correct = 0
layout_valid = 0

for n_iter in range(int(num_batches)):
    n_begin = int((n_iter % num_batches)*N)
    n_end = int(min(n_begin+N, num_questions))

    # set up input and output tensors
    h = sess.partial_run_setup(
        [nmn3_model.predicted_tokens, scores],
        [text_seq_batch, seq_length_batch, image_batch,
         compiler.loom_input_tensor, expr_validity_batch])

    # Part 0 & 1: Run Convnet and generate module layout
    tokens = sess.partial_run(h, nmn3_model.predicted_tokens,
        feed_dict={text_seq_batch: text_seq_array[:, n_begin:n_end],
                   seq_length_batch: seq_length_array[n_begin:n_end],
                   image_batch: image_array[n_begin:n_end]})

    h = None # notice parameter sending
    [h, (tokens)] = RunNet(Sess=sess,
                    Fetches=[nmn3_model.predicted_tokens, scores],
                    Feeds=[text_seq_batch, seq_length_batch, image_batch,
                    compiler.loom_input_tensor, expr_validity_batch],
                    Fetches_cur=nmn3_model.predicted_tokens,
                    Feed_dict={text_seq_batch: text_seq_array[:, n_begin:n_end],
                    seq_length_batch: seq_length_array[n_begin:n_end],
                    image_batch: image_array[n_begin:n_end]},
                    Handle=h)

    # compute the accuracy of the predicted layout
    gt_tokens = gt_layout_array[:, n_begin:n_end]
    layout_correct += np.sum(np.all(np.logical_or(tokens == gt_tokens,
                                                  gt_tokens == assembler.EOS_idx),
                                    axis=0))

    # Assemble the layout tokens into network structure
    expr_list, expr_validity_array = assembler.assemble(tokens)
    layout_valid += np.sum(expr_validity_array)
    labels = vqa_label_array[n_begin:n_end]
    # Build TensorFlow Fold input for NMN
    expr_feed = compiler.build_feed_dict(expr_list)
    expr_feed[expr_validity_batch] = expr_validity_array

    # Part 2: Run NMN and learning steps
    [h, (scores_val)] = RunNet(Sess=sess,
                        Fetches=[nmn3_model.predicted_tokens, scores],
                        Feeds=[text_seq_batch, seq_length_batch, image_batch,
                        compiler.loom_input_tensor, expr_validity_batch],
                        Fetches_cur=scoress,
                        Feed_dict=expr_feed,
                        Handle=h)

    # compute accuracy
    predictions = np.argmax(scores_val, axis=1)
    answer_correct += np.sum(np.logical_and(expr_validity_array,
                                            predictions == labels))

# Print effectiveness to cmd and file
answer_accuracy = answer_correct / num_questions
layout_accuracy = layout_correct / num_questions
layout_validity = layout_valid / num_questions
print("answer accuracy =", answer_accuracy, "on", '_'.join(image_sets))
print("layout accuracy =", layout_accuracy, "on", '_'.join(image_sets))
print("layout validity =", layout_validity, "on", '_'.join(image_sets))
with open(save_file, 'w') as f:
    print("answer accuracy =", answer_accuracy, "on", '_'.join(image_sets), file=f)
    print("layout accuracy =", layout_accuracy, "on", '_'.join(image_sets), file=f)
    print("layout validity =", layout_validity, "on", '_'.join(image_sets), file=f)
