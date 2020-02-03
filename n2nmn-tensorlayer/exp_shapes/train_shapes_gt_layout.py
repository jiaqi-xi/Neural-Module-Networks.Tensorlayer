from __future__ import absolute_import, division, print_function

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import tensorflow as tf
import tensorlayer as tl

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
encoder_dropout = True
decoder_dropout = True
decoder_sampling = True

# Data files
image_sets = ['train.large', 'train.med', 'train.small', 'train.tiny']

# Training parameters
exp_name = "shapes_gt_layout"
snapshot_dir = './exp_shapes/tfmodel/%s/' % exp_name

# Log params
log_dir = './exp_shapes/tb/%s/' % exp_name

# Preparation
[num_questions, training_images, num_batches, num_vocab_txt,
assembler, num_vocab_nmn, text_seq_array, seq_length_array,
gt_layout_array, image_array, vqa_label_array] = Pre(image_sets)

# Network inputs
text_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_batch = tf.placeholder(tf.float32, [None, H_im, W_im, 3])
expr_validity_batch = tf.placeholder(tf.bool, [None])
vqa_label_batch = tf.placeholder(tf.int32, [None])
gt_layout_batch = tf.placeholder(tf.int32, [None, None])
# Prepare for summary
loss_ph = tf.placeholder(tf.float32, [])
entropy_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])

# The model
nmn3_model = NMN3ModelAtt(image_batch, text_seq_batch,
    seq_length_batch, T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim,
    num_layers=num_layers, EOS_idx=assembler.EOS_idx,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    decoder_sampling=decoder_sampling,
    num_choices=num_choices, 
    gt_layout_batch=gt_layout_batch)

compiler = nmn3_model.compiler
scores = nmn3_model.scores
log_seq_prob = nmn3_model.log_seq_prob

# Loss function
avg_sample_loss = tl.cost.cross_entropy(scores, 
    vqa_label_batch,"softmax_loss_per_sample")
seq_likelihood_loss = tf.reduce_mean(-log_seq_prob)

total_training_loss = seq_likelihood_loss + avg_sample_loss
total_loss = total_training_loss + weight_decay * nmn3_model.l2_reg

# Train with Adam
solver = tf.train.AdamOptimizer()
gradients = solver.compute_gradients(total_loss)

# Clip gradient by L2 norm
# gradients = gradients_part1+gradients_part2
gradients = [(tf.clip_by_norm(g, max_grad_l2_norm), v)
             for g, v in gradients]
solver_op = solver.apply_gradients(gradients)

# Training operation
# Partial-run can't fetch training operations
# some workaround to make partial-run work
with tf.control_dependencies([solver_op]):
    train_step = tf.constant(0)

# Write summary to TensorBoard
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

tf.summary.scalar("avg_sample_loss", loss_ph)
tf.summary.scalar("entropy", entropy_ph)
tf.summary.scalar("avg_accuracy", accuracy_ph)
log_step = tf.summary.merge_all()

os.makedirs(snapshot_dir, exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots

tl.layers.initialize_global_variables(sess)

avg_accuracy = 0
accuracy_decay = 0.99

for n_iter in range(max_iter):
    n_begin = int((n_iter % num_batches)*N)
    n_end = int(min(n_begin+N, num_questions))

    # set up input and output tensors
    h = sess.partial_run_setup(
        [nmn3_model.predicted_tokens, nmn3_model.entropy_reg,
         scores, avg_sample_loss, train_step],
        [text_seq_batch, seq_length_batch, image_batch, gt_layout_batch,
         compiler.loom_input_tensor, vqa_label_batch])

    # Part 0 & 1: Run Convnet and generate module layout
    tokens, entropy_reg_val = sess.partial_run(h,
        (nmn3_model.predicted_tokens, nmn3_model.entropy_reg),
        feed_dict={text_seq_batch: text_seq_array[:, n_begin:n_end],
                   seq_length_batch: seq_length_array[n_begin:n_end],
                   image_batch: image_array[n_begin:n_end],
                   gt_layout_batch: gt_layout_array[:, n_begin:n_end]})
    # Assemble the layout tokens into network structure
    expr_list, expr_validity_array = assembler.assemble(tokens)
    # all expr should be valid (since they are ground-truth)
    assert(np.all(expr_validity_array))
    labels = vqa_label_array[n_begin:n_end]
    # Build TensorFlow Fold input for NMN
    expr_feed = compiler.build_feed_dict(expr_list)
    expr_feed[vqa_label_batch] = labels

    # Part 2: Run NMN and learning steps
    scores_val, avg_sample_loss_val, _ = sess.partial_run(
        h, (scores, avg_sample_loss, train_step), feed_dict=expr_feed)

    # compute accuracy
    predictions = np.argmax(scores_val, axis=1)
    accuracy = np.mean(np.logical_and(expr_validity_array,
                                      predictions == labels))
    avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)

    # Add to TensorBoard summary
    if n_iter % log_interval == 0 or (n_iter+1) == max_iter:
        print("iter = %d\n\tloss = %f, accuracy (cur) = %f, "
              "accuracy (avg) = %f, entropy = %f" %
              (n_iter, avg_sample_loss_val, accuracy,
               avg_accuracy, -entropy_reg_val))
        summary = sess.run(log_step, {loss_ph: avg_sample_loss_val,
                                      entropy_ph: -entropy_reg_val,
                                      accuracy_ph: avg_accuracy})
        log_writer.add_summary(summary, n_iter)

    # Save snapshot
    if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)
