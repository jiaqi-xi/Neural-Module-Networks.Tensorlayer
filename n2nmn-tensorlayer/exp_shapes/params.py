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
import json

from models_shapes.nmn3_assembler import Assembler
from models_shapes.nmn3_model import NMN3ModelAtt

# Module parameters
H_im = 30
W_im = 30
num_choices = 2
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 256
num_layers = 2
# true/false different
T_encoder = 15
T_decoder = 11
N = 256

# Data files
vocab_shape_file = './exp_shapes/data/vocabulary_shape.txt'
vocab_layout_file = './exp_shapes/data/vocabulary_layout.txt'

training_text_files = './exp_shapes/shapes_dataset/%s.query_str.txt'
training_image_files = './exp_shapes/shapes_dataset/%s.input.npy'
training_label_files = './exp_shapes/shapes_dataset/%s.output'
training_gt_layout_file = './exp_shapes/data/%s.query_layout_symbols.json'
image_mean_file = './exp_shapes/data/image_mean.npy'

# Training parameters
weight_decay = 5e-4
max_grad_l2_norm = 10
max_iter = 40000
snapshot_interval = 10000

# Log params
log_interval = 20

# Load training data
training_questions = []
training_labels = []
training_images_list = []
gt_layout_list = []


def Pre(image_sets):

    for image_set in image_sets:
        with open(training_text_files % image_set) as f:
            training_questions = [l.strip() for l in f.readlines()]
        with open(training_label_files % image_set) as f:
            training_labels = [l.strip() == 'true' for l in f.readlines()]
        training_images_list.append(np.load(training_image_files % image_set))
        with open(training_gt_layout_file % image_set) as f:
            gt_layout_list = json.load(f)

    num_questions = len(training_questions)
    training_images = np.concatenate(training_images_list)

    # Shuffle the training data
    # fix random seed for data repeatibility
    np.random.seed(3)
    shuffle_inds = np.random.permutation(num_questions)

    def shuffle_array(x):
        return [x[i] for i in shuffle_inds]

    training_questions = shuffle_array(training_questions)
    training_labels = shuffle_array(training_labels)
    training_images = shuffle_array(training_images)
    gt_layout_list = shuffle_array(gt_layout_list)

    # number of training batches
    num_batches = np.ceil(num_questions / N)

    # Load vocabulary
    with open(vocab_shape_file) as f:
        vocab_shape_list = [s.strip() for s in f.readlines()]
    vocab_shape_dict = {vocab_shape_list[n]:n for n in range(len(vocab_shape_list))}
    num_vocab_txt = len(vocab_shape_list)

    assembler = Assembler(vocab_layout_file)
    num_vocab_nmn = len(assembler.module_names)

    # Turn the questions into vocabulary indices
    text_seq_array = np.zeros((T_encoder, num_questions), np.int32)
    seq_length_array = np.zeros(num_questions, np.int32)
    gt_layout_array = np.zeros((T_decoder, num_questions), np.int32)

    for n_q in range(num_questions):
        tokens = training_questions[n_q].split()
        seq_length_array[n_q] = len(tokens)
        for t in range(len(tokens)):
            text_seq_array[t, n_q] = vocab_shape_dict[tokens[t]]
        gt_layout_array[:, n_q] = assembler.module_list2tokens(
            gt_layout_list[n_q], T_decoder)

    image_mean = np.load(image_mean_file)
    image_array = (training_images - image_mean).astype(np.float32)
    vqa_label_array = np.array(training_labels, np.int32)
    return (num_questions, training_images, num_batches, num_vocab_txt,
            assembler, num_vocab_nmn, text_seq_array, seq_length_array,
            gt_layout_array, image_array, vqa_label_array)