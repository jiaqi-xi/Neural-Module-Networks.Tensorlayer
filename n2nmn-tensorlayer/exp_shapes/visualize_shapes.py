#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd ../


# In[ ]:


from __future__ import absolute_import, division, print_function

gpu_id = 0  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import tensorflow as tf
# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))

from models_shapes.nmn3_assembler import Assembler
from models_shapes.nmn3_model import NMN3ModelAtt


# In[ ]:


# Module parameters
H_im = 30
W_im = 30
num_choices = 2
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 256
num_layers = 2
encoder_dropout = False
decoder_dropout = False
decoder_sampling = False
T_encoder = 15
T_decoder = 8
N = 256

exp_name = "shapes_gt_layout"
snapshot_name = "00040000"
snapshot_file = './exp_shapes/tfmodel/%s/%s' % (exp_name, snapshot_name)

# Data files
vocab_shape_file = './exp_shapes/data/vocabulary_shape.txt'
vocab_layout_file = './exp_shapes/data/vocabulary_layout.txt'
# image_sets = ['train.large', 'train.med', 'train.small', 'train.tiny']
image_sets = ['val']
# image_sets = ['test']
training_text_files = './exp_shapes/shapes_dataset/%s.query_str.txt'
training_image_files = './exp_shapes/shapes_dataset/%s.input.npy'
training_label_files = './exp_shapes/shapes_dataset/%s.output'
image_mean_file = './exp_shapes/data/image_mean.npy'

save_dir = './exp_shapes/results/%s/%s.%s' % (exp_name, snapshot_name + '_vis', '_'.join(image_sets))
os.makedirs(save_dir, exist_ok=True)


# In[ ]:


# Load vocabulary
with open(vocab_shape_file) as f:
    vocab_shape_list = [s.strip() for s in f.readlines()]
vocab_shape_dict = {vocab_shape_list[n]:n for n in range(len(vocab_shape_list))}
num_vocab_txt = len(vocab_shape_list)

assembler = Assembler(vocab_layout_file)
num_vocab_nmn = len(assembler.module_names)

# Load training data
training_questions = []
training_labels = []
training_images_list = []

for image_set in image_sets:
    with open(training_text_files % image_set) as f:
        training_questions += [l.strip() for l in f.readlines()]
    with open(training_label_files % image_set) as f:
        training_labels += [l.strip() == 'true' for l in f.readlines()]
    training_images_list.append(np.load(training_image_files % image_set))

num_questions = len(training_questions)
training_images = np.concatenate(training_images_list)

# Shuffle the training data
# fix random seed for data repeatibility
np.random.seed(3)
shuffle_inds = np.random.permutation(num_questions)

training_questions = [training_questions[idx] for idx in shuffle_inds]
training_labels = [training_labels[idx] for idx in shuffle_inds]
training_images = training_images[shuffle_inds]

# number of training batches
num_batches = np.ceil(num_questions / N)

# Turn the questions into vocabulary indices
text_seq_array = np.zeros((T_encoder, num_questions), np.int32)
seq_length_array = np.zeros(num_questions, np.int32)
for n_q in range(num_questions):
    tokens = training_questions[n_q].split()
    seq_length_array[n_q] = len(tokens)
    for t in range(len(tokens)):
        text_seq_array[t, n_q] = vocab_shape_dict[tokens[t]]
        
image_mean = np.load(image_mean_file)
image_array = (training_images - image_mean).astype(np.float32)
vqa_label_array = np.array(training_labels, np.int32)


# In[ ]:


# Network inputs
text_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_batch = tf.placeholder(tf.float32, [None, H_im, W_im, 3])
expr_validity_batch = tf.placeholder(tf.bool, [None])

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
    num_choices=num_choices)

compiler = nmn3_model.compiler
scores = nmn3_model.scores


# In[ ]:


from models_shapes.nmn3_modules import Modules
from models_shapes.nmn3_assembler import _module_input_num


# In[ ]:


image_feature_grid = nmn3_model.image_feat_grid
word_vecs = nmn3_model.word_vecs
atts = nmn3_model.atts

image_feat_grid_ph = tf.placeholder(tf.float32, image_feature_grid.get_shape())
word_vecs_ph = tf.placeholder(tf.float32, word_vecs.get_shape())
modules = Modules(image_feat_grid_ph, word_vecs_ph, num_choices)

batch_idx = tf.constant([0], tf.int32)
time_idx = tf.placeholder(tf.int32, [1])
input_0 = tf.placeholder(tf.float32, [1, 3, 3, 1])
input_1 = tf.placeholder(tf.float32, [1, 3, 3, 1])


# In[ ]:


# Manually construct each module outside TensorFlow fold for visualization
with tf.variable_scope("neural_module_network/layout_execution", reuse=True):
    FindOutput = modules.FindModule(time_idx, batch_idx)
    TransformOutput = modules.TransformModule(input_0, time_idx, batch_idx)
    AndOutput = modules.AndModule(input_0, input_1, time_idx, batch_idx)
    AnswerOutput = modules.AnswerModule(input_0, time_idx, batch_idx)


# In[ ]:


snapshot_saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
snapshot_saver.restore(sess, snapshot_file)


# In[ ]:


def eval_module(module_name, inputs, t, image_feat_grid_val, word_vecs_val):
    feed_dict = {image_feat_grid_ph: image_feat_grid_val,
                 word_vecs_ph: word_vecs_val,
                 time_idx: [t]}
    # print('evaluating module ' + module_name)
    if 'input_0' in inputs:
        feed_dict[input_0] = inputs['input_0']
    if 'input_1' in inputs:
        feed_dict[input_1] = inputs['input_1']
    if module_name == "_Find":
        result = sess.run(FindOutput, feed_dict)
    elif module_name == "_Transform":
        result = sess.run(TransformOutput, feed_dict)
    elif module_name == "_And":
        result = sess.run(AndOutput, feed_dict)
    elif module_name == "_Answer":
        result = sess.run(AnswerOutput, feed_dict)
    else:
        raise ValueError("invalid module name: " + module_name)

    return result

def eval_expr(layout_tokens, image_feat_grid_val, word_vecs_val):
    invalid_scores = np.array([[0, 0]], np.float32)
    # Decoding Reverse Polish Notation with a stack
    decoding_stack = []
    all_output_stack = []
    for t in range(len(layout_tokens)):
        # decode a module/operation
        module_idx = layout_tokens[t]
        if module_idx == assembler.EOS_idx:
            break
        module_name = assembler.module_names[module_idx]
        input_num = _module_input_num[module_name]

        # Get the input from stack
        inputs = {}
        for n_input in range(input_num-1, -1, -1):
            stack_top = decoding_stack.pop()
            inputs["input_%d" % n_input] = stack_top
        result = eval_module(module_name, inputs, t,
                             image_feat_grid_val, word_vecs_val)
        decoding_stack.append(result)
        all_output_stack.append((t, module_name, result[0]))

    result = decoding_stack[0]
    return result, all_output_stack


# In[ ]:


def expr2str(expr, indent=4):
    name = expr['module']
    input_str = []
    if 'input_0' in expr:
        input_str.append('\n'+' '*indent+expr2str(expr['input_0'], indent+4))
    if 'input_1' in expr:
        input_str.append('\n'+' '*indent+expr2str(expr['input_1'], indent+4))
    expr_str = name[1:]+"["+str(expr['time_idx'])+"]"+"("+", ".join(input_str)+")"
    return expr_str

def visualize(n):
    n_begin = n
    n_end = n + 1

    # set up input and output tensors
    h = sess.partial_run_setup(
        [nmn3_model.predicted_tokens, image_feature_grid, word_vecs, atts, scores],
        [text_seq_batch, seq_length_batch, image_batch,
         compiler.loom_input_tensor, expr_validity_batch])

    # Part 0 & 1: Run Convnet and generate module layout
    tokens, image_feat_grid_val, word_vecs_val, atts_val =         sess.partial_run(h, (nmn3_model.predicted_tokens, image_feature_grid, word_vecs, atts),
        feed_dict={text_seq_batch: text_seq_array[:, n_begin:n_end],
                   seq_length_batch: seq_length_array[n_begin:n_end],
                   image_batch: image_array[n_begin:n_end]})

    # Assemble the layout tokens into network structure
    expr_list, expr_validity_array = assembler.assemble(tokens)
    labels = vqa_label_array[n_begin:n_end].astype(np.bool)

    # Build TensorFlow Fold input for NMN
    expr_feed = compiler.build_feed_dict(expr_list)
    expr_feed[expr_validity_batch] = expr_validity_array

    # Part 2: Run NMN and learning steps
    scores_val = sess.partial_run(h, scores, feed_dict=expr_feed)
    predictions = np.argmax(scores_val, axis=1).astype(np.bool)

    layout_tokens = tokens.T[0]
    _, all_output_stack = eval_expr(layout_tokens, image_feat_grid_val, word_vecs_val)
    
    plt.close('all')
    plt.figure(figsize=(12, 9))
    plt.subplot(3, 3, 1)
    plt.imshow((image_array[n]+image_mean)[:, :, ::-1].astype(np.uint8))
    plt.title(training_questions[n])
    plt.axis('off')
    plt.subplot(3, 3, 2)
    plt.axis('off')
    plt.imshow(np.ones((3, 3, 3), np.float32))
    plt.text(0, 1, 'Predicted layout:\n\n' + expr2str(expr_list[0])+
             '\n\nlabel: '+str(labels[0])+'\nprediction: '+str(predictions[0]))
    plt.subplot(3, 3, 3)
    plt.imshow(atts_val.reshape(atts_val.shape[:2]), interpolation='nearest', cmap='Reds')
    encoder_words = [(vocab_shape_list[w]
                     if n_w < seq_length_array[n_begin] else ' ')
                     for n_w, w in enumerate(text_seq_array[:, n_begin])]
    decoder_words = [(assembler.module_names[w][1:]+'[%d]'%n_w
                      if w != assembler.EOS_idx else '<eos>')
                     for n_w, w in enumerate(layout_tokens)]
    plt.xticks(np.arange(T_encoder), encoder_words, rotation=90)
    plt.yticks(np.arange(T_decoder), decoder_words)
    plt.colorbar()
    for t, module_name, results in all_output_stack:
        result = all_output_stack[0][2]
        plt.subplot(3, 3, t+4)
        if results.ndim > 2:
            plt.imshow(results[..., 0], interpolation='nearest', vmin=-1.5, vmax=1.5, cmap='Reds')
            plt.axis('off')
        else:
            plt.imshow(results.reshape((1, 2)), interpolation='nearest', vmin=-1.5, vmax=1.5, cmap='Reds')
            plt.xticks([0, 1], ['False', 'True'])
        plt.title('output from '+module_name[1:]+"["+str(t)+"]")
        plt.colorbar()


# In[ ]:


for n in range(100):
    print('visualizing %d' % n)
    visualize(n)
    plt.savefig(os.path.join(save_dir, '%08d.jpg' % n))
    plt.close('all')

print('visualizations saved to', save_dir)

