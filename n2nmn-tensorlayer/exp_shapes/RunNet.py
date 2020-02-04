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

def RunNet(Sess, Fetches, Feeds, Fetches_cur, Feed_dict, Handle):
    if Handle is None:
        Handle = Sess.partial_run_setup(fetches=Fetches, feeds=Feeds)
    result=Sess.partial_run(handle=Handle, fetches=Fetches_cur, feed_dict=Feed_dict)
    return (Handle, (result))