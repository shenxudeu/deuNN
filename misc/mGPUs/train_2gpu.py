"""
This is an example showing how to train a simple MLP on 2 gpus
Adapted from https://github.com/uoguelph-mlrg/theano_multi_gpu/blob/master/dual_mlp.py
A complicated version is the dual-gpu-AlexNet
https://github.com/uoguelph-mlrg/theano_alexnet
"""

import os, sys
import time
from multiprocessing import Process, Queue

import zmq
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

import theano.sandbox.cuda
import theano
import theano.tensor as T

import theano.misc.pycuda_init
import theano.misc.pycuda_utils

from logistic_sgd import load_data
from mlp import MLP

