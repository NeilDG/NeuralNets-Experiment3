# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os
from model import vanilla_cycle_gan as cg
import constants
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
from utils import plot_utils


class RelightingTrainer:

    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.iteration = opts.iteration
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}

    def update_penalties(self):
        # what penalties to use for losses?
        self.cycle_weight = 10.0

    def train(self, tensor_a, tensor_b):
        print(tensor_a, tensor_b)

        # what to put to losses dict for visdom reporting?

    def visdom_visualize(self, rgb_tensor, albedo_tensor, label = "Training"):
        with torch.no_grad():
            # infer
            self.visdom_reporter.plot_image(rgb_tensor, str(label) + " Input RGB Images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(albedo_tensor, str(label) + " Input Albedo Images - " + constants.RELIGHTING_VERSION + constants.ITERATION)

    def load_saved_state(self, iteration, checkpoint, model_key, optimizer_key):
        self.iteration = iteration
        # load model

    def save_states(self, epoch, iteration, path, model_key, optimizer_key):
        print()
        # save model