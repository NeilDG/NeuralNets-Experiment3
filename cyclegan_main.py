# -*- coding: utf-8 -*-
"""
Main entry for GAN training
Created on Sun Apr 19 13:22:06 2020

@author: delgallegon
"""

from __future__ import print_function
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from loaders import dataset_loader
from trainers import cyclegan_trainer, early_stopper
import constants
from transforms import cyclegan_transforms

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00005")
parser.add_option('--batch_size', type=int, help="Batch size of network before step is performed.", default="128")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--img_per_iter', type=int, help="Num images to load per iteration", default="128")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help="Min epochs", default=120)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.STYLE_TRANSFER_VERSION = opts.version_name
    constants.STYLE_TRANSFER_CHECKPATH = 'checkpoint/' + constants.STYLE_TRANSFER_VERSION + "_" + constants.ITERATION + '.pt'

    # COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers, " ", opts.version_name)
        constants.imgx_dir = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.imgy_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 6/azimuth/*/rgb/*.png"
        constants.imgx_dir_test = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.imgy_dir_test = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 6/azimuth/*/rgb/*.png"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        opts.num_workers = 8
        print("Using CCS configuration. Workers: ", opts.num_workers, " ", opts.version_name)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers, " ", opts.version_name)
        constants.imgx_dir = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.imgy_dir = "/home/neil_delgallego/SynthWeather Dataset 6/azimuth/*/rgb/*.png"

    elif (constants.server_config == 4):
        opts.num_workers = 12
        constants.imgx_dir = "C:/Datasets/Places Dataset/*.jpg"
        constants.imgy_dir = "C:/Datasets/SynthWeather Dataset 6/azimuth/*/rgb/*.png"
        constants.imgx_dir_test = constants.imgx_dir
        constants.imgy_dir_test = constants.imgy_dir

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers, " ", opts.version_name)
    else:
        opts.num_workers = 12
        constants.imgx_dir = "E:/Places Dataset/*.jpg"
        constants.imgy_dir = "E:/SynthWeather Dataset 6/azimuth/*/rgb/*.png"
        constants.imgx_dir_test = constants.imgx_dir
        constants.imgy_dir_test = constants.imgy_dir
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers, " ", opts.version_name)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)
    
    # manualSeed = random.randint(1, 10000) # use if you want new results
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)
    
    gt = cyclegan_trainer.CycleGANTrainer(device, opts)
    start_epoch = 0
    iteration = 0
    last_metric = 10000.0
    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000, last_metric)

    if(opts.load_previous):
        checkpoint = torch.load(constants.STYLE_TRANSFER_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1   
        iteration = checkpoint['iteration'] + 1
        gt.load_saved_state(checkpoint)
 
        print("Loaded checkpt: %s Current epoch: %d" % (constants.STYLE_TRANSFER_CHECKPATH, start_epoch))
        print("===================================================")
    
    # Create the dataloader
    train_loader = dataset_loader.load_da_dataset_train(constants.imgx_dir, constants.imgy_dir, opts)
    test_loader = dataset_loader.load_da_dataset_test(constants.imgx_dir_test, constants.imgy_dir_test, opts)

    if (opts.test_mode == 1):
        print("Plotting test images...")
        imgx_batch, imgy_batch = next(iter(train_loader))
        imgx_tensor = imgx_batch.to(device)
        imgy_tensor = imgy_batch.to(device)

        # gt.train(imgx_tensor, imgy_tensor, 0)
        gt.visdom_visualize(imgx_tensor, imgy_tensor, "Train")

        imgx_batch, imgy_batch = next(iter(test_loader))
        imgx_tensor = imgx_batch.to(device)
        imgy_tensor = imgy_batch.to(device)

        # gt.train(imgx_tensor, imgy_tensor)
        gt.visdom_visualize(imgx_tensor, imgy_tensor, "Test")

    else:
        print("Starting Training Loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                imgx_batch, imgy_batch = train_data
                imgx_tensor = imgx_batch.to(device)
                imgy_tensor = imgy_batch.to(device)

                gt.train(imgx_tensor, imgy_tensor, i)
                iteration = iteration + 1

                x2y, _ = gt.test(imgx_tensor, imgy_tensor)
                stopper_method.test(gt, epoch, iteration, x2y, imgy_tensor)  # stop training if reconstruction no longer becomes close to Y

                if (i % 256 == 0):
                    gt.visdom_visualize(imgx_tensor, imgy_tensor, "Train")

                    gt.save_states_checkpt(epoch, iteration)
                    imgx_batch, imgy_batch = test_data
                    imgx_tensor = imgx_batch.to(device)
                    imgy_tensor = imgy_batch.to(device)

                    gt.visdom_visualize(imgx_tensor, imgy_tensor, "Test")
                    gt.visdom_plot(iteration)

                if (stopper_method.did_stop_condition_met()):
                        break

            if (stopper_method.did_stop_condition_met()):
                break

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

