# -*- coding: utf-8 -*-
# Cycle consistent relighting trainer
import random

import kornia
from model import iteration_table, embedding_network
from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from model import usi3d_gan
from model.modules import image_pool
import constants
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn
from model.iteration_table import IterationTable
from trainers import paired_trainer
from transforms import iid_transforms
from utils import plot_utils
from utils import tensor_utils
from custom_losses import ssim_loss, iid_losses
import lpips

class IIDTrainer:

    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr

        self.iteration = opts.iteration
        self.it_table = iteration_table.IterationTable()
        self.use_bce = self.it_table.is_bce_enabled(self.iteration, IterationTable.NetworkType.ALBEDO)
        self.da_enabled = opts.da_enabled

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.multiscale_grad_loss = iid_losses.MultiScaleGradientLoss(4)
        self.reflect_cons_loss = iid_losses.ReflectConsistentLoss(sample_num_per_area=1, split_areas=(1, 1))

        self.default_light_color = "255,255,255"

        self.D_A_pool = image_pool.ImagePool(50)
        self.D_S_pool = image_pool.ImagePool(50)
        self.D_Z_pool = image_pool.ImagePool(50)

        self.iid_op = iid_transforms.IIDTransform().to(self.gpu_device)

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config
        self.net_config = opts.net_config
        self.albedo_mode = opts.albedo_mode

        if(self.da_enabled == 1):
            self.initialize_da_network(opts.da_version_name)
            self.initialize_shading_network(net_config, num_blocks, 6)
            if(self.albedo_mode >= 1):
                self.initialize_albedo_network(net_config, num_blocks, 6)
                self.initialize_parsing_network(6)
            self.initialize_shadow_network(net_config, num_blocks, 6)
        else:
            self.initialize_shading_network(net_config, num_blocks, 3)
            if (self.albedo_mode >= 1):
                self.initialize_albedo_network(net_config, num_blocks, 3)
                self.initialize_parsing_network(6)
            self.initialize_shadow_network(net_config, num_blocks, 3)

        if(self.albedo_mode == 2):
            self.initialize_unlit_network(3, opts)

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()
        self.optimizerG_shading = torch.optim.Adam(itertools.chain(self.G_S.parameters(), self.G_Z.parameters()), lr=self.g_lr)
        self.optimizerD_shading = torch.optim.Adam(itertools.chain(self.D_S.parameters(), self.D_Z.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_shading, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_shading, patience=100000 / self.batch_size, threshold=0.00005)

        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_da_network(self, da_version_name):
        self.embedder = embedding_network.EmbeddingNetworkFFA(blocks=6).to(self.gpu_device)
        checkpoint = torch.load("checkpoint/" + da_version_name + ".pt", map_location=self.gpu_device)
        self.embedder.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        print("Loaded embedding network: ", da_version_name)

        self.decoder_fixed = embedding_network.DecodingNetworkFFA().to(self.gpu_device)
        print("Loaded fixed decoder network")

    def initialize_shading_network(self, net_config, num_blocks, input_nc):
        if (net_config == 1):
            self.G_S = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_S = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_S = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, use_cbam=True).to(self.gpu_device)
        elif (net_config == 4):
            params = {'dim': 64,  # number of filters in the bottommost layer
                      'mlp_dim': 256,  # number of filters in MLP
                      'style_dim': 8,  # length of style code
                      'n_layer': 3,  # number of layers in feature merger/splitor
                      'activ': 'relu',  # activation function [relu/lrelu/prelu/selu/tanh]
                      'n_downsample': 2,  # number of downsampling layers in content encoder
                      'n_res': num_blocks,  # number of residual blocks in content encoder/decoder
                      'pad_type': 'reflect'}
            self.G_S = usi3d_gan.AdaINGen(input_dim=input_nc, output_dim=3, params=params).to(self.gpu_device)
        else:
            self.G_S = cycle_gan.GeneratorV2(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_S = cycle_gan.Discriminator(input_nc=3).to(self.gpu_device)  # use CycleGAN's discriminator

    def initialize_parsing_network(self, input_nc):
        self.G_P = unet_gan.UNetClassifier(num_channels=input_nc, num_classes=2).to(self.gpu_device)
        self.optimizerP = torch.optim.Adam(itertools.chain(self.G_P.parameters()), lr=self.g_lr)
        self.schedulerP = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerP, patience=100000 / self.batch_size, threshold=0.00005)

    def initialize_albedo_network(self, net_config, num_blocks, input_nc):
        if (net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_A = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, use_cbam=True).to(self.gpu_device)
        elif(net_config == 4):
            params = {'dim': 64,                     # number of filters in the bottommost layer
                      'mlp_dim': 256,                # number of filters in MLP
                      'style_dim': 8,                # length of style code
                      'n_layer': 3,                  # number of layers in feature merger/splitor
                      'activ': 'relu',               # activation function [relu/lrelu/prelu/selu/tanh]
                      'n_downsample': 2,             # number of downsampling layers in content encoder
                      'n_res': num_blocks,                    # number of residual blocks in content encoder/decoder
                      'pad_type': 'reflect'}
            self.G_A = usi3d_gan.AdaINGen(input_dim=input_nc, output_dim=3, params=params).to(self.gpu_device)
        else:
            self.G_A = cycle_gan.GeneratorV2(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(input_nc=3).to(self.gpu_device)  # use CycleGAN's discriminator

        self.optimizerG_albedo = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD_albedo = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG_albedo = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_albedo, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_albedo = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_albedo, patience=100000 / self.batch_size, threshold=0.00005)

    def initialize_unlit_network(self, input_nc, opts):
        checkpoint = torch.load("./checkpoint/" + opts.unlit_checkpt_file, map_location=self.gpu_device)
        net_config = checkpoint['net_config']
        num_blocks = checkpoint['num_blocks']

        if (net_config == 1):
            self.G_unlit = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_unlit = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_unlit = ffa.FFA(gps=input_nc, blocks=num_blocks).to(self.gpu_device)

        print("Loaded unlit network: " + opts.unlit_checkpt_file)

    def initialize_shadow_network(self, net_config, num_blocks, input_nc):
        if (net_config == 1):
            self.G_Z = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_Z = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_Z = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, use_cbam=True).to(self.gpu_device)
        elif (net_config == 4):
            params = {'dim': 64,  # number of filters in the bottommost layer
                      'mlp_dim': 256,  # number of filters in MLP
                      'style_dim': 8,  # length of style code
                      'n_layer': 3,  # number of layers in feature merger/splitor
                      'activ': 'relu',  # activation function [relu/lrelu/prelu/selu/tanh]
                      'n_downsample': 2,  # number of downsampling layers in content encoder
                      'n_res': num_blocks,  # number of residual blocks in content encoder/decoder
                      'pad_type': 'reflect'}
            self.G_Z = usi3d_gan.AdaINGen(input_dim=input_nc, output_dim=3, params=params).to(self.gpu_device)
        else:
            self.G_Z = cycle_gan.GeneratorV2(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_Z = cycle_gan.Discriminator(input_nc=3).to(self.gpu_device)  # use CycleGAN's discriminator

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict_s = {}
        self.losses_dict_s[constants.G_LOSS_KEY] = []
        self.losses_dict_s[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_s[constants.SSIM_LOSS_KEY] = []
        self.GRADIENT_LOSS_KEY = "GRADIENT_LOSS_KEY"
        self.RGB_RECONSTRUCTION_LOSS_KEY = "RGB_RECONSTRUCTION_LOSS_KEY"
        self.losses_dict_s[self.GRADIENT_LOSS_KEY ] = []
        self.losses_dict_s[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[constants.G_LOSS_KEY] = "Shading + Shadow G loss per iteration"
        self.caption_dict_s[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict_s[self.GRADIENT_LOSS_KEY] = "Gradient loss per iteration"
        self.caption_dict_s[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"

        # what to store in visdom?
        self.losses_dict_a = {}
        self.losses_dict_a[constants.G_LOSS_KEY] = []
        self.losses_dict_a[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_a[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_a[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_a[constants.SSIM_LOSS_KEY] = []
        self.losses_dict_a[self.GRADIENT_LOSS_KEY] = []
        self.losses_dict_a[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_a[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_a[constants.D_A_REAL_LOSS_KEY] = []
        self.MS_GRAD_LOSS_KEY = "MS_GRAD_LOSS_KEY"
        self.REFLECTIVE_LOSS_KEY = "REFLECTIVE_LOSS_KEY"
        self.losses_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY] = []
        self.losses_dict_a[self.MS_GRAD_LOSS_KEY] = []
        self.losses_dict_a[self.REFLECTIVE_LOSS_KEY] = []

        self.caption_dict_a = {}
        self.caption_dict_a[constants.G_LOSS_KEY] = "Albedo G loss per iteration"
        self.caption_dict_a[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_a[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_a[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_a[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict_a[self.GRADIENT_LOSS_KEY] = "Gradient loss per iteration"
        self.caption_dict_a[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_a[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_a[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"
        self.caption_dict_a[self.MS_GRAD_LOSS_KEY] = "Multiscale gradient loss per iteration"
        self.caption_dict_a[self.REFLECTIVE_LOSS_KEY] = "Reflective loss per iteration"

        self.losses_dict_p = {}
        self.losses_dict_p[constants.LIKENESS_LOSS_KEY] = []

        self.caption_dict_p = {}
        self.caption_dict_p[constants.LIKENESS_LOSS_KEY] = "Classifier loss per iteration"

    def normalize(self, light_angle):
        std = light_angle / 360.0
        min = -1.0
        max = 1.0
        scaled = std * (max - min) + min

        return scaled

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            return self.mse_loss(pred, target)
        else:
            return self.bce_loss(pred, target)

    def gradient_loss_term(self, pred, target):
        pred_gradient = kornia.filters.spatial_gradient(pred)
        target_gradient = kornia.filters.spatial_gradient(target)

        return self.mse_loss(pred_gradient, target_gradient)

    def lpip_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    def ssim_loss(self, pred, target):
        pred_normalized = (pred * 0.5) + 0.5
        target_normalized = (target * 0.5) + 0.5

        return self.ssim_loss(pred_normalized, target_normalized)

    def update_penalties(self, adv_weight, rgb_l1_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.rgb_l1_weight = rgb_l1_weight

        print("Learning rate for G: ", str(self.g_lr))
        print("Learning rate for D: ", str(self.d_lr))
        print("====================================")
        print("Adv weight: ", str(self.adv_weight))

        print("======ALBEDO======")
        print("RGB reconstruction weight: ", str(self.rgb_l1_weight))
        print("Likeness weight: ", str(self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("LPIP weight: ", str(self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("SSIM weight: ", str(self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("Gradient weight: ", str(self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("Multiscale gradient weight: ", str(self.it_table.get_multiscale_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("Reflective loss weight: ", str(self.it_table.get_reflect_cons_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))

        print("======SHADING======")
        print("Likeness weight: ", str(self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADING)))
        print("LPIP weight: ", str(self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADING)))
        print("SSIM weight: ", str(self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADING)))
        print("Gradient weight: ", str(self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))

        print("======SHADOW======")
        print("Likeness weight: ", str(self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADOW)))
        print("LPIP weight: ", str(self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADOW)))
        print("SSIM weight: ", str(self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADOW)))
        print("Gradient weight: ", str(self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))

    def train(self, input_rgb_tensor, unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor):
        self.train_shading(input_rgb_tensor, albedo_tensor, shading_tensor, shadow_tensor)

        if(self.albedo_mode >= 1):
            self.train_parser(input_rgb_tensor, self.iid_op.create_sky_reflection_masks(albedo_tensor))
            self.train_albedo(input_rgb_tensor, albedo_tensor, unlit_tensor, shading_tensor, shadow_tensor)

    def reshape_input(self, input_tensor):
        rgb_embedding, w1, w2, w3 = self.embedder.get_embedding(input_tensor)
        rgb_feature_rep = self.decoder_fixed.get_decoding(input_tensor, rgb_embedding, w1, w2, w3)

        return torch.cat([input_tensor, rgb_feature_rep], 1)

    def get_feature_rep(self, input_tensor):
        rgb_embedding, w1, w2, w3 = self.embedder.get_embedding(input_tensor)
        rgb_feature_rep = self.decoder_fixed.get_decoding(input_tensor, rgb_embedding, w1, w2, w3)

        return rgb_feature_rep

    def train_shading(self, input_rgb_tensor, albedo_tensor, shading_tensor, shadow_tensor):
        with amp.autocast():
            if (self.da_enabled == 1):
                input = self.reshape_input(input_rgb_tensor)

            self.optimizerD_shading.zero_grad()

            #shading discriminator
            rgb2shading = self.G_S(input)
            self.D_S.train()
            prediction = self.D_S(shading_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_S_real_loss = self.adversarial_loss(self.D_S(shading_tensor), real_tensor) * self.adv_weight
            D_S_fake_loss = self.adversarial_loss(self.D_S_pool.query(self.D_S(rgb2shading.detach())), fake_tensor) * self.adv_weight

            #shadow discriminator
            rgb2shadow = self.G_Z(input)
            self.D_Z.train()
            prediction = self.D_Z(shadow_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_Z_real_loss = self.adversarial_loss(self.D_Z(shadow_tensor), real_tensor) * self.adv_weight
            D_Z_fake_loss = self.adversarial_loss(self.D_Z_pool.query(self.D_Z(rgb2shadow.detach())), fake_tensor) * self.adv_weight

            errD = D_S_real_loss + D_S_fake_loss + D_Z_real_loss + D_Z_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.fp16_scaler.step(self.optimizerD_shading)
                self.schedulerD_shading.step(errD)

            self.optimizerG_shading.zero_grad()

            #shading generator
            self.G_S.train()
            rgb2shading = self.G_S(input)
            S_likeness_loss = self.l1_loss(rgb2shading, shading_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_lpip_loss = self.lpip_loss(rgb2shading, shading_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_ssim_loss = self.ssim_loss(rgb2shading, shading_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_gradient_loss = self.gradient_loss_term(rgb2shading, shading_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADING)
            prediction = self.D_S(rgb2shading)
            real_tensor = torch.ones_like(prediction)
            S_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            # shadow generator
            self.G_Z.train()
            rgb2shadow = self.G_Z(input)
            Z_likeness_loss = self.l1_loss(rgb2shadow, shadow_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            Z_lpip_loss = self.lpip_loss(rgb2shadow, shadow_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            Z_ssim_loss = self.ssim_loss(rgb2shadow, shadow_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            Z_gradient_loss = self.gradient_loss_term(rgb2shadow, shadow_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            prediction = self.D_Z(rgb2shadow)
            real_tensor = torch.ones_like(prediction)
            Z_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_like = self.iid_op.produce_rgb(albedo_tensor, self.G_S(input), self.G_Z(input))
            rgb_l1_loss = self.l1_loss(rgb_like, input_rgb_tensor) * self.rgb_l1_weight

            errG = S_likeness_loss + S_lpip_loss + S_ssim_loss + S_gradient_loss + S_adv_loss + \
                   Z_likeness_loss + Z_lpip_loss + Z_ssim_loss + Z_gradient_loss + Z_adv_loss + rgb_l1_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG_shading)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(S_likeness_loss.item() + Z_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(S_lpip_loss.item() + Z_lpip_loss.item())
            self.losses_dict_s[constants.SSIM_LOSS_KEY].append(S_ssim_loss.item() + Z_ssim_loss.item())
            self.losses_dict_s[self.GRADIENT_LOSS_KEY].append(S_gradient_loss.item() + Z_gradient_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(S_adv_loss.item() + Z_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_S_fake_loss.item() + D_Z_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_S_real_loss.item())
            self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())

    def train_albedo(self, input_rgb_tensor, albedo_tensor, unlit_tensor, shading_tensor, shadow_tensor):
        with amp.autocast():
            if (self.albedo_mode == 2):
                input = unlit_tensor
                # print("Using unlit tensor")
            else:
                input = input_rgb_tensor

            albedo_masks = self.iid_op.create_sky_reflection_masks(albedo_tensor)
            # input_rgb_tensor = input_rgb_tensor * albedo_masks
            albedo_tensor = albedo_tensor * albedo_masks
            albedo_masks = torch.cat([albedo_masks, albedo_masks, albedo_masks], 1)

            if(self.da_enabled == 1):
                input = self.reshape_input(input)

            # produce initial albedo
            rgb2albedo = self.G_A(input)

            # albedo discriminator
            self.D_A.train()
            self.optimizerD_albedo.zero_grad()
            prediction = self.D_A(albedo_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(albedo_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A_pool.query(self.D_A(rgb2albedo.detach())), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()

            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.schedulerD_albedo.step(errD)
                self.fp16_scaler.step(self.optimizerD_albedo)

            self.optimizerG_albedo.zero_grad()

            # albedo generator
            self.G_A.train()
            rgb2albedo = self.G_A(input)
            A_likeness_loss = self.l1_loss(rgb2albedo, albedo_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_lpip_loss = self.lpip_loss(rgb2albedo, albedo_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_ssim_loss = self.ssim_loss(rgb2albedo, albedo_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_gradient_loss = self.gradient_loss_term(rgb2albedo, albedo_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_ms_grad_loss = self.multiscale_grad_loss(rgb2albedo, albedo_tensor, albedo_masks.float()) * self.it_table.get_multiscale_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_reflective_loss = self.reflect_cons_loss(rgb2albedo, albedo_tensor, input_rgb_tensor, albedo_masks.float()) * self.it_table.get_reflect_cons_weight(self.iteration, IterationTable.NetworkType.ALBEDO)

            prediction = self.D_A(rgb2albedo)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_like = self.iid_op.produce_rgb(rgb2albedo, shading_tensor, shadow_tensor)
            rgb_l1_loss = self.l1_loss(rgb_like, input_rgb_tensor) * self.rgb_l1_weight

            errG = A_likeness_loss + A_lpip_loss + A_ssim_loss + A_gradient_loss + A_adv_loss + A_ms_grad_loss + A_reflective_loss + rgb_l1_loss
            self.fp16_scaler.scale(errG).backward()
            self.schedulerG_albedo.step(errG)
            self.fp16_scaler.step(self.optimizerG_albedo)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_a[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_a[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_a[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
            self.losses_dict_a[constants.LPIP_LOSS_KEY].append(A_lpip_loss.item())
            self.losses_dict_a[constants.SSIM_LOSS_KEY].append(A_ssim_loss.item())
            self.losses_dict_a[self.GRADIENT_LOSS_KEY].append(A_gradient_loss.item())
            self.losses_dict_a[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict_a[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict_a[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
            self.losses_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())
            self.losses_dict_a[self.MS_GRAD_LOSS_KEY].append(A_ms_grad_loss.item())
            self.losses_dict_a[self.REFLECTIVE_LOSS_KEY].append(A_reflective_loss.item())

    def train_parser(self, input_rgb_tensor, mask_tensor):
        with amp.autocast():
            if (self.da_enabled == 1):
                input = self.reshape_input(input_rgb_tensor)

            mask_tensor_inv = 1 - mask_tensor
            output = torch.cat([mask_tensor, mask_tensor_inv], 1)
            self.G_P.train()
            self.optimizerP.zero_grad()

            # print("Shapes: ", np.shape(self.G_P(input)), np.shape(output))
            mask_loss = self.bce_loss(self.G_P(input), output)
            self.fp16_scaler.scale(mask_loss).backward()
            self.fp16_scaler.step(self.optimizerP)
            self.schedulerG_shading.step(mask_loss)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_p[constants.LIKENESS_LOSS_KEY].append(mask_loss.item())


    def test(self, input_rgb_tensor):
        with torch.no_grad():
            rgb2shading = self.G_S(input_rgb_tensor)
            rgb2shadow = self.G_Z(input_rgb_tensor)
            # rgb2albedo = self.G_A(input)
            rgb2albedo = self.iid_op.extract_albedo(input_rgb_tensor, rgb2shading, rgb2shadow)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)
        return rgb_like

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_s", iteration, self.losses_dict_s, self.caption_dict_s, constants.IID_CHECKPATH)
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_a, self.caption_dict_a, constants.IID_CHECKPATH)
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_p", iteration, self.losses_dict_p, self.caption_dict_p, constants.IID_CHECKPATH)

    def visdom_visualize(self, input_rgb_tensor, unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor, label = "Train"):
        with torch.no_grad():
            if (self.albedo_mode == 2):
                self.G_unlit.eval()
                # input_rgb_tensor = self.G_unlit(input_rgb_tensor).detach()

            mask_tensor = self.iid_op.create_sky_reflection_masks(albedo_tensor)
            rgb2albedo, rgb2shading, rgb2shadow, rgb2mask = self.decompose(input_rgb_tensor)
            embedding_rep = self.get_feature_rep(input_rgb_tensor)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

            # print("Difference between Albedo vs Recon: ", self.l1_loss(rgb2albedo, albedo_tensor).item())  #0.42321497201919556

            # self.visdom_reporter.plot_image(rgb_noshadow, str(label) + " Input RGB Images - Shadow " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(embedding_rep, str(label) + " Embedding Maps - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(unlit_tensor, str(label) + " Unlit Images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, str(label) + " RGB Reconstruction - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2albedo, str(label) + " RGB2Albedo images - " + constants.IID_VERSION + constants.ITERATION, True)
            self.visdom_reporter.plot_image(albedo_tensor, str(label) + " Albedo images - " + constants.IID_VERSION + constants.ITERATION)

            # print("Sample output: ", rgb2mask[0,0,0,0].item())
            self.visdom_reporter.plot_image(rgb2mask, str(label) + " Albedo-Mask-Like - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(mask_tensor, str(label) + " Albedo Masks - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2shading, str(label) + " RGB2Shading images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(shading_tensor, str(label) + " Shading images - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2shadow, str(label) + " RGB2Shadow images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(shadow_tensor, str(label) + " Shadow images - " + constants.IID_VERSION + constants.ITERATION)

    def visdom_visualize_iid(self, rgb_tensor, albedo_tensor, albedo_mask, shading_tensor, shadow_tensor, label = "Train"):
        self.visdom_reporter.plot_image(rgb_tensor, str(label) + " Input RGB Images - " + constants.IID_VERSION + constants.ITERATION)
        self.visdom_reporter.plot_image(albedo_tensor, str(label) + " Albedo images - " + constants.IID_VERSION + constants.ITERATION)
        self.visdom_reporter.plot_image(albedo_mask, str(label) + " Albedo Masks - " + constants.IID_VERSION + constants.ITERATION)
        self.visdom_reporter.plot_image(shading_tensor, str(label) + " Shading images - " + constants.IID_VERSION + constants.ITERATION)
        self.visdom_reporter.plot_image(shadow_tensor, str(label) + " Shadow images - " + constants.IID_VERSION + constants.ITERATION)

    def visdom_measure(self, input_rgb_tensor, albedo_tensor, shading_tensor, shadow_tensor, label="Training"):
        with torch.no_grad():
            rgb2albedo, rgb2shading, rgb2shadow, rgb2mask = self.decompose(input_rgb_tensor)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

            # plot metrics
            # rgb2albedo = (rgb2albedo * 0.5) + 0.5
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
            rgb2shading = (rgb2shading * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5

            psnr_albedo = np.round(kornia.metrics.psnr(rgb2albedo, albedo_tensor, max_val=1.0).item(), 4)
            ssim_albedo = np.round(1.0 - kornia.losses.ssim_loss(rgb2albedo, albedo_tensor, 5).item(), 4)
            psnr_shading = np.round(kornia.metrics.psnr(rgb2shading, shading_tensor, max_val=1.0).item(), 4)
            ssim_shading = np.round(1.0 - kornia.losses.ssim_loss(rgb2shading, shading_tensor, 5).item(), 4)
            psnr_shadow = np.round(kornia.metrics.psnr(rgb2shadow, shadow_tensor, max_val=1.0).item(), 4)
            ssim_shadow = np.round(1.0 - kornia.losses.ssim_loss(rgb2shadow, shadow_tensor, 5).item(), 4)
            psnr_rgb = np.round(kornia.metrics.psnr(rgb_like, input_rgb_tensor, max_val=1.0).item(), 4)
            ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, input_rgb_tensor, 5).item(), 4)
            display_text = str(label) + " - Versions: " + constants.IID_VERSION + constants.ITERATION + \
                           "<br> Albedo PSNR: " + str(psnr_albedo) + "<br> Albedo SSIM: " + str(ssim_albedo) + \
                           "<br> Shading PSNR: " + str(psnr_shading) + "<br> Shading SSIM: " + str(ssim_shading) + \
                           "<br> Shadow PSNR: " + str(psnr_shadow) + "<br> Shadow SSIM: " + str(ssim_shadow) + \
                           "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ssim_rgb)

            self.visdom_reporter.plot_text(display_text)

    # must have a shading generator network first
    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            rgb2albedo, rgb2shading, rgb2shadow, rgb2mask = self.decompose(rw_tensor)
            embedding_rep = self.get_feature_rep(rw_tensor)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(embedding_rep, "Real World Embeddings - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, "Real World A2B images - " + constants.IID_VERSION + constants.ITERATION)

    def visdom_measure_gta(self, gta_rgb, gta_albedo):
        with torch.no_grad():
            rgb2albedo, rgb2shading, rgb2shadow, rgb2mask = self.decompose(gta_rgb)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

            self.visdom_reporter.plot_image(gta_albedo, "GTA Albedo - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb2albedo, "GTA Albedo-Like - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2shading, "GTA Shading-Like - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(gta_rgb, "GTA RGB - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, "GTA RGB-Like - " + constants.IID_VERSION + constants.ITERATION)

    def decompose(self, rgb_tensor):
        self.G_S.eval()
        self.G_Z.eval()

        with torch.no_grad():
            if (self.da_enabled == 1):
                input = self.reshape_input(rgb_tensor)
            else:
                input = rgb_tensor

            rgb2shading = self.G_S(input)
            rgb2shadow = self.G_Z(input)

            if (self.albedo_mode == 1):
                self.G_A.eval()
                rgb2albedo = self.G_A(input)
            elif (self.albedo_mode == 2):
                self.G_A.eval()
                rgb2mask = self.G_P(input)
                rgb2mask = torch.round(rgb2mask)[:,0,:,:]
                rgb2mask = torch.unsqueeze(rgb2mask, 1)

                input = self.reshape_input(rgb_tensor)
                rgb2albedo = self.G_A(input)

                #normalize to 0-1
                rgb_tensor = tensor_utils.normalize_to_01(rgb_tensor)
                rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)

                rgb2albedo = rgb2albedo * rgb2mask
                rgb2albedo = self.iid_op.mask_fill_nonzeros(rgb2albedo)

            else:
                rgb2albedo = self.iid_op.extract_albedo(rgb_tensor, rgb2shading, rgb2shadow)

        return rgb2albedo, rgb2shading, rgb2shadow, rgb2mask

    def infer_shading(self, rw_tensor):
        self.G_S.eval()
        with torch.no_grad():
            if (self.da_enabled == 1):
                rw_tensor = self.reshape_input(rw_tensor)

            return self.G_S(rw_tensor)

    def load_saved_state(self, checkpoint):
        if(self.albedo_mode >= 1):
            self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
            self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.G_S.load_state_dict(checkpoint[constants.GENERATOR_KEY + "S"])
        self.D_S.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "S"])
        self.G_Z.load_state_dict(checkpoint[constants.GENERATOR_KEY + "Z"])
        self.D_Z.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "Z"])

        if (self.albedo_mode >= 1):
            self.optimizerG_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"])
            self.optimizerD_albedo.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"])
            self.schedulerG_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "A"])
            self.schedulerD_albedo.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "A"])

        if (self.albedo_mode == 2):
            print("Loaded P network")
            self.G_P.load_state_dict(checkpoint[constants.GENERATOR_KEY + "P"])
            self.optimizerP.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "P"])
            self.schedulerP.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "P"])

        self.optimizerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"])
        self.optimizerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"])
        self.schedulerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "S"])
        self.schedulerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "S"])

    def save_states_checkpt(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}

        if(self.albedo_mode >= 1):
            netGA_state_dict = self.G_A.state_dict()
            netDA_state_dict = self.D_A.state_dict()
            optimizerGalbedo_state_dict = self.optimizerG_albedo.state_dict()
            optimizerDalbedo_state_dict = self.optimizerD_albedo.state_dict()
            schedulerGalbedo_state_dict = self.schedulerG_albedo.state_dict()
            schedulerDalbedo_state_dict = self.schedulerD_albedo.state_dict()
            save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
            save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
            save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerGalbedo_state_dict
            save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerDalbedo_state_dict
            save_dict[constants.GENERATOR_KEY + "scheduler" + "A"] = schedulerGalbedo_state_dict
            save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = schedulerDalbedo_state_dict

        if (self.albedo_mode == 2):
            netGP_state_dict = self.G_P.state_dict()
            optimizerP_state_dict = self.optimizerP.state_dict()
            schedulerP_state_dict = self.schedulerP.state_dict()
            save_dict[constants.GENERATOR_KEY + "P"] = netGP_state_dict
            save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "P"] = optimizerP_state_dict
            save_dict[constants.GENERATOR_KEY + "scheduler" + "P"] = schedulerP_state_dict

        netGS_state_dict = self.G_S.state_dict()
        netDS_state_dict = self.D_S.state_dict()
        netGZ_state_dict = self.G_Z.state_dict()
        netDZ_state_dict = self.D_Z.state_dict()

        optimizerGshading_state_dict = self.optimizerG_shading.state_dict()
        optimizerDshading_state_dict = self.optimizerD_shading.state_dict()
        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()

        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict
        save_dict[constants.GENERATOR_KEY + "Z"] = netGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "Z"] = netDZ_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "S"] = schedulerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "S"] = schedulerDshading_state_dict

        torch.save(save_dict, constants.IID_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}

        if (self.albedo_mode >= 1):
            netGA_state_dict = self.G_A.state_dict()
            netDA_state_dict = self.D_A.state_dict()
            optimizerGalbedo_state_dict = self.optimizerG_albedo.state_dict()
            optimizerDalbedo_state_dict = self.optimizerD_albedo.state_dict()
            schedulerGalbedo_state_dict = self.schedulerG_albedo.state_dict()
            schedulerDalbedo_state_dict = self.schedulerD_albedo.state_dict()
            save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
            save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
            save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerGalbedo_state_dict
            save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerDalbedo_state_dict
            save_dict[constants.GENERATOR_KEY + "scheduler" + "A"] = schedulerGalbedo_state_dict
            save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = schedulerDalbedo_state_dict

        if(self.albedo_mode == 2):
            netGP_state_dict = self.G_P.state_dict()
            optimizerP_state_dict = self.optimizerP.state_dict()
            schedulerP_state_dict = self.schedulerP.state_dict()
            save_dict[constants.GENERATOR_KEY + "P"] = netGP_state_dict
            save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY +  "P"] = optimizerP_state_dict
            save_dict[constants.GENERATOR_KEY + "scheduler" + "P"] = schedulerP_state_dict

        netGS_state_dict = self.G_S.state_dict()
        netDS_state_dict = self.D_S.state_dict()
        netGZ_state_dict = self.G_Z.state_dict()
        netDZ_state_dict = self.D_Z.state_dict()

        optimizerGshading_state_dict = self.optimizerG_shading.state_dict()
        optimizerDshading_state_dict = self.optimizerD_shading.state_dict()
        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()

        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict
        save_dict[constants.GENERATOR_KEY + "Z"] = netGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "Z"] = netDZ_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "S"] = schedulerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "S"] = schedulerDshading_state_dict

        torch.save(save_dict, constants.IID_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))