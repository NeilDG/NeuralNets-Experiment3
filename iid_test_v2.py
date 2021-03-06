import glob
import itertools
import sys
from optparse import OptionParser
import random

import cv2
import kornia
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from config import iid_server_config
from loaders import dataset_loader
from trainers import iid_trainer
from trainers import early_stopper
from transforms import iid_transforms
import constants
from utils import plot_utils, tensor_utils
from trainers import trainer_factory
from trainers import albedo_trainer, shading_trainer, albedo_mask_trainer

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--version', type=str, default="")
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--input_path', type=str)
parser.add_option('--output_path', type=str)
parser.add_option('--img_size', type=int, default=(256, 256))

def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.plot_enabled = opts.plot_enabled

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/unlit/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/unlit/"
        constants.DATASET_PLACES_PATH = constants.rgb_dir_ws

        print("Using CCS configuration. Workers: ", opts.num_workers)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/home/neil_delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.albedo_dir = "/home/neil_delgallego/SynthWeather Dataset 8/albedo/"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "D:/Datasets/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "D:/Datasets/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.albedo_dir = "D:/Datasets/SynthWeather Dataset 8/albedo/"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "E:/SynthWeather Dataset 8/unlit/"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers)

def measure_performance():
    visdom_reporter = plot_utils.VisdomReporter()

    GTA_BASE_PATH = "E:/IID-TestDataset/GTA/"
    RGB_PATH = GTA_BASE_PATH + "/input/"
    ALBEDO_PATH = GTA_BASE_PATH + "/albedo/"

    RESULT_A_PATH = GTA_BASE_PATH + "/li_eccv18/"
    RESULT_B_PATH = GTA_BASE_PATH + "/yu_cvpr19/"
    RESULT_C_PATH = GTA_BASE_PATH + "/yu_eccv20/"
    RESULT_D_PATH = GTA_BASE_PATH + "/zhu_iccp21/"
    RESULT_E_PATH = GTA_BASE_PATH + "/ours/"

    rgb_list = glob.glob(RGB_PATH + "*.png")
    albedo_list = glob.glob(ALBEDO_PATH + "*.png")
    a_list = glob.glob(RESULT_A_PATH + "*.png")
    b_list = glob.glob(RESULT_B_PATH + "*.png")
    c_list = glob.glob(RESULT_C_PATH + "*.png")
    d_list = glob.glob(RESULT_D_PATH + "*.png")
    e_list = glob.glob(RESULT_E_PATH + "*.png")

    IMG_SIZE = (320, 240)

    albedo_tensor = None
    albedo_a_tensor = None
    albedo_b_tensor = None
    albedo_c_tensor = None
    albedo_d_tensor = None
    albedo_e_tensor = None


    for i, (rgb_path, albedo_path, a_path, b_path, c_path, d_path, e_path) in enumerate(zip(rgb_list, albedo_list, a_list, b_list, c_list, d_list, e_list)):
        albedo_img = tensor_utils.load_metric_compatible_albedo(albedo_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_a_img = tensor_utils.load_metric_compatible_albedo(a_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_b_img = tensor_utils.load_metric_compatible_albedo(b_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_c_img = tensor_utils.load_metric_compatible_albedo(c_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_d_img = tensor_utils.load_metric_compatible_albedo(d_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_e_img = tensor_utils.load_metric_compatible_albedo(e_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)

        psnr_albedo_a = np.round(kornia.metrics.psnr(albedo_a_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_a = np.round(1.0 - kornia.losses.ssim_loss(albedo_a_img, albedo_img, 5).item(), 4)
        psnr_albedo_b = np.round(kornia.metrics.psnr(albedo_b_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_b = np.round(1.0 - kornia.losses.ssim_loss(albedo_b_img, albedo_img, 5).item(), 4)
        psnr_albedo_c = np.round(kornia.metrics.psnr(albedo_c_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_c = np.round(1.0 - kornia.losses.ssim_loss(albedo_c_img, albedo_img, 5).item(), 4)
        psnr_albedo_d = np.round(kornia.metrics.psnr(albedo_d_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_d = np.round(1.0 - kornia.losses.ssim_loss(albedo_d_img, albedo_img, 5).item(), 4)
        psnr_albedo_e = np.round(kornia.metrics.psnr(albedo_e_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_e = np.round(1.0 - kornia.losses.ssim_loss(albedo_e_img, albedo_img, 5).item(), 4)
        display_text = "Image " +str(i)+ " Albedo <br>" \
                                     "li_eccv18 PSNR: " + str(psnr_albedo_a) + "<br> SSIM: " + str(ssim_albedo_a) + "<br>" \
                                     "yu_cvpr19 PSNR: " + str(psnr_albedo_b) + "<br> SSIM: " + str(ssim_albedo_b) + "<br>" \
                                     "yu_eccv20 PSNR: " + str(psnr_albedo_c) + "<br> SSIM: " + str(ssim_albedo_c) + "<br>" \
                                     "zhu_iccp21 PSNR: " + str(psnr_albedo_d) + "<br> SSIM: " + str(ssim_albedo_d) + "<br>" \
                                     "Ours PSNR: " + str(psnr_albedo_e) + "<br> SSIM: " + str(ssim_albedo_e) + "<br>"

        visdom_reporter.plot_text(display_text)

        if(i == 0):
            albedo_tensor = albedo_img
            albedo_a_tensor = albedo_a_img
            albedo_b_tensor = albedo_b_img
            albedo_c_tensor = albedo_c_img
            albedo_d_tensor = albedo_d_img
            albedo_e_tensor = albedo_e_img
        else:
            albedo_tensor = torch.cat([albedo_tensor, albedo_img], 0)
            albedo_a_tensor = torch.cat([albedo_a_tensor, albedo_a_img], 0)
            albedo_b_tensor = torch.cat([albedo_b_tensor, albedo_b_img], 0)
            albedo_c_tensor = torch.cat([albedo_c_tensor, albedo_c_img], 0)
            albedo_d_tensor = torch.cat([albedo_d_tensor, albedo_d_img], 0)
            albedo_e_tensor = torch.cat([albedo_e_tensor, albedo_e_img], 0)

    psnr_albedo_a = np.round(kornia.metrics.psnr(albedo_a_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_a = np.round(1.0 - kornia.losses.ssim_loss(albedo_a_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_b = np.round(kornia.metrics.psnr(albedo_b_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_b = np.round(1.0 - kornia.losses.ssim_loss(albedo_b_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_c = np.round(kornia.metrics.psnr(albedo_c_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_c = np.round(1.0 - kornia.losses.ssim_loss(albedo_c_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_d = np.round(kornia.metrics.psnr(albedo_d_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_d = np.round(1.0 - kornia.losses.ssim_loss(albedo_d_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_e = np.round(kornia.metrics.psnr(albedo_e_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_e = np.round(1.0 - kornia.losses.ssim_loss(albedo_e_tensor, albedo_tensor, 5).item(), 4)
    display_text = str(constants.IID_VERSION) + str(constants.ITERATION) + "<br>" \
                   "Mean Albedo PSNR, SSIM: <br>" \
                    "li_eccv18 PSNR: " + str(psnr_albedo_a) + "<br> SSIM: " + str(ssim_albedo_a) + "<br>" \
                    "yu_cvpr19 PSNR: " + str(psnr_albedo_b) + "<br> SSIM: " + str(ssim_albedo_b) + "<br>" \
                    "yu_eccv20 PSNR: " + str(psnr_albedo_c) + "<br> SSIM: " + str(ssim_albedo_c) + "<br>" \
                    "zhu_iccp21 PSNR: " + str(psnr_albedo_d) + "<br> SSIM: " + str(ssim_albedo_d) + "<br>" + \
                    "Ours PSNR: " + str(psnr_albedo_e) + "<br> SSIM: " + str(ssim_albedo_e) + "<br>"

    visdom_reporter.plot_text(display_text)

    visdom_reporter.plot_image(albedo_tensor, "Albedo GT")
    visdom_reporter.plot_image(albedo_a_tensor, "Albedo li_eccv18")
    visdom_reporter.plot_image(albedo_b_tensor, "Albedo yu_cvpr19")
    visdom_reporter.plot_image(albedo_c_tensor, "Albedo yu_eccv20")
    visdom_reporter.plot_image(albedo_d_tensor, "Albedo zhu_iccp21")
    visdom_reporter.plot_image(albedo_e_tensor, "Albedo Ours")

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    print(constants.rgb_dir_ws, constants.albedo_dir)
    plot_utils.VisdomReporter.initialize()
    visdom_reporter = plot_utils.VisdomReporter.getInstance()

    iid_server_config.IIDServerConfig.initialize(opts.version)
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_network_config_from_version(opts.version)
    print("General config:", general_config)
    print("Network config: ", network_config)

    iid_op = iid_transforms.IIDTransform()

    test_loader = dataset_loader.load_iid_datasetv2_test(constants.rgb_dir_ws, constants.rgb_dir_ns, constants.unlit_dir, constants.albedo_dir, 256, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    tf = trainer_factory.TrainerFactory(device, opts)
    mask_t, albedo_t, shading_t = tf.get_all_trainers()

    for i, (test_data, rw_data) in enumerate(zip(test_loader, itertools.cycle(rw_loader))):
        with torch.no_grad():
            _, rgb_ws_batch, rgb_ns_batch, albedo_batch, unlit_batch = test_data
            rgb_ws_tensor = rgb_ws_batch.to(device)
            rgb_ns_tensor = rgb_ns_batch.to(device)
            albedo_tensor = albedo_batch.to(device)
            unlit_tensor = unlit_batch.to(device)
            rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)

            input = {"rgb" : rgb_ws_tensor, "unlit" : unlit_tensor, "albedo" : albedo_tensor}
            rgb2mask = mask_t.test(input)
            rgb2albedo = albedo_t.test(input)
            rgb2shading, rgb2shadow = shading_t.test(input)
            rgb_like = iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

            #normalize everything
            rgb_ws_tensor = tensor_utils.normalize_to_01(rgb_ws_tensor)
            shading_tensor = tensor_utils.normalize_to_01(shading_tensor)
            shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)
            albedo_tensor = tensor_utils.normalize_to_01(albedo_tensor)
            rgb2shading = tensor_utils.normalize_to_01(rgb2shading)
            rgb2shadow = tensor_utils.normalize_to_01(rgb2shadow)
            rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
            rgb2albedo = rgb2albedo * rgb2mask
            rgb2albedo = iid_op.mask_fill_nonzeros(rgb2albedo)

            visdom_reporter.plot_image(rgb_ws_tensor, "Input RGB Images - " + opts.version + str(opts.iteration))
            visdom_reporter.plot_image(rgb_like, "RGB Reconstruction Images - " + opts.version + str(opts.iteration))
            visdom_reporter.plot_image(unlit_tensor, "Input Unlit Images - " + opts.version + str(opts.iteration))

            visdom_reporter.plot_image(albedo_tensor, "GT Albedo - " + opts.version + str(opts.iteration))
            visdom_reporter.plot_image(rgb2albedo, "RGB2Albedo - " + opts.version + str(opts.iteration))

            visdom_reporter.plot_image(shading_tensor, "GT Shading - " + opts.version + str(opts.iteration))
            visdom_reporter.plot_image(rgb2shading, "RGB2Shading - " + opts.version + str(opts.iteration))

            visdom_reporter.plot_image(shadow_tensor, "GT Shadow - " + opts.version + str(opts.iteration))
            visdom_reporter.plot_image(rgb2shadow, "RGB2Shadow - " + opts.version + str(opts.iteration))

            psnr_albedo = np.round(kornia.metrics.psnr(rgb2albedo, albedo_tensor, max_val=1.0).item(), 4)
            ssim_albedo = np.round(1.0 - kornia.losses.ssim_loss(rgb2albedo, albedo_tensor, 5).item(), 4)
            psnr_shading = np.round(kornia.metrics.psnr(rgb2shading, shading_tensor, max_val=1.0).item(), 4)
            ssim_shading = np.round(1.0 - kornia.losses.ssim_loss(rgb2shading, shading_tensor, 5).item(), 4)
            psnr_shadow = np.round(kornia.metrics.psnr(rgb2shadow, shadow_tensor, max_val=1.0).item(), 4)
            ssim_shadow = np.round(1.0 - kornia.losses.ssim_loss(rgb2shadow, shadow_tensor, 5).item(), 4)
            psnr_rgb = np.round(kornia.metrics.psnr(rgb_like, rgb_ws_tensor, max_val=1.0).item(), 4)
            ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, rgb_ws_tensor, 5).item(), 4)
            display_text = "Test Set - Versions: " + opts.version + "_" + str(opts.iteration) + \
                           "<br> Albedo PSNR: " + str(psnr_albedo) + "<br> Albedo SSIM: " + str(ssim_albedo) + \
                           "<br> Shading PSNR: " + str(psnr_shading) + "<br> Shading SSIM: " + str(ssim_shading) + \
                           "<br> Shadow PSNR: " + str(psnr_shadow) + "<br> Shadow SSIM: " + str(ssim_shadow) + \
                           "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ssim_rgb)

            visdom_reporter.plot_text(display_text)
            break

    #check RW performance
    _, input_rgb_batch = next(iter(rw_loader))
    input_rgb_tensor = input_rgb_batch.to(device)
    rgb2unlit = tf.get_unlit_network()(input_rgb_tensor)
    input = {"rgb": input_rgb_tensor, "unlit": rgb2unlit}
    rgb2mask = mask_t.test(input)
    rgb2albedo = albedo_t.test(input)
    rgb2shading, rgb2shadow = shading_t.test(input)

    # normalize everything
    input_rgb_tensor = tensor_utils.normalize_to_01(input_rgb_tensor)
    rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
    rgb2albedo = rgb2albedo * rgb2mask
    rgb2albedo = iid_op.mask_fill_nonzeros(rgb2albedo)
    rgb_like = iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

    visdom_reporter.plot_image(input_rgb_tensor, "RW RGB Images - " + opts.version + str(opts.iteration))
    visdom_reporter.plot_image(rgb_like, "RW RGB Reconstruction Images - " + opts.version + str(opts.iteration))
    visdom_reporter.plot_image(rgb2unlit, "RW RGB2Unlit Images - " + opts.version + str(opts.iteration))
    visdom_reporter.plot_image(rgb2albedo, "RW RGB2Albedo - " + opts.version + str(opts.iteration))
    visdom_reporter.plot_image(rgb2shading, "RW RGB2Shading - " + opts.version + str(opts.iteration))
    visdom_reporter.plot_image(rgb2shadow, "RW RGB2Shadow - " + opts.version + str(opts.iteration))

    #measure GTA performance
    img_list = glob.glob(opts.input_path + "*.jpg") + glob.glob(opts.input_path + "*.png")
    print("Images found: ", len(img_list))

    normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    for i, input_path in enumerate(img_list, 0):
        filename = input_path.split("\\")[-1]
        input_rgb_tensor = tensor_utils.load_metric_compatible_img(input_path, cv2.COLOR_BGR2RGB, True, True, opts.img_size).to(device)
        input_rgb_tensor = normalize_op(input_rgb_tensor)
        rgb2unlit = tf.get_unlit_network()(input_rgb_tensor)

        input = {"rgb": input_rgb_tensor, "unlit": rgb2unlit}
        rgb2mask = mask_t.test(input)
        rgb2albedo = albedo_t.test(input)
        rgb2shading, rgb2shadow = shading_t.test(input)

        # normalize everything
        rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
        rgb2albedo = rgb2albedo * rgb2mask
        rgb2albedo = iid_op.mask_fill_nonzeros(rgb2albedo)
        rgb_like = iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

        vutils.save_image(rgb2albedo.squeeze(), opts.output_path + filename)

    measure_performance()

if __name__ == "__main__":
    main(sys.argv)