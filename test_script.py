#Script to use for testing

import os

def test_relighting():
    # os.system("python \"iid_train.py\" --server_config=5 --img_to_load=1000 --load_previous=1 --test_mode=0 --patch_size=64 --batch_size=128 --net_config=1 --num_blocks=6 "
    #           "--plot_enabled=1 --min_epochs=10 --version_name=\"maps2rgb_rgb2maps_v4.06\" --iteration=5")

    # os.system("python \"iid_test.py\"  --net_config=1 --num_blocks=6 --version_name=\"maps2rgb_rgb2maps_v4.08\" --iteration=5 "
    #           "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    # os.system("python \"iid_test.py\"  --net_config=1 --num_blocks=6 --version_name=\"maps2rgb_rgb2maps_v4.08\" --iteration=6 "
    #           "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")
    #
    # os.system("python \"iid_test.py\"  --net_config=1 --num_blocks=6 --version_name=\"maps2rgb_rgb2maps_v4.08\" --iteration=7 "
    #           "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")
    #
    # os.system("python \"iid_test.py\"  --net_config=1 --num_blocks=6 --version_name=\"maps2rgb_rgb2maps_v4.08\" --iteration=8 "
    #           "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")
    #
    #
    os.system("python \"iid_test.py\"  --net_config=2 --num_blocks=0 --version_name=\"maps2rgb_rgb2maps_v4.09\" --iteration=5 "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    os.system("python \"iid_test.py\"  --net_config=2 --num_blocks=0 --version_name=\"maps2rgb_rgb2maps_v4.09\" --iteration=6 "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    os.system("python \"iid_test.py\"  --net_config=2 --num_blocks=0 --version_name=\"maps2rgb_rgb2maps_v4.09\" --iteration=7 "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

    os.system("python \"iid_test.py\"  --net_config=2 --num_blocks=0 --version_name=\"maps2rgb_rgb2maps_v4.09\" --iteration=8 "
              "--input_path=\"E:/IID-TestDataset/GTA/input/\" --output_path=\"E:/IID-TestDataset/GTA/ours/\"")

def main():
    # os.system("python \"processing/dataset_creator.py\"")

    # os.system("python \"defer_render_test.py\" --shadow_multiplier=1.0 --shading_multiplier=1.0")
    #

    # os.system("python \"iid_render_main_2.py\" --num_workers=12 --img_to_load=-1 --patch_size=32 "
    #           "--net_config_a=5 --num_blocks_a=0 --net_config_s1=2 --num_blocks_s1=0 --net_config_s2=1 --num_blocks_s2=6 --net_config_s3=1 --num_blocks_s3=6 "
    #           "--version_albedo=\"rgb2albedo_v7.22\" --iteration_a=8 "
    #           "--version_shading=\"rgb2shading_v8.07\" --iteration_s1=5 "
    #           "--version_shadow=\"rgb2shadow_v8.07\" --iteration_s2=5 "
    #           "--version_shadow_remap=\"shadow2relight_v1.07\" --iteration_s3=8 "
    #           "--mode=azimuth --light_color=\"255,255,255\" --test_code=100")

    test_relighting()



if __name__ == "__main__":
    main()