from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms
import time
import networks
from layers import disp_to_depth
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default="/data/zhangyawei/Lite-HDGF/splits/eigen/test_files.txt")

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        default="/data/zhangyawei/Lite-HDGF/Lite-HDGF")

    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--model', type=str, default="lite-HDGF-small",
                        choices=["lite-HDGF", "lite-HDGF-small", "lite-HDGF-tiny", "lite-HDGF-8m"])
    parser.add_argument('--ext', type=str, default="png")
    parser.add_argument("--no_cuda", action='store_true')
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.load_weights_folder is not None, \
        "You must specify the --load_weights_folder parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.LiteHDGF(model=args.model,
                                    height=feed_height,
                                    width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path) and not args.test:
        paths = [args.image_path]
    elif os.path.isfile(args.image_path) and args.test:
        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        paths = []
        with open(args.image_path) as f:
            filenames = f.readlines()
            for filename in filenames:
                line = filename.split()
                folder = line[0]
                frame_index = int(line[1])
                side = line[2]
                f_str = "{:010d}{}".format(frame_index, '.png')
                image_path = os.path.join('/data/hanxj/kitti', folder, "image_0{}/data".format(side_map[side]), f_str)
                paths.append(image_path)
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    total_time_encoder = 0
    total_time_decoder = 0
    total_time_all = 0

    output_directory = "/data/zhangyawei/Lite-HDGF/keshihua/tp/"
    os.makedirs(output_directory, exist_ok=True)
    npy_directory = "/data/zhangyawei/Lite-HDGF/keshihua/npy/"
    os.makedirs(npy_directory, exist_ok=True)

    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.png"):
                continue

            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            output_name = idx
            input_image = input_image.to(device)

            since = time.time()
            features, _, _ = encoder(input_image)
            time_encoder_all = []
            time_1 = time.time()
            time_encoder = ((time_1 - since) / 16) * 1000
            time_encoder_all.append(time_encoder)

            outputs = depth_decoder(features)
            time_decoder_all = []
            time_2 = time.time()
            time_decoder = ((time_2 - time_1) / 16) * 1000
            time_decoder_all.append(time_decoder)

            total_time_encoder += time_encoder
            total_time_decoder += time_decoder
            total_time_all += time_encoder + time_decoder

            disp = outputs[("disp", 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            name_dest_npy = os.path.join(npy_directory, "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

            time_encoder_all = np.mean(time_encoder_all[0])
            time_decoder_all = np.mean(time_decoder_all[0])
            ALL = time_encoder_all + time_decoder_all
            time_encoder_all = ("%.3f" % time_encoder_all)
            time_decoder_all = ("%.3f" % time_decoder_all)
            ALL = ("%.3f" % ALL)
            print('\t' + ('Speed time of the enoder is {0} ms').format(time_encoder_all))
            print('\t' + 'Speed time of the decoder is {0} ms'.format(time_decoder_all))
            print('\t' + 'Speed time of full model is {0} ms'.format(ALL))

    average_time_encoder = total_time_encoder / len(paths)
    average_time_decoder = total_time_decoder / len(paths)
    average_time_all = total_time_all / len(paths)

    time_encoder_avg = ("%.3f" % average_time_encoder)
    time_decoder_avg = ("%.3f" % average_time_decoder)
    ALL_avg = ("%.3f" % average_time_all)

    print('\t' + ('Average Speed time of the encoder is {0} ms').format(time_encoder_avg))
    print('\t' + 'Average Speed time of the decoder is {0} ms'.format(time_decoder_avg))
    print('\t' + 'Average Speed time of full model is {0} ms'.format(ALL_avg))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
