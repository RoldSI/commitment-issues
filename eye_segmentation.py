import argparse
import gc
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsummary
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from skimage import measure
from torchvision import transforms
from tqdm import tqdm

import utils
from train import MobileNetV2_CS
from utils import OpenEDS, Rescale, ToTensor, Normalize
import time



CHECKPOINT_PATH = '/home/maja/Repository/Eye_VR_Segmentation/checkpoints/checkpoint-subs35-best.pt'
IMAGE_PATH = '/home/maja/Projects/ZEISS/20240427_150511.jpg'

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--image_path', type=str, default=IMAGE_PATH)
    args.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH)
    args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0','mps'])
    args.add_argument('--debug', action='store_true', default=False)
    args = args.parse_args()

    image_path = args.image_path
    target_device = args.device
    device = torch.device(target_device)
    model = MobileNetV2_CS()
    model.to(device)

    print('Load model from {}'.format(CHECKPOINT_PATH))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(target_device))
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_height, image_width = image.shape
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.to(device)
        # average execution time
        if args.debug:
            start_time = time.time()
            for i in range(100):
                output = model(image)
            print("--- %s seconds ---" % ((time.time() - start_time)/100))
        else:
            output = model(image)
        # output = model(image)
        binary_image = output[1].cpu().numpy()[0, :, :]
        probs = output[0].cpu().numpy()[0, :, :, :]
        # Extract masks from binary_image
        masks = []
        for channel in range(probs.shape[0]):
            mask = np.where(binary_image == channel, 1, 0)
            masks.append(mask)
            # Save mask as an image
            mask_image = np.uint8(mask * 255)
            mask_image = cv2.resize(mask_image, (image_width, image_height))
            cv2.imwrite('mask_{}.png'.format(channel), mask_image)

if __name__ == '__main__':
    main()
