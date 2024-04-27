#! /usr/bin/env python3

import argparse
from image_api import fetch_image
from eye_segmentation_onnx import model
import cv2
import time
import sys

def main():
    segmentation_type = None
    segmentation_model_path = None
    image_path = None
    live = False
    verbose = True

    args = argparse.ArgumentParser()
    args.add_argument('--segmentation_type', type=str, required=True, choices=["onnx"], default=segmentation_type, help="Choose segmentation type:\nOptions are:\n- onnx")
    args.add_argument('--segmentation_model_path', type=str, default=segmentation_model_path, help="Path to the ONNX model file.\n(optional to overwrite default)")
    args.add_argument('--image_path', type=str, default=image_path, help="Path to the input image.\nDisables live video feed for testing")
    args.add_argument('--live', action='store_true', help="setting to true uses the life")
    args.add_argument('--verbose', type=bool, default=live, help="enable status/debug output (always on for now)")
    args = args.parse_args()

    print("starting COMMITMENT ISSUES pipeline")

    frame_counter = 0
    start_time = time.time()
    while True:
        # OBTAINING IMAGE
        if(live):
            image = fetch_image()
        elif(args.image_path):
            print(args.image_path)
            image = cv2.imread(args.image_path)
        else:
            image = cv2.imread('/home/maja/Projects/ZEISS/6.jpeg')
        
        # SEGMENTATION
        if(args.segmentation_type=='onnx'):
            if(segmentation_model_path):
                model(image=image, model_path=segmentation_model_path)
            else:
                model(image=image)

        # POSITIONING (TODO!!)

        # VERBOSE
        if(not args.live):
            end_time = time.time()
            print(f"Execution took {end_time - start_time:.2f} seconds")
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            break
        if(not verbose): continue
        frame_counter += 1
        if(frame_counter == 5):
            print(f"FPS: {frame_counter/(time.time() - start_time):.2f}")
            start_time = time.time()
            frame_counter = 0
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
