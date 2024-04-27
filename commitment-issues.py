#! /usr/bin/env python3

import argparse
from image_api import fetch_image
from eye_segmentation_onnx import model
from segmentation_classical import find_pupil
import cv2
import time
import sys
import numpy as np
from pynput import keyboard

def main():
    segmentation_type = None
    segmentation_model_path = None
    image_path = None
    live = False
    verbose = True
    global setup, ground_truth_position, current_position
    setup = False
    ground_truth_position = [0, 0]
    current_position = [0, 0]

    args = argparse.ArgumentParser()
    args.add_argument('--segmentation_type', type=str, required=True, choices=["onnx", "classic"], default=segmentation_type, help="Choose segmentation type:\nOptions are:\n- onnx")
    args.add_argument('--segmentation_model_path', type=str, default=segmentation_model_path, help="Path to the ONNX model file.\n(optional to overwrite default)")
    args.add_argument('--image_path', type=str, default=image_path, help="Path to the input image.\nDisables live video feed for testing")
    args.add_argument('--live', action='store_true', help="setting to true uses the life")
    args.add_argument('--verbose', type=bool, default=live, help="enable status/debug output (always on for now)")
    args = args.parse_args()

    print("starting COMMITMENT ISSUES pipeline")
    cv2.namedWindow('raw-image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('masked-image', cv2.WINDOW_NORMAL)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_counter = 0
    start_time = time.time()
    while True:
        # OBTAINING IMAGE
        print("OBTAINING IMAGE")
        if(args.live):
            image = fetch_image()
        elif(args.image_path):
            print(args.image_path)
            image = cv2.imread(args.image_path)
        else:
            image = cv2.imread('./test-images/6.jpeg')
        cv2.imshow('raw-image', image)
        print("OBTAINING IMAGE FINISHED")
        
        # SEGMENTATION
        print("SEGMENTATION")
        if(args.segmentation_type=='classic'):
            [centroid, iris_mask] = find_pupil(image)
            current_position = centroid.flatten().astype(int)
            print(f"current_position: {current_position}")
            iris_color_mask = cv2.cvtColor(iris_mask, cv2.COLOR_GRAY2BGR)
        elif(args.segmentation_type=='onnx'):
            if(segmentation_model_path):
                segmented_masks = model(image=image, model_path=segmentation_model_path)
            else:
                segmented_masks = model(image=image)
            iris_mask = segmented_masks[2]
            _, iris_mask = cv2.threshold(iris_mask, 127, 255, cv2.THRESH_BINARY)
            iris_color_mask = cv2.cvtColor(iris_mask, cv2.COLOR_GRAY2BGR)
            white_pixel_coords = cv2.findNonZero(iris_mask)
        else:
            print("FATAL ARGUMENT ERROR!")
            exit(1)
        print("SEGMENTATION FINISHED")

        # IRIS POSITIONING
        print("IRIS POSITIONING")
        if not args.segmentation_type=='classic':
            if white_pixel_coords is None:
                print("NO DETECTION! REPOSITION PLEASE!")
                continue
            else:
                centroid = np.mean(white_pixel_coords, axis=0)
                current_position = centroid.flatten().astype(int)
        print("IRIS POSITIONING FINISHED")

        # POSITIONING (TODO!!)
        print("POSITIONING")
        if setup:
            cv2.circle(iris_color_mask, (ground_truth_position[0], ground_truth_position[1]), 5, (0, 255, 0), -1)
            cv2.arrowedLine(iris_color_mask, tuple(ground_truth_position), tuple(current_position), (0, 0, 255), 2)
        else:
            cv2.circle(iris_color_mask, (current_position[0], current_position[1]), 5, (0, 255, 255), -1)
        cv2.imshow('masked-image', iris_color_mask)
        print("POSITIONING FINISHED")

        # # TRACKING OUTPUT
        if(setup):
            print(f"current offset is: {ground_truth_position - current_position}")

        # VERBOSE
        if(not args.live):
            end_time = time.time()
            print(f"Execution took {end_time - start_time:.2f} seconds")
            print(f"current position: {current_position}")
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            break
        cv2.waitKey(1)
        if(not verbose): continue
        frame_counter += 1
        if(True):
            print(f"FPS: {frame_counter/(time.time() - start_time):.2f}")
            start_time = time.time()
            frame_counter = 0
    cv2.destroyAllWindows()

def on_press(key):
    # Function to handle key press
    try:
        if key.char == 'e':  # Check if 'a' is pressed
            global setup
            if not setup:
                setup = True
                global ground_truth_position, current_position
                ground_truth_position = current_position
                print("Calibration successful!!")
    except AttributeError:
        pass

if __name__ == '__main__':
    main()
