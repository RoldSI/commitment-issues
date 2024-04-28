#! /usr/bin/env python3

import argparse
from image_api import fetch_image
from eye_segmentation_onnx import model
from segmentation_classical import find_pupil
from segmentation_test import testo
import cv2
import time
import sys
import numpy as np
from pynput import keyboard

EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR = 246.77  # experimented
EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER = 213.1  # experimented

# Global variables
setup = False
ground_truth_position = [0, 0]
current_position = [0, 0]
quit_program = False  # To exit on "escape" key press

# Function to handle key press
def on_press(key):
    try:
        global setup, ground_truth_position, current_position, quit_program
        if key.char == 'e':  # Check if 'e' is pressed for calibration
            if not setup:
                setup = True
                ground_truth_position = current_position
                print("Calibration successful!!")
        elif key == keyboard.Key.esc:  # Exit on "escape"
            quit_program = True
    except AttributeError:
        pass

def main():
    segmentation_type = None
    segmentation_model_path = None
    image_path = None
    live = False
    verbose = True

    args = argparse.ArgumentParser()
    args.add_argument('--segmentation_type', type=str, required=True, choices=["onnx", "classic", "testo"], help="Choose segmentation type")
    args.add_argument('--segmentation_model_path', type=str, default=segmentation_model_path, help="Path to the ONNX model file (optional)")
    args.add_argument('--image_path', type=str, default=image_path, help="Path to the input image")
    args.add_argument('--live', action='store_true', help="Use live video feed")
    args.add_argument('--verbose', type=bool, default=live, help="Enable status/debug output")
    args = args.parse_args()

    print("Starting COMMITMENT ISSUES pipeline")
    cv2.namedWindow('raw-image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('masked-image', cv2.WINDOW_NORMAL)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_counter = 0
    start_time = time.time()
    detection_timeout = 10  # 10-second timeout for iris detection
    elapsed_time = 0  # Timer to track detection timeout

    while True:
        if quit_program:  # Exit loop on "escape"
            break

        # OBTAINING IMAGE
        print("OBTAINING IMAGE")
        if args.live:
            image = fetch_image()
        elif args.image_path:
            print(args.image_path)
            image = cv2.imread(args.image_path)
        else:
            image = cv2.imread('./test-images/6.jpeg')

        cv2.imshow('raw-image', image)
        print("OBTAINING IMAGE FINISHED")

        # SEGMENTATION
        print("SEGMENTATION")
        if args.segmentation_type == 'testo':
            [iris_mask, current_position] = testo(image)
            iris_color_mask = iris_mask
        elif args.segmentation_type == 'classic':
            [centroid, iris_mask] = find_pupil(image)
            current_position = centroid.flatten().astype(int)
            iris_color_mask = cv2.cvtColor(iris_mask, cv2.COLOR_GRAY2BGR)
        elif args.segmentation_type == 'onnx':
            if args.segmentation_model_path:
                segmented_masks = model(image=image, model_path=args.segmentation_model_path)
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
        if not args.segmentation_type == 'classic' and not args.segmentation_type == 'testo':
            if white_pixel_coords is None:
                elapsed_time += 1  # Incrementing elapsed time for each loop iteration
                if elapsed_time >= detection_timeout:
                    print("Iris detection timeout. Exiting...")
                    break
            else:
                centroid = np.mean(white_pixel_coords, axis=0)
                current_position = centroid.flatten().astype(int)
                elapsed_time = 0  # Reset elapsed time if iris is detected
        else:
            centroid = np.mean(white_pixel_coords, axis=0)
            current_position = centroid.flatten().astype(int)

        print("IRIS POSITIONING FINISHED")

        # POSITIONING (as it was)
        print("POSITIONING")
        if setup:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 255, 0)
            thickness = 8
            diff_position = current_position - ground_truth_position

            cv2.circle(iris_color_mask, (current_position[0], current_position[1]), 10, (0, 255, 0), -1)
            goal_position = current_position + diff_position
            cv2.arrowedLine(iris_color_mask, tuple(current_position), tuple(goal_position), (0, 0, 255), 2)

            hor_diff_angle = diff_position[0] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR
            hor_angle = np.arctan(hor_diff_angle)
            hor_degree = np.rad2deg(hor_angle)

            ver_diff_angle = diff_position[0] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER
            ver_angle = np.arctan(ver_diff_angle)
            ver_degree = np.rad2deg(ver_angle)

            image = cv2.putText(iris_color_mask, "hor gaze angle is " + str(hor_degree) + " and ver angle is " + str(ver_degree), org, font, fontScale, color, thickness, cv2.LINE_AA)
            print("difference:", diff_position)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.circle(iris_color_mask, (current_position[0], current_position[1]), 10, (0, 255, 255), -1)

            diff_position = ground_truth_position - current_position
            hor_diff_angle = diff_position[0] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR
            hor_angle = np.arctan(hor_diff_angle)
            hor_degree = np.rad2deg(hor_angle)

            ver_diff_angle = diff_position[0] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER
            ver_angle = np.arctan(ver_diff_angle)
            ver_degree is np.rad2deg(ver_angle)

            image = cv2.putText(iris_color_mask, "hor gaze angle is " + str(hor_degree) + " and ver angle is " + str(ver_degree), org, font, fontScale, color, thickness, cv2.LINE_AA)
            print("difference:", diff_position)
        
        cv2.imshow('masked-image', iris_color_mask)
        print("POSITIONING FINISHED")

        # TRACKING OUTPUT
        if setup:
            print(f"current offset is: {ground_truth_position - current_position}")

        # VERBOSE
        if not args.live:
            end_time = time.time()
            print(f"Execution took {end_time - start_time:.2f} seconds")
            print(f"current position: {current_position}")
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            break

        if not args.verbose:
            continue

        frame_counter += 1
        if True:
            print(f"FPS: {frame_counter / (time.time() - start_time):.2f}")
            start_time = time.time()
            frame_counter = 0
        
        cv2.waitKey(100)  # Pause to avoid high CPU usage

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
