#! /usr/bin/env python3

import argparse
import cv2
import numpy as np
import time
from pynput import keyboard
from image_api import fetch_image
from eye_segmentation_onnx import model
from segmentation_classical import find_pupil
from segmentation_test import testo

# Constants for eye ball radius
EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR = 246.77  # experimented
EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER = 213.1  # experimented

# Global variables
setup = False
ground_truth_position = [0, 0]
current_position = [0, 0]

# Key press handler
def on_press(key):
    try:
        if key.char == 'e':  # Check if 'e' is pressed for setup
            global setup
            if not setup:
                setup = True
                global ground_truth_position, current_position
                ground_truth_position = current_position.copy()
                print("Calibration successful!!")
    except AttributeError:
        pass


def main():
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument('--segmentation_type', type=str, required=True, choices=["onnx", "classic", "testo"], help="Choose segmentation type")
    args.add_argument('--segmentation_model_path', type=str, help="Path to the ONNX model file")
    args.add_argument('--image_path', type=str, help="Path to the input image")
    args.add_argument('--live', action='store_true', help="Use live video feed")
    args.add_argument('--verbose', type=bool, default=True, help="Enable debug output")
    args = args.parse_args()

    # Setup OpenCV windows
    cv2.namedWindow('raw-image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('masked-image', cv2.WINDOW_NORMAL)

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_counter = 0
    start_time = time.time()

    # Main processing loop
    while True:
        # OBTAINING IMAGE
        print("OBTAINING IMAGE")
        if args.live:
            image = fetch_image()
        elif args.image_path:
            image = cv2.imread(args.image_path)
        else:
            image = cv2.imread('./test-images/6.jpeg')
        
        cv2.imshow('raw-image', image)

        # SEGMENTATION
        print("SEGMENTATION")
        if args.segmentation_type == 'testo':
            iris_mask, current_position = testo(image)
            iris_color_mask = image
        elif args.segmentation_type == 'classic':
            centroid, iris_mask = find_pupil(image)
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
        else:
            print("Invalid segmentation type!")
            exit(1)

        # IRIS POSITIONING
        white_pixel_coords = cv2.findNonZero(iris_mask)
        if not args.segmentation_type == 'classic' and not args.segmentation_type == 'testo':
            if white_pixel_coords is None:
                print("NO DETECTION! REPOSITION PLEASE!")
                continue
            else:
                centroid = np.mean(white_pixel_coords, axis=0)
                current_position = centroid.flatten().astype(int)

        # POSITIONING
        if setup:
            # Calculate goal position and arrow direction
            diff_position = np.array(ground_truth_position) - np.array(current_position)
            goal_position = np.array(current_position) + diff_position

            # Draw the arrow and other information
            cv2.circle(iris_color_mask, (current_position[0], current_position[1]), 10, (0, 255, 0), -1)
            cv2.arrowedLine(iris_color_mask, tuple(current_position), tuple(goal_position), (0, 0, 255), 2)

            # Calculate angles
            hor_diff_angle = diff_position[0] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR
            hor_angle = np.arctan(hor_diff_angle)
            hor_degree = np.rad2deg(hor_angle)

            ver_diff_angle = diff_position[1] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER
            ver_angle = np.arctan(ver_diff_angle)
            ver_degree = np.rad2deg(ver_angle)

            # Display gaze angle
            cv2.putText(iris_color_mask, f"Hor gaze angle: {hor_degree}°, Ver angle: {ver_degree}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.circle(iris_color_mask, (current_position[0], current_position[1]), 10, (0, 255, 255), -1)

        cv2.imshow('masked-image', iris_color_mask)

        if not args.live:
            print(f"Execution took {time.time() - start_time:.2f} seconds")
            break
        cv2.waitKey(1)

        frame_counter += 1
        if frame_counter >= 10:
            print(f"FPS: {frame_counter / (time.time() - start_time):.2f}")
            start_time = time.time()
            frame_counter = 0
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()