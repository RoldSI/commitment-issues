import argparse
import cv2
import time
import numpy as np
from pynput import keyboard

# Constants for eye movement
EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR = 246.77
EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER = 213.1

# Global variables for tracking gaze
setup = False
ground_truth_position = np.array([0, 0])
current_position = np.array([0, 0])
quit_program = False

# Key press handler for calibration and quitting
def on_press(key):
    global setup, ground_truth_position, current_position, quit_program
    try:
        if key.char == 'e':
            if not setup:
                setup = True
                ground_truth_position = current_position.copy()
                print("Calibration successful!")
        elif key == keyboard.Key.esc:
            quit_program = True
    except AttributeError:
        pass

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--video_path', type=str, default='filename.avi', help="Path to the video file")
    args = args.parse_args()

    print("Starting gaze tracking pipeline")

    cv2.namedWindow('raw-image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('masked-image', cv2.WINDOW_NORMAL)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_counter = 0
    start_time = time.time()

    video_cap = cv2.VideoCapture(args.video_path)

    while video_cap.isOpened():
        if quit_program:
            break

        ret, frame = video_cap.read()
        if not ret:
            break
        
        # Simulated current pupil position for demonstration purposes
        current_position = np.array([int(frame.shape[1] / 2), int(frame.shape[0] / 2)])

        # Display raw image
        cv2.imshow('raw-image', frame)

        # UI Elements for calibration and visualization
        if setup:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 0)  # Green
            thickness = 2
            
            # Displacement vector
            displacement_vector = current_position - ground_truth_position
            
            # Draw circle for current position
            cv2.circle(frame, tuple(current_position), 10, (0, 255, 0), -1)
            
            # Draw arrow indicating gaze shift
            cv2.arrowedLine(
                frame,
                tuple(ground_truth_position),
                tuple(current_position),
                (0, 0, 255),  # Red
                2,
                tipLength=0.3
            )
            
            # Calculate horizontal and vertical angles
            hor_angle = np.rad2deg(np.arctan(displacement_vector[0] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR))
            ver_angle = np.rad2deg(np.arctan(displacement_vector[1] / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER))
            
            # Display text information about angles
            cv2.putText(
                frame,
                f"Hor Angle: {hor_angle:.2f} degrees",
                (50, 50),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Ver Angle: {ver_angle:.2f} degrees",
                (50, 100),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        else:
            # Calibration not yet done, only mark current position
            cv2.circle(frame, tuple(current_position), 10, (0, 255, 255), -1)  # Yellow

        # Display the processed frame with UI elements
        cv2.imshow('masked-image', frame)

        if frame_counter % 10 == 0:
            print(f"FPS: {frame_counter / (time.time() - start_time):.2f}")
        
        frame_counter += 1
        if cv2.waitKey(100) == 27:  # Exit on 'ESC'
            break

    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
