import os
import glob
import cv2
import numpy as np
from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

# Initialize the 2D and 3D detectors
detector_2d = Detector2D()
camera = CameraModel(focal_length=561.5, resolution=[400, 400])
detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)

# Function to preprocess the image
def preprocess_image(image):
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred

# Function to detect the pupil and return its position
def detect_pupil(image):
    preprocessed = preprocess_image(image)

    thresholded = cv2.adaptiveThreshold(
        preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area and len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            if axes[0] / axes[1] < 1.2:
                largest_area = area
                largest_contour = contour

    if largest_contour is not None:
        # Ensure correct structure with proper indices and default confidence
        ellipse = cv2.fitEllipse(largest_contour)
        return {
            "ellipse": {
                "center": (ellipse[0][0], ellipse[0][1]),
                "axes": (ellipse[1][0], ellipse[1][1]),
                "angle": ellipse[2],
            },
            "confidence": 1.0,  # Default confidence
        }

    return None

# Function to estimate gaze direction with pye3d
def estimate_gaze_with_pye3d(image, frame_number, fps):
    result_2d = detect_pupil(image)

    if result_2d is None:
        print("Failed to detect pupil. Skipping gaze estimation.")
        return None

    result_2d["timestamp"] = frame_number / fps
    if "confidence" not in result_2d:
        result_2d["confidence"] = 1.0  # Default confidence

    try:
        result_3d = detector_3d.update_and_detect(result_2d, preprocess_image(image))
    except Exception as e:
        print(f"Error in 3D detection: {e}")
        return None

    if "gaze_direction" not in result_3d:
        print("3D detection failed. 'gaze_direction' not found.")
        return None

    return result_3d["gaze_direction"]

# Process all PNG images from the 'Pictures' folder
def process_images(image_files):
    frame_number = 0
    fps = 30  # Default FPS for timing calculations

    for image_file in image_files:
        eye_image = cv2.imread(image_file)

        if eye_image is None:
            print(f"Could not read image: {image_file}")
            continue

        # Estimate gaze direction with pye3d
        gaze_vector = estimate_gaze_with_pye3d(eye_image, frame_number, fps)

        if gaze_vector:
            print(f"Gaze vector for {image_file}: {gaze_vector}")

            # Show the image with gaze direction indication
            cv2.imshow(f"Gaze Direction for {image_file}", eye_image)

            cv2.waitKey(500)  # Display for 500 milliseconds

        frame_number += 1

        # Clean up OpenCV windows
        cv2.destroyAllWindows()

# Get all PNG images from the 'Pictures' folder
image_files = glob.glob("Pictures/*.png")

if not image_files:
    print("No images found in the 'Pictures' folder.")
else:
    # Process images with pye3d
    process_images(image_files)
