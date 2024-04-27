import os
import glob
import cv2
import numpy as np

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred

# Function to detect the pupil and return its position
def detect_pupil(image):
    # Preprocess the image
    preprocessed = preprocess_image(image)

    # Use adaptive thresholding to isolate dark regions (pupil)
    thresholded = cv2.adaptiveThreshold(
        preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest valid contour (representing the pupil)
    largest_contour = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area and len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Ensure the contour is sufficiently circular
            if axes[0] / axes[1] < 1.2:
                largest_area = area
                largest_contour = contour

    # Explicitly check if the largest contour is valid
    if largest_contour is not None:
        return cv2.fitEllipse(largest_contour)
    
    return None

# Function to estimate gaze direction
def estimate_gaze_direction(eye_image):
    # Find the pupil
    pupil = detect_pupil(eye_image)

    if pupil is not None:
        # Draw the ellipse on the eye image
        cv2.ellipse(eye_image, pupil, (0, 255, 0), 2)

        # Calculate the center of the pupil
        pupil_center = (int(pupil[0][0]), int(pupil[0][1]))

        # Define the center of the eye image
        eye_center = (eye_image.shape[1] // 2, eye_image.shape[0] // 2)

        # Calculate gaze direction relative to the eye's center
        gaze_direction = (pupil_center[0] - eye_center[0], eye_center[1] - pupil_center[1])

        # Draw an arrowed line to indicate gaze direction
        cv2.arrowedLine(eye_image, eye_center, pupil_center, (255, 0, 0), 2)

        return gaze_direction
    
    return None

# Get all PNG images from the 'Pictures' folder in the current directory
image_files = glob.glob("Pictures/*.png")

# Process each image and estimate gaze direction
for image_file in image_files:
    # Load the image
    eye_image = cv2.imread(image_file)

    # Estimate gaze direction
    gaze_direction = estimate_gaze_direction(eye_image)

    if gaze_direction is not None:
        print(f"Gaze direction for {image_file}: {gaze_direction}")

        # Show the image with gaze direction indication
        cv2.imshow(f"Gaze Direction for {image_file}", eye_image)

        # Wait for a specific time to avoid blocking
        cv2.waitKey(500)  # Display for 500 milliseconds
        cv2.destroyAllWindows()
    else:
        print(f"No pupil detected in {image_file}. Gaze direction cannot be estimated.")
