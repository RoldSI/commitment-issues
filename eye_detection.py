#! /usr/bin/env python3

import cv2
import numpy as np

def estimate_gaze_direction(eye_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect ellipses using Hough transform
    ellipses = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=5, maxRadius=50)
    
    if ellipses is not None:
        ellipses = np.round(ellipses[0, :]).astype("int")
        for (x, y, r) in ellipses:
            # Draw the ellipse on the eye image
            cv2.circle(eye_image, (x, y), r, (0, 255, 0), 4)
            cv2.circle(eye_image, (x, y), 2, (0, 0, 255), 3)
            
            # Calculate gaze direction vector
            gaze_direction = (x - eye_image.shape[1] // 2, y - eye_image.shape[0] // 2)
            cv2.arrowedLine(eye_image, (eye_image.shape[1] // 2, eye_image.shape[0] // 2), 
                            (x, y), (255, 0, 0), 2)
            
            return gaze_direction
    
    return None

# Load the image of the eye
eye_image = cv2.imread("20240427_150511.jpg")

# Estimate gaze direction
gaze_direction = estimate_gaze_direction(eye_image)

# Display the result
if gaze_direction is not None:
    print("Gaze direction:", gaze_direction)
    cv2.imshow("Eye Image with Gaze Direction", eye_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No ellipse detected. Gaze direction cannot be estimated.")
