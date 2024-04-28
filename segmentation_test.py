import cv2
import numpy as np
from matplotlib import pyplot as plt



def show_image(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np_array = np.array(image)
    plt.imshow(np_array)
    plt.show()

def brighten_image(image, brightness_factor):
    # Convert image to float32 for better accuracy in calculations
    image_float = image.astype(np.float32)
    
    # Add the brightness factor to each pixel value
    brightened_image = image_float + brightness_factor
    
    # Clip pixel values to the valid range [0, 255]
    brightened_image = np.clip(brightened_image, 0, 255)
    
    # Convert the image back to uint8 format
    brightened_image = brightened_image.astype(np.uint8)
    
    return brightened_image

def add_threshold(img, value):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray_img, value, 255, cv2.THRESH_BINARY_INV)
    return im_bw

def add_floodfill(img):
    # Copy image for floodFill
    im_floodfill = img.copy()
    print('shape', im_floodfill.shape)
    
    # Mask for floodFill (h+2, w+2 to allow filling at edges)
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Seed point for floodFill
    seed_point = (0, 0)
    
    # Fill the background with 255 (white)
    cv2.floodFill(im_floodfill, mask, seed_point, 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine original image and inverted floodfilled image
    im_out = cv2.bitwise_or(im_bw, im_floodfill_inv)

# cap = cv2.VideoCapture("videos/filename.avi")

# while True:
def testo(frame):
    bright_image = brighten_image(frame, 100)
    bw_img = add_threshold(bright_image, 120)

    image = np.copy(bright_image)

    # Find contours:
    contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw contours:
    # new_image = cv2.drawContours(image, contours, 0, (0, 255, 0), 2)

    # Calculate image moments of the detected contour
    # M = cv2.moments(contours[0])

    # Print center (debugging):
    # print("center X : '{}'".format(round(M['m10'] / M['m00'])))
    # print("center Y : '{}'".format(round(M['m01'] / M['m00'])))

    # Find the contour with the maximum area
    if len(contours) != 0:
        max_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(max_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        return [image, [centroid_x, centroid_y]]
    else:
        return [image, [0, 0]]

    # # Draw the centroid on the original image
    # image = cv2.circle(image, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

    # # Calculate the radius of the circle
    # # radius = int(cv2.arcLength(max_contour, True) / (4 * np.pi))
    # radius = 130
    # print(radius)

    # cv2.circle(image, (centroid_x, centroid_y), radius, (0, 0, 255), 2)

    # cv2.imshow("frame", image)
    # cv2.waitKey(3)
    # input("enter")

# cv2.destroyAllWindows()
# cap.release()