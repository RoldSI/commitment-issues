#! /usr/bin/env python3

# Commented out IPython magic to ensure Python compatibility.
import cv2
from matplotlib import pyplot as plt
import numpy as np

def show_image(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np_array = np.array(image)
    plt.imshow(np_array)
    plt.show()


def find_pupil(image) -> np.ndarray[2,]:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # avg_val = np.average(img)
    # img[np.where(img>avg_val*.5)] = 255
    img[np.where(img>np.percentile(img, 5))] = 255
    img[np.where(img<0.3)] = 0

    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    _, _, stats, centroids = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=4)
    i_max = np.argsort(stats[:,cv2.CC_STAT_AREA])[-2]
    pupil_point = centroids[i_max,:]
    return [pupil_point, closed]

# if __name__ == "__main__":
#     img_path = "1.jpeg"
#     pupil_point = find_pupil(img_path)

#     image = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
#     plt.imshow(np.array(image))
#     plt.scatter(pupil_point[0], pupil_point[1])
#     plt.show()