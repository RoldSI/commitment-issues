#! /usr/bin/env python3

import requests
import io
import numpy as np
import cv2

# URL of the image source
url = 'http://169.254.66.91/camera/image'
headers = {'accept': 'image/jpeg'}

# Credentials for HTTP Basic Authentication
auth = ('admin', 'sdi')

def fetch_image():
    response = requests.get(url, headers=headers, auth=auth)
    response.raise_for_status()  # Raises an error on a bad status
    image_bytes = response.content  # Get the image content as bytes
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)  # Convert bytes to numpy array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Decode the image using OpenCV
    return image
