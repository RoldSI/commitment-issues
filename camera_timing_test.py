#! /usr/bin/env python3

import requests
import io
import time

# URL of the image source
url = 'http://169.254.66.91/camera/image'
headers = {'accept': 'image/jpeg'}

# Credentials for HTTP Basic Authentication
auth = ('admin', 'sdi')

def fetch_image():
    response = requests.get(url, headers=headers, auth=auth)
    response.raise_for_status()  # Raises an error on a bad status
    return io.BytesIO(response.content)  # Return bytes object instead of an Image object

try:
    last_time = time.time()
    frame_count = 0
    while True:
        fetch_image()  # Fetch the image
        current_time = time.time()
        fps = frame_count / (current_time - last_time)
        frame_count = frame_count+1
        print(f"FPS: {fps:.2f}")
        # print(f"Time between images: {current_time - last_time:.2f} seconds")
        # last_time = current_time

except KeyboardInterrupt:
    print("Stopped by user.")
except Exception as e:
    print(f"An error occurred: {e}")
