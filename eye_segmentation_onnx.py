#! /usr/bin/env python3

import argparse
import cv2
import onnxruntime as ort
import numpy as np
import sys


# model_path = 'models/iris_semseg_upp_scse_mobilenetv2.onnx'
# image_path = '/home/maja/Projects/ZEISS/6.jpeg'

def model(image, model_path='./models/iris_semseg_upp_scse_mobilenetv2.onnx'):    
    ort_session = ort.InferenceSession(model_path)

    # to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # to rgb  
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Preprocess the image: resize to 480x640
    image = cv2.resize(image, (640, 480))

    # normalize and convert to float
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    # convert to tensor 
    # image = torch.tensor(image).permute(2, 0, 1)
    # # add batch dimension
    # image = image.unsqueeze(0)

    # Perform inference with the model
    output = ort_session.run(None, {'input':image})
    output = output[0][0]

    mask = output[0,:,:]

    masks = []
    for channel in range(output.shape[0]):
        mask = np.where(output[channel,:,:] > 0.5, 1, 0)
        masks.append(mask)

    # Convert masks to OpenCV images (numpy arrays)
    masks_cv2 = [np.uint8(mask * 255) for mask in masks]

    return masks_cv2

    # for channel in range(output.shape[0]):
    #     mask = np.where(output[channel,:,:]>0.5, 1, 0)
    #     # Save mask as an image
    #     mask_image = np.uint8(mask * 255)
    #     #mask_image = cv2.resize(mask_image, (image_width, image_height))
    #     cv2.imwrite('mask_{}.png'.format(channel), mask_image)

if __name__ == '__main__':
    print("THIS IS NOT SUPPOSED TO BE AN ENTRYPOINT!")
    sys.exit(1)
