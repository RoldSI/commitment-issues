#! /usr/bin/env python3

import argparse
import cv2
import onnxruntime as ort
import numpy as np


model_path = 'models/iris_semseg_upp_scse_mobilenetv2.onnx'
image_path = '/home/maja/Projects/ZEISS/6.jpeg'

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str, default=model_path, help='Path to the ONNX model file')
    args.add_argument('--image_path', type=str, default=image_path, help='Path to the input image')
    args = args.parse_args()

    image_pth = args.image_path
    
    ort_session = ort.InferenceSession(args.model_path)

    # Load the image
    image = cv2.imread(image_pth)
    # to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # to rgb  
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Preprocess the image (if needed)
    # resizee to 480x640
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

    for channel in range(output.shape[0]):
        mask = np.where(output[channel,:,:]>0.5, 1, 0)
        # Save mask as an image
        mask_image = np.uint8(mask * 255)
        #mask_image = cv2.resize(mask_image, (image_width, image_height))
        cv2.imwrite('mask_{}.png'.format(channel), mask_image)

    # Process the output (if needed)
    # ...

    # Display the result (if needed)
    # ...

if __name__ == '__main__':
    main()
