import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/maja/Downloads/iris_landmark.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess input image
input_image = cv2.imread("62.jpeg")
image_height, image_width,_ = input_image.shape
img = input_image.copy()

#convert from bgr to rgb
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
x_scale, y_scale = image_width/64, image_height/64

#resize image
input_image = cv2.resize(input_image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
# input_image = input_image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
#input_image = np.array(input_image, dtype=np.float32)
input_image = np.expand_dims(input_image.astype(np.float32) /127.5 -1.0, axis=0)  # Add batch dimension if the model expects it

# Normalize input image if necessary


# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

landmarks = interpreter.get_tensor(output_details[1]['index'])[0]

for i in range(len(landmarks)//3):
    x, y, z = int(landmarks[i]/64.0*image_width), int(landmarks[i+1]/64.0*image_height), int(landmarks[i+2])
    cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

for i in range(len(output_data)//3):
    x, y, z = int(output_data[i]*x_scale), int(output_data[i+1]*y_scale), int(output_data[i+2])
    cv2.drawMarker(img, (x, y), (255,0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

# Display the result
cv2.imshow("Landmarks Detected", img)
cv2.waitKey(0)  
cv2.destroyAllWindows()

# Postprocess output if necessary
# For example, convert output to labels or do further processing

print("Output:", output_data)
