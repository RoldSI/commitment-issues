import cv2 as cv
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Define indices for the left and right eyes
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Define indices for the left and right iris
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Initialize video capture from the webcam
cap = cv.VideoCapture(8)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for a more natural view and convert to RGB for MediaPipe processing
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Extract mesh points for landmarks
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])

            # Get iris centers and radii for left and right eyes
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            # Calculate the relative position to determine which eye is in view
            if l_cx < r_cx:
                # Left eye is more to the left
                iris_center = np.array([l_cx, l_cy], dtype=np.int32)
                color = (0, 255, 0)  # Green for left eye
                radius = int(l_radius)
            else:
                # Right eye is more to the right
                iris_center = np.array([r_cx, r_cy], dtype=np.int32)
                color = (255, 0, 0)  # Red for right eye
                radius = int(r_radius)

            # Draw the iris circle with the appropriate color
            cv.circle(frame, iris_center, radius, color, 2, cv.LINE_AA)

        # Increment frame count for FPS calculation
        frame_count += 1

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display the FPS on the frame
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame with eye tracking and FPS
        cv.imshow('Eye Tracking', frame)

        # Exit on pressing the 'Escape' key
        if cv.waitKey(1) & 0xFF == 27:  # Escape key to stop
            break

cap.release()
cv.destroyAllWindows()
