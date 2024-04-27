import cv2 as cv
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Define indices for the left eye and iris
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [474, 475, 476, 477]

# Initialize video capture from the webcam
cap = cv.VideoCapture(8)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

# Use MediaPipe Face Mesh to track face landmarks
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

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # Process the frame with Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                for p in results.multi_face_landmarks[0].landmark
            ])

            # Draw the left iris circle
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)

            # Draw a circle around the left iris
            cv.circle(frame, center_left, int(l_radius), (0, 255, 0), 2, cv.LINE_AA)

            # Compute the gaze direction vector (from the iris center to an edge of the left eye)
            left_eye_corner = mesh_points[LEFT_EYE[0]]
            gaze_vector = left_eye_corner - center_left
            print("Gaze Vector:", gaze_vector)

        # Increment frame count for FPS calculation
        frame_count += 1

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display the FPS on the frame
        cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame with iris tracking, gaze vector, and FPS
        cv.imshow('Iris Tracking with Gaze Direction and FPS', frame)

        # Exit on Escape key
        if cv.waitKey(1) & 0xFF == 27:  # Escape key to stop
            break

cap.release()
cv.destroyAllWindows()
