import cv2
import time
from retinaface import RetinaFace
import numpy as np
from insightface import app
from insightface.app import FaceAnalysis

# Initialize the video capture (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize Face Recognition model using InsightFace's FaceAnalysis
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' if using a GPU
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Placeholder for the uploaded reference image feature vector
reference_feature = None

# Display menu options
print("Options:")
print("1. Upload Reference Image")
print("2. Start Real-Time Face Recognition")
choice = input("Enter your choice (1/2): ")

# Function to load and compute feature vector for reference image
def upload_reference_image():
    global reference_feature
    ref_image_path = "E:/FaceV/sample/sample1.jpg"
    ref_image = cv2.imread(ref_image_path)

    if ref_image is None:
        print("Error: Could not load image.")
        return False

    # Detect and extract feature vector for the reference image
    faces = face_app.get(ref_image)
    if len(faces) > 0:
        reference_feature = faces[0].embedding  # Assume first detected face is the target
        print("Reference image uploaded successfully.")
        return True
    else:
        print("Error: No face detected in the reference image.")
        return False

if choice == '1':
    if not upload_reference_image():
        print("Failed to upload reference image. Exiting.")
        exit()
elif choice != '2':
    print("Invalid choice. Exiting.")
    exit()

# Main loop for real-time face recognition
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally to correct mirroring
    frame = cv2.flip(frame, 1)  # 1 means flipping around y-axis

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(frame)

    # Loop through detected faces
    if isinstance(faces, dict):
        for face_id, face_info in faces.items():
            # Extract facial area (bounding box)
            facial_area = face_info['facial_area']
            x1, y1, x2, y2 = facial_area

            # Draw rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Perform recognition if reference feature is loaded
            if reference_feature is not None:
                face_crop = frame[y1:y2, x1:x2]
                detected_faces = face_app.get(face_crop)

                if len(detected_faces) > 0:
                    face_feature = detected_faces[0].embedding  # Assume first detected face is the target
                    # Compute similarity (cosine distance)
                    similarity = np.dot(reference_feature, face_feature) / (np.linalg.norm(reference_feature) * np.linalg.norm(face_feature))
                    if similarity > 0.4:  # Adjust threshold as needed
                        #cv2.putText(frame, 'Match', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, 'Match', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                     #   cv2.putText(frame, 'No Match', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.putText(frame, 'No Match', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Optionally, draw facial landmarks (eyes, nose, mouth)
            landmarks = face_info['landmarks']
            for key, point in landmarks.items():
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:  # Update FPS every second
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
