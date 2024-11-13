import cv2
import time
import os
from retinaface import RetinaFace
import numpy as np
from insightface.app import FaceAnalysis
import pickle

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
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=1, det_size=(640, 640))

# Load all embeddings from the directory
def load_embeddings(embedding_dir="embeddings"):
    embeddings = {}
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    for filename in os.listdir(embedding_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(embedding_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    embedding = pickle.load(f)
                    embeddings[filename] = embedding
            except Exception as e:
                print(f"Error loading embedding from {filename}: {e}")
    return embeddings

# Load embeddings
embeddings = load_embeddings()

# Main loop for real-time face recognition
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally to correct mirroring
    frame = cv2.flip(frame, 1)

    # Detect faces using RetinaFace on the full frame
    faces = RetinaFace.detect_faces(frame)

    # Variable to store the match status text
    match_status = None

    # Loop through detected faces
    if isinstance(faces, dict):
        for face_id, face_info in faces.items():
            # Verify that 'facial_area' exists in face_info
            if 'facial_area' in face_info:
                facial_area = face_info['facial_area']
                x1, y1, x2, y2 = facial_area

                # Draw rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Perform recognition directly on the full frame
                detected_faces = face_app.get(frame)
                if len(detected_faces) > 0:
                    face_feature = detected_faces[0].embedding
                    print("Detected face in full frame for comparison.")

                    # Variable to track the best match
                    best_match_score = -1
                    best_match_name = None

                    # Compare with all stored embeddings
                    for name, stored_embedding in embeddings.items():
                        # Compute similarity score
                        similarity = np.dot(stored_embedding, face_feature) / (np.linalg.norm(stored_embedding) * np.linalg.norm(face_feature))
                        print(f"Similarity Score with {name}: {similarity}")

                        # Update the best match
                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_match_name = name

                    # Display match or no match based on similarity threshold
                    if best_match_score > 0.4:  # Adjust threshold as needed
                        match_status = f'Match with {best_match_name}'
                        cv2.putText(frame, match_status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        match_status = 'No Match'
                        cv2.putText(frame, match_status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    print("No face detected in the full frame; skipping similarity calculation.")

    # Display the resulting frame
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
