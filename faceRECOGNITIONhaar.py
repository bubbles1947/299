import cv2
import time
from insightface.app import FaceAnalysis
import numpy as np

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
    ref_image_path = "C:/Users/USER/Desktop/FaceV/sample/sample1.jpg"  # Replace with your image path
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

# Initialize Haar Cascade for face detection (for better performance)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Main loop for real-time face recognition
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally to correct mirroring
    frame = cv2.flip(frame, 1)  # 1 means flipping around y-axis

    # Convert to grayscale for Haar Cascade detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    # Variable to store the match status text
    match_status = None

    # Loop through detected faces
    for (x1, y1, w, h) in faces:
        x2, y2 = x1 + w, y1 + h

        # Draw rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Perform recognition if reference feature is loaded
        if reference_feature is not None:
            face_crop = frame[y1:y2, x1:x2]
            detected_faces = face_app.get(face_crop)

            if len(detected_faces) > 0:
                face_feature = detected_faces[0].embedding  # Assume first detected face is the target

                # Compute cosine similarity (match)
                similarity = np.dot(reference_feature, face_feature) / (np.linalg.norm(reference_feature) * np.linalg.norm(face_feature))
                
                if similarity > 0.3:  # Lower threshold for matching
                    match_status = 'Match'
                    cv2.putText(frame, match_status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    match_status = 'No Match'
                    cv2.putText(frame, match_status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Optionally, draw facial landmarks (eyes, nose, mouth)
        if isinstance(faces, dict):  # RetinaFace landmarks (if you used RetinaFace)
            landmarks = faces.get('landmarks', {})
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
