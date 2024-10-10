import cv2
import time
from retinaface import RetinaFace

# Initialize the video capture (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

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

            # Optionally, draw facial landmarks (eyes, nose, mouth)
          #  landmarks = face_info['landmarks']
           # for key, point in landmarks.items():
            #    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)


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
    cv2.imshow('Real-Time Face Detection', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

