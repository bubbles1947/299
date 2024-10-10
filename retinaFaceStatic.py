import cv2
from retinaface import RetinaFace

# Load the image where faces need to be detected
image_path = 'E:/FaceV/statpic/o7.jpg'
image = cv2.imread(image_path)

# Perform face detection using RetinaFace
faces = RetinaFace.detect_faces(image)

# Check if faces were detected
if isinstance(faces, dict):
    for face_id, face_info in faces.items():
        # Extract facial area (bounding box)
        facial_area = face_info['facial_area']
        x1, y1, x2, y2 = facial_area

        # Draw rectangle around the face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optionally, draw facial landmarks (eyes, nose, mouth)
      #  landmarks = face_info['landmarks']
       # for key, point in landmarks.items():
        #    cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

# Show the image with detected faces
cv2.imshow('Detected Faces', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
