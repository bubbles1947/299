import cv2
import matplotlib.pyplot as plt

# Load and resize the image
imagePath = 'E:/FaceV/statpic/hasina.jpg'
img = cv2.imread(imagePath)

if img is None:
    print("Error: Could not read the image.")
else:
    img = cv2.resize(img, (2667, 4000))  # Resize to (width, height)


img = cv2.resize(img, (2667, 4000))  # Resize to (width, height)

# Convert image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the face detection classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Detect faces in the grayscale image
faces = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5,minSize=(40, 40)
)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Convert the image to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(5, 5))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
