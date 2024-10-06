import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to the model and configuration file
modelFile = "E:/FaceV/res10_300x300_ssd_iter_140000_fp16.caffemodel"  # Replace with correct path
configFile = "E:/FaceV/deploy.prototxt"  # Replace with correct path

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the image
imagePath = 'E:/FaceV/statpic/g2.jpg'  # Replace with your image path
img = cv2.imread(imagePath)

if img is None:
    print("Error: Could not read the image.")
else:
    (h, w) = img.shape[:2]

    # Prepare the image for the DNN model
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform forward pass to get the face detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 7)
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()