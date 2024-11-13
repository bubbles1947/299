import cv2
import os
from insightface.app import FaceAnalysis
import pickle

# Initialize the FaceAnalysis model
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=1, det_size=(640, 640))

# Ensure the embeddings directory exists
embeddings_dir = "embeddings"
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

# Ask the user for an image file to process
image_path = input("Please enter the path of the image to generate embeddings: ")
embedding_filename = input("Enter a unique name for the embedding (e.g., face1.pkl): ").strip()

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Detect faces in the image
faces = face_app.get(image)

# If faces are detected, process each face
if len(faces) > 0:
    for i, face in enumerate(faces):
        # Extract the face embedding
        face_embedding = face.embedding

        # Generate a filename based on the image name and face index
      #  image_name = os.path.basename(image_path)
       # embedding_filename = f"{os.path.splitext(image_name)[0]}_face_{i+1}.pkl"
        #embedding_path = os.path.join(embeddings_dir, embedding_filename)

        # Save the embedding to a file
        with open(embedding_filename, 'wb') as f:
            pickle.dump(face_embedding, f)

        print(f"Generated and saved embedding to {embedding_filename}")
else:
    print("No faces detected in the image.")
