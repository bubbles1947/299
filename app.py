import cv2
import pickle
import mysql.connector
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)

# Directory for saving uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Connect to the MySQL database
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="persons"
        )
        if connection.is_connected():
            print("Connected to MySQL database!")
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Upload embedding to the database
def upload_embedding_to_db(username, embedding_data, connection):
    try:
        cursor = connection.cursor()
        insert_query = "INSERT INTO face_embeddings (username, embedding) VALUES (%s, %s)"
        cursor.execute(insert_query, (username, embedding_data))
        connection.commit()
        cursor.close()
        print("Embedding uploaded to database successfully!")
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")

# Process the image and create an embedding
def process_image_and_upload(file_path, username, connection):
    # Initialize FaceAnalysis model for embedding generation
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=1, det_size=(640, 640))
    
    # Load the selected image and generate embedding
    image = cv2.imread(file_path)
    faces = face_app.get(image)
    
    if faces:
        # Save the first detected face embedding
        embedding_data = pickle.dumps(faces[0].embedding)
        
        # Upload the embedding to the database
        upload_embedding_to_db(username, embedding_data, connection)
        return {"message": "Embedding generated and uploaded successfully!"}
    else:
        return {"error": "No face detected in the image."}

# API route to handle image upload and process it
@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Get the username and the image from the request
    username = request.form.get("username")
    file = request.files.get('file')
    
    if not username or not file:
        return jsonify({"error": "Username or file missing"}), 400
    
    # Save the uploaded file to a temporary location
    filepath = f"uploads/{file.filename}"
    file.save(filepath)

    # Connect to the database
    connection = connect_to_db()

    if connection:
        # Process the image and upload the embedding
        result = process_image_and_upload(filepath, username, connection)
        connection.close()
        return jsonify(result)
    else:
        return jsonify({"error": "Database connection failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Running the server on port 5000
