from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import os
import subprocess
from retinaface import RetinaFace

app = Flask(__name__)

# Directory for saving uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to process incoming frames
@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Receive and decode the frame from the request
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(frame)
    if isinstance(faces, dict):
        for face_id, face_info in faces.items():
            facial_area = face_info['facial_area']
            x1, y1, x2, y2 = facial_area
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Draw facial landmarks
            landmarks = face_info['landmarks']
            for _, point in landmarks.items():
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

    # Encode frame back to JPEG and return as a response
    _, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


# Route to upload an image and process it with testGUI3.py
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    username = request.form.get('username')
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not username:
        return jsonify({"error": "Username not provided"}), 400

    # Save the file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run testGUI3.py with the image and username
    result = subprocess.run(
        ["python", "testGUI3.py", filepath, username],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    return jsonify({"message": "Image processed successfully", "username": username})


# Route to start face recognition using faceRecUpWtime.py
@app.route('/start_face_recognition', methods=['POST'])
def start_face_recognition():
    result = subprocess.run(
        ["python", "faceRecUpWtime.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    return jsonify({"message": "Face recognition started"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
