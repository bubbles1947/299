from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os

app = Flask(__name__)

# Directory for saving uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Face Recognition Web App Backend is Running"

# API to upload an image and run `testGUI3.py`
@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    username = request.form.get("username")
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if username is None:
        return jsonify({"error": "Username not provided"}), 400

    # Save the file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run testGUI3.py
    result = subprocess.run(
        ["python", "testGUI3.py", filepath, username],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    return jsonify({"message": "Image processed successfully", "username": username})

# API to run face recognition
@app.route("/start_face_recognition", methods=["POST"])
def start_face_recognition():
    result = subprocess.run(
        ["python", "faceRecUpWtime.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    return jsonify({"message": "Face recognition started"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500)
