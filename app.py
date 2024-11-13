from flask import Flask, Response, request
import cv2
import numpy as np
from retinaface import RetinaFace
import time

app = Flask(__name__)

# Route to handle incoming frames
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
