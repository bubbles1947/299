from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Serve the front-end (e.g., subscription.html)
@app.route('/')
def index():
    return render_template('subscription.html')

# Route to handle form submission and image upload
@app.route('/submit_image', methods=['POST'])
def submit_image():
    # Get the form data from the request
    username = request.form.get("username")
    file = request.files.get("file")

    if not username or not file:
        return jsonify({"error": "Username or file missing"}), 400

    # Prepare the data to send to the uploadImage.py server
    url = "http://127.0.0.1:5000/upload_image"  # This points to the uploadImage.py Flask server
    files = {'file': file}
    data = {'username': username}

    # Send a POST request to upload the image and generate the embedding
    response = requests.post(url, files=files, data=data)

    # Return the response from uploadImage.py server
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to upload the image or generate embedding"}), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  
