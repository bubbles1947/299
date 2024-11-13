import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pickle
import mysql.connector
from insightface.app import FaceAnalysis

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
        messagebox.showinfo("Success", "Embedding generated and uploaded successfully!")
    else:
        messagebox.showerror("Error", "No face detected in the image.")

# GUI for image upload and embedding generation
def create_gui():
    # Connect to database
    connection = connect_to_db()

    # Define functions for the GUI
    def select_image():
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select an Image",
                                               filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            username = username_entry.get()
            if username:
                # Generate and upload the embedding
                process_image_and_upload(file_path, username, connection)
            else:
                messagebox.showwarning("Input Error", "Please enter a username.")
    
    # Set up the GUI window
    root = tk.Tk()
    root.title("Face Embedding Uploader")
    root.geometry("400x200")

    # Username input
    tk.Label(root, text="Enter Username:").pack(pady=10)
    username_entry = tk.Entry(root)
    username_entry.pack()

    # Upload button
    upload_button = tk.Button(root, text="Upload Image and Generate Embedding", command=select_image)
    upload_button.pack(pady=20)

    # Run the GUI loop
    root.mainloop()

    # Close the database connection when GUI is closed
    if connection:
        connection.close()

# Run the GUI application
if __name__ == "__main__":
    create_gui()
