import os
import pickle
import cv2
import mysql.connector
import tkinter as tk
from tkinter import filedialog, messagebox
from generate_embeddings import face_app  # Assuming generate_embeddings.py contains the face_app initialization

# Database connection setup
def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',       # Your database host
            user='root',            # Your MySQL username
            password='password',    # Your MySQL password
            database='lunar_lore'   # Database name
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Function to upload embedding to the database
def upload_embedding_to_db(username, embedding_file_path):
    try:
        connection = connect_to_database()
        if connection is None:
            return
        
        cursor = connection.cursor()
        
        # Read the embedding from the .pkl file
        with open(embedding_file_path, 'rb') as f:
            embedding = f.read()
        
        # Prepare the SQL insert query
        image_filename = os.path.basename(embedding_file_path)
        insert_query = """INSERT INTO face_embeddings (username, embedding, image_filename)
                          VALUES (%s, %s, %s)"""
        
        # Execute the query and commit the transaction
        cursor.execute(insert_query, (username, embedding, image_filename))
        connection.commit()
        
        messagebox.showinfo("Success", f"Embedding for {username} uploaded successfully.")
    
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Database error: {err}")
    
    finally:
        cursor.close()
        connection.close()

# Function to handle the upload and processing
def process_upload():
    # Ask the user to select an image file using a file dialog
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")))
    
    if not image_path:
        return
    
    # Ask for the username input
    username = username_entry.get()
    
    if not username:
        messagebox.showerror("Input Error", "Please enter a username.")
        return
    
    # Ensure the image file exists
    if not os.path.exists(image_path):
        messagebox.showerror("Error", "Image file not found.")
        return
    
    # Generate embeddings using the existing generate_embeddings.py logic
    faces = face_app.get(cv2.imread(image_path))
    
    if len(faces) > 0:
        for i, face in enumerate(faces):
            # Save the generated embedding as a .pkl file
            embedding_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_face_{i+1}.pkl"
            embedding_path = os.path.join("embeddings", embedding_filename)

            # Save the embedding to a .pkl file
            with open(embedding_path, 'wb') as f:
                pickle.dump(face.embedding, f)
            
            print(f"Generated embedding saved to {embedding_path}")

            # Upload the embedding to the database
            upload_embedding_to_db(username, embedding_path)
    else:
        messagebox.showerror("Face Detection Error", "No faces detected in the image.")

# Creating the GUI window
root = tk.Tk()
root.title("Face Embedding Upload")

# Set the size of the window
root.geometry("400x300")

# Username label and entry field
username_label = tk.Label(root, text="Username:")
username_label.pack(pady=10)

username_entry = tk.Entry(root, width=30)
username_entry.pack(pady=5)

# Upload button
upload_button = tk.Button(root, text="Upload Image", command=process_upload, height=2, width=20)
upload_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
