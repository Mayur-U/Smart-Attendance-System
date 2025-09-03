import os
# --- Force TensorFlow CPU ---
# Set environment variable BEFORE importing TensorFlow (or libraries that import it like DeepFace)
# This tells TensorFlow to ignore GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2 # OpenCV for detection and image loading
import pickle
import csv
import numpy as np
import time
# Import DeepFace AFTER setting the environment variable
from deepface import DeepFace

# Optional: Verify TensorFlow sees only CPU
# import tensorflow as tf
# try:
#     tf_gpus = tf.config.list_physical_devices('GPU')
#     print(f"TensorFlow sees {len(tf_gpus)} GPUs.")
#     if not tf_gpus:
#         print("TensorFlow is correctly configured to use CPU only.")
#     else:
#         print("Warning: TensorFlow still sees GPUs despite environment variable.")
# except Exception as e:
#     print(f"Could not check TensorFlow devices: {e}")
# --- End Force TensorFlow CPU ---


# --- Configuration ---
# Path to the directory containing student image folders
IMAGE_DIR = "students_info/dataset/student_images"

# Path to the student data CSV file (optional but good for validation)
STUDENT_CSV = "students_info/student_data.csv"

# Output file to save the encodings (embeddings) and names
ENCODING_FILE = "encodings_deepface.pkl" # Changed filename slightly

# --- OpenCV DNN Face Detector Configuration ---
PROTOTXT_PATH = "deploy.prototxt" # Path to the downloaded prototxt file
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel" # Path to the downloaded caffemodel file
CONFIDENCE_THRESHOLD = 0.5 # Minimum probability to filter weak detections

# --- DeepFace Configuration ---
# Choose a model supported by DeepFace. 'VGG-Face' is default and robust.
DEEPFACE_MODEL_NAME = "VGG-Face"
DEEPFACE_DETECTOR_BACKEND = "opencv" # or 'skip' if we use our OpenCV detection

# --- Initialization ---
known_embeddings = [] # Store embeddings (encodings)
known_names = [] # Store names corresponding to embeddings
processed_students = set()
student_name_map = {} # Optional: Map StudentID to Name from CSV

print("Starting face encoding process...")
print(f"Using OpenCV DNN for face detection and DeepFace ({DEEPFACE_MODEL_NAME}) for encoding.")
print("Attempting to force CPU usage for TensorFlow backend.")

# --- Load OpenCV DNN Model (for our own detection) ---
try:
    print("Loading OpenCV DNN face detector model...")
    face_detector_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("Model loaded successfully.")
except cv2.error as e:
    print(f"Error loading OpenCV DNN model: {e}")
    print(f"Ensure '{PROTOTXT_PATH}' and '{MODEL_PATH}' are in the correct directory.")
    exit()

# --- Optional: Load student data from CSV for validation ---
try:
    with open(STUDENT_CSV, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            student_id = row.get('StudentID')
            student_name = row.get('Name')
            if student_id and student_name:
                student_name_map[student_id] = student_name
    print(f"Loaded student data from {STUDENT_CSV}")
except FileNotFoundError:
    print(f"Warning: {STUDENT_CSV} not found. Cannot validate names against CSV.")
except Exception as e:
    print(f"Error loading {STUDENT_CSV}: {e}")


# --- Iterate through student folders ---
if not os.path.isdir(IMAGE_DIR):
    print(f"Error: Image directory not found at {IMAGE_DIR}")
    exit()

print(f"Scanning image directory: {IMAGE_DIR}")

student_folders = [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]

if not student_folders:
    print(f"Error: No student folders found in {IMAGE_DIR}. Check the path and folder structure.")
    exit()

total_students = len(student_folders)
processed_count = 0
start_time_total = time.time()

for student_folder in student_folders:
    processed_count += 1
    student_path = os.path.join(IMAGE_DIR, student_folder)

    parts = student_folder.split('_', 1)
    student_id_from_folder = parts[0]
    student_name_from_folder = parts[1].replace('_', ' ') if len(parts) > 1 else student_id_from_folder
    student_name = student_name_map.get(student_id_from_folder, student_name_from_folder)

    print(f"\nProcessing student ({processed_count}/{total_students}): {student_name} (Folder: {student_folder})")

    image_files = [f for f in os.listdir(student_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"  Warning: No image files found for {student_name}")
        continue

    student_encoded = False

    # --- Iterate through images for the current student ---
    for image_file in image_files:
        image_path = os.path.join(student_path, image_file)
        print(f"  Processing image: {image_file}...", end="")
        start_time_image = time.time()

        try:
            # Load the image using OpenCV (DeepFace prefers this)
            image = cv2.imread(image_path)
            if image is None:
                print(" Error: Could not read image.")
                continue

            # --- Optional: Use OpenCV DNN to find the main face first ---
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            face_detector_net.setInput(blob)
            detections = face_detector_net.forward()

            best_face_crop = None
            max_confidence = 0

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD:
                     if confidence > max_confidence: # Keep only the most confident detection
                        max_confidence = confidence
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        pad = 10
                        startX = max(0, startX - pad)
                        startY = max(0, startY - pad)
                        endX = min(w, endX + pad)
                        endY = min(h, endY + pad)
                        best_face_crop = image[startY:endY, startX:endX]

            if best_face_crop is None:
                 print(" No face found by OpenCV detector.")
                 continue
            # --- End of Optional OpenCV Detection ---


            # --- Generate Face Embedding using DeepFace ---
            # Pass the cropped face image directly to DeepFace.represent
            embedding_objs = DeepFace.represent(
                img_path = best_face_crop, # Use the cropped face
                model_name = DEEPFACE_MODEL_NAME,
                enforce_detection = False, # We already cropped the face
                detector_backend = 'skip' # Skip internal detection
            )

            if embedding_objs and 'embedding' in embedding_objs[0]:
                embedding = embedding_objs[0]['embedding'] # Extract the embedding vector
                known_embeddings.append(embedding)
                known_names.append(student_name)
                student_encoded = True
                total_time_image = time.time() - start_time_image
                print(f" Encoded successfully (total: {total_time_image:.2f}s).")
            else:
                print(" Could not generate embedding using DeepFace.")


        except ValueError as ve:
            if "Face could not be detected" in str(ve) or "cannot reshape" in str(ve):
                 print(f" DeepFace couldn't process the detected face region.")
            else:
                 print(f" Error processing {image_file}: {ve}")
        except Exception as e:
             # Catch other potential errors, including TensorFlow ones now
            print(f" Error processing {image_file}: {e}")


    if student_encoded:
        processed_students.add(student_name)

# --- Save the embeddings and names ---
end_time_total = time.time()
print(f"\nProcessed {len(processed_students)} unique students.")
print(f"Generated {len(known_embeddings)} total embeddings.")
print(f"Total processing time: {end_time_total - start_time_total:.2f} seconds.")

if known_embeddings:
    print(f"Saving embeddings and names to {ENCODING_FILE}...")
    data_to_save = {"embeddings": np.array(known_embeddings), "names": known_names}
    try:
        with open(ENCODING_FILE, "wb") as f:
            pickle.dump(data_to_save, f)
        print("Data saved successfully using pickle.")
    except Exception as e:
        print(f"Error saving data with pickle: {e}")
else:
    print("No embeddings were generated. Nothing to save.")

print("\nEncoding process finished.")