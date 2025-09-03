# Smart Attendance System using Face Recognition

This project implements an automated attendance system that utilizes real-time face detection and recognition to mark student attendance. It generates attendance reports, records video sessions, takes periodic snapshots, and can notify absent students via email.

## Features

* **Real-Time Face Detection:** Uses a custom-trained YOLOv8 model to detect faces from a live camera feed.
* **Real-Time Face Recognition:** Employs the DeepFace library (with models like VGG-Face) to recognize detected faces against a database of known student embeddings.
* **Automated Attendance Marking:** Marks students as "Present" upon successful recognition during the session. Students not detected remain "Absent".
* **Unknown Face Handling:** Detects faces not present in the known database and labels them as "Unknown".
* **Excel Report Generation:** Creates an Excel (`.xlsx`) file summarizing the attendance status (StudentID, Name, EmailAddress, Status) for all registered students at the end of the session, saved in a `reports/` directory.
* **Email Notification:** Automatically sends email notifications to students marked as "Absent" (requires configuration).
* **Session Video Recording:** Records the entire processed video stream (with bounding boxes and labels) to an MP4 file, saved in a `recordings/` directory.
* **Periodic Snapshots (Optional):** Can be configured to save snapshots of the video feed at regular intervals (e.g., every 20 minutes for HR sessions), saved in a `snapshots/` directory.
* **Configurable:** Allows setting camera resolution, recognition thresholds, email credentials, output directories, and recording FPS.

## Technology Stack

* **Language:** Python 3
* **Face Detection:** Ultralytics YOLOv8 (custom trained model)
* **Face Recognition:** DeepFace (using TensorFlow backend)
* **Core Libraries:**
    * OpenCV (`opencv-python`) - Camera access, image/video processing, drawing. 
    * NumPy - Numerical operations.
    * Pandas & Openpyxl - Data handling and Excel report generation.
    * Ultralytics - Loading and running the YOLOv8 model.
    * DeepFace - Face embedding generation and comparison.
    * TensorFlow (`tensorflow`, `tf-keras`) - Backend for DeepFace models.
    * smtplib, ssl, email - Standard Python libraries for sending emails.

## Project Structure

attendance/
│
├── attendance_app.py           # Main real-time application script
├── encode_faces.py             # Script to generate face embeddings
├── requirements.txt            # Python package dependencies
├── config.py                   # (Recommended) For storing email credentials securely
├── student_data.csv            # CSV file with StudentID, Name, EmailAddress
├── .gitignore                  # Git ignore file
├── README.md                   # This file
│
├── dataset/                    # Folder for training/encoding data
│   └── student_images/         # Contains subfolders for each student
│       ├── StudentID_Name1/    # Folder for student 1
│       │   ├── image1.jpg
│       │   └── image2.png
│       └── StudentID_Name2/    # Folder for student 2
│           └── image1.jpg
│
├── best.pt                 # Your trained YOLOv8 model weights
├── deploy.prototxt         # OpenCV DNN model structure
└── res10_300x300_ssd_iter_140000.caffemodel # OpenCV DNN model weights
│
├── encodings_deepface.pkl      # Output file from encode_faces.py
│
├── attendance_info/
│   └── reports/                    # Output directory for Excel reports (created automatically)
│   └──  recordings/                 # Output directory for video recordings (created automatically)
│   └── snapshots/                  # Output directory for periodic snapshots (created automatically)
│
└── attendance_env_new/         # Python virtual environment (should be in .gitignore)
## Setup and Usage

1.  **Clone the Repository:**
    ```bash
    git clone [<your-repository-url>](https://github.com/maaxone098/SMART-ATTENDANCE-SYSTEM.git)
    cd <repository-name>
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv attendance_env_new
    # Activate it:
    # Windows: .\attendance_env_new\Scripts\activate
    # Linux/macOS: source attendance_env_new/bin/activate
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    * Create the `student_data.csv` file with columns `StudentID`, `Name`, `EmailAddress`. Ensure names exactly match how they will be stored in the encodings (usually derived from folder names, stripped of leading/trailing spaces).
    * Create the `dataset/student_images/` directory.
    * Inside `student_images`, create a sub-folder for each student (e.g., `StudentID_Student Name`).
    * Place several clear photos of each student in their respective folder.

5.  **Obtain Models:**
    * Place your trained YOLOv8 model weights file (`best.pt`) in the project directory (or update `YOLO_MODEL_PATH` in `attendance_app.py`).
    * Download the OpenCV DNN face detector files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) and place them in the project directory (or update paths in the scripts).

6.  **Generate Face Encodings:**
    * Run the encoding script:
        ```bash
        python encode_faces.py
        ```
    * This will create the `encodings_deepface.pkl` file.

7.  **Configure Email (Important!):**
    * **(Recommended)** Create a `config.py` file:
        ```python
        # config.py
        EMAIL_SENDER = "your_email@gmail.com"
        EMAIL_PASSWORD = "your_16_digit_app_password" # Use Gmail App Password
        ```
    * Ensure `config.py` is listed in your `.gitignore` file.
    * Modify `attendance_app.py` to import and use `config.EMAIL_SENDER` and `config.EMAIL_PASSWORD` (if you haven't already).
    * *Alternatively (less secure):* Directly edit the `EMAIL_SENDER` and `EMAIL_PASSWORD` variables in `attendance_app.py`.

8.  **Run the Application:**
    ```bash
    python attendance_app.py
    ```
    * A window showing the camera feed with detections and recognitions should appear.
    * Press 'q' in the window to quit the application.

9.  **Check Outputs:**
    * The attendance report (`.xlsx`) will be saved in the `reports/` folder.
    * The video recording (`.mp4`) will be saved in the `recordings/` folder.
    * Snapshots (`.png`) will be saved in the `snapshots/` folder if `HR_SESSION_ACTIVE` is `True`.

## Notes

* Ensure camera permissions are granted for the script.
* Email sending requires correct credentials (preferably Gmail App Password) and appropriate security settings on the sender's email account (e.g., enabling 2FA).
* Recognition accuracy depends heavily on the quality of the encoding images and the live camera feed (lighting, angles, occlusions). Adjust `RECOGNITION_THRESHOLD` in `attendance_app.py` as needed.
* Performance (FPS) depends on system hardware (CPU/GPU) and the chosen camera resolution. Higher resolutions significantly increase processing load. Consider lowering `DESIRED_WIDTH`/`DESIRED_HEIGHT` or `RECORDING_FPS` if experiencing lag.