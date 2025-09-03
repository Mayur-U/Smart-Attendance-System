import os
from deepface import DeepFace
DeepFace.build_model("ArcFace")  # pre-load with torch
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # avoid TensorFlow optimizations
os.environ["KERAS_BACKEND"] = "torch"       # force torch backend


import os
import cv2
import numpy as np
import pickle
import time
import pandas as pd
import smtplib
import ssl
from email.message import EmailMessage
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# NEW: torch for device selection; ultralytics for YOLO
import torch
from ultralytics import YOLO
from deepface import DeepFace

# ---------------------------
# IMPORTANT: GPU ENABLE
# ---------------------------
# Removed: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Init] Torch device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")
if DEVICE == "cpu":
    print("[Init] Running on CPU. To use GPU, install CUDA + torch with CUDA, and TensorFlow-gpu for DeepFace.")

# --- Configuration ---
YOLO_MODEL_PATH = "yolov8n.pt"
ENCODING_FILE = "encodings_deepface.pkl"
STUDENT_CSV = "students_info/student_data.csv"

# --- OpenCV DNN Face Detector Configuration (Optional, kept as fallback) ---
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD_DNN = 0.5

# --- DeepFace Configuration (FASTER) ---
DEEPFACE_MODEL_NAME = "ArcFace"  # Faster & accurate; alternatives: "Facenet512"
DISTANCE_METRIC = "cosine"       # 'cosine', 'euclidean', 'euclidean_l2'
RECOGNITION_THRESHOLD = 0.50     # Tune after testing (cosine: lower is better)

# --- Performance knobs (NEW) ---
DETECT_EVERY_N_FRAMES = 3        # Run YOLO/recognition every Nth frame (reduce load)
YOLO_IMG_SIZE = 416              # Smaller inference size for speed (320/416/512/640)
MAX_EMBED_THREADS = 2            # Thread pool for DeepFace per detection cycle
SESSION_RECOOL_S = 0.0           # Cooldown per face (kept simple; recognition only on detect frames)

# --- Camera Configuration (LOWER RES for speed) ---
CAMERA_INDEX = 0                 # 0 default; or RTSP/HTTP string
DESIRED_WIDTH = 960              # 960x540 is a good balance
DESIRED_HEIGHT = 540

# --- Display Window Configuration ---
DISPLAY_WIDTH = 960              # UI size only

# --- Attendance & Feature Flags ---
ATTENDANCE_SESSION_DURATION = 2 * 60
REQUIRED_RECOGNITION_DURATION = 1 * 60
HR_SESSION_ACTIVE = True
SNAPSHOT_INTERVAL = 1 * 60
RECORDING_FPS = 10               # Lower FPS to reduce IO load
RECORD_VIDEO = False             # Toggle recording to reduce lag

# --- Output Directories ---
SNAPSHOT_DIR = "attendance_info/snapshots"
RECORDINGS_DIR = "attendance_info/recordings"
REPORTS_DIR = "attendance_info/reports"

# --- Email Configuration ---
import config
EMAIL_SENDER = config.EMAIL_SENDER
EMAIL_PASSWORD = config.EMAIL_PASSWORD
EMAIL_SERVER = "smtp.gmail.com"
EMAIL_PORT = 465

# --- DeepFace distance helpers (robust import across versions) ---
print("[Init] Importing DeepFace distance functions...")
findCosineDistance = None
findEuclideanDistance = None
l2_normalize = None

try:
    # Newer layout
    from deepface.modules import verification
    findCosineDistance = verification.find_cosine_distance
    findEuclideanDistance = verification.find_euclidean_distance
    l2_normalize = verification.l2_normalize
    print("[Init] Loaded distance functions from deepface.modules.verification")
except Exception:
    try:
        # Older layout
        from deepface.commons import distance as verification_old
        findCosineDistance = verification_old.findCosineDistance
        findEuclideanDistance = verification_old.findEuclideanDistance
        l2_normalize = verification_old.l2_normalize
        print("[Init] Loaded distance functions from deepface.commons.distance")
    except Exception as e:
        print("FATAL: Could not import DeepFace distance functions. Error:", e)
        raise SystemExit(1)

# --- Create Output Directories ---
print("Creating output directories if they don't exist...")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Load YOLO Model ---
print("Loading YOLOv8 model...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    # Move model to GPU if available
    try:
        yolo_model.to(DEVICE)
    except Exception:
        pass  # .to may be a no-op in some Ultralytics versions; device passed at call-time anyway
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model from {YOLO_MODEL_PATH}: {e}")
    raise SystemExit(1)

# --- Load Face Encodings (Embeddings) ---
print(f"Loading known face embeddings from {ENCODING_FILE}...")
try:
    with open(ENCODING_FILE, "rb") as f:
        data = pickle.load(f)
        known_embeddings_list = data["embeddings"]
        known_names = data["names"]
        if not known_names:
            print("Error: No names found in encoding file.")
            raise SystemExit(1)
        known_embeddings_np = np.array(known_embeddings_list)
        unique_stripped_names = sorted(list(set(name.strip() for name in known_names)))
        print(f"Loaded {len(unique_stripped_names)} unique known names.")
        print(f"Total embeddings loaded: {len(known_embeddings_list)}")
except FileNotFoundError:
    print(f"Error: Encoding file not found at {ENCODING_FILE}")
    raise SystemExit(1)
except Exception as e:
    print(f"Error loading encoding file: {e}")
    raise SystemExit(1)

# --- Load Student Data (for email lookup) ---
print(f"Loading student data from {STUDENT_CSV}...")
student_info_map = {}
df_students = None
try:
    df_students = pd.read_csv(STUDENT_CSV, skipinitialspace=True)
    df_students.columns = df_students.columns.str.strip()
    required_cols = ['Name', 'StudentEmailAddress', 'ParentEmailAddress']
    if not all(col in df_students.columns for col in required_cols):
        print(f"Error: {STUDENT_CSV} must contain columns: {', '.join(required_cols)}")
        df_students = None
    else:
        df_students['Name'] = df_students['Name'].str.strip()
        for _, row in df_students.iterrows():
            name = row['Name']
            student_info_map[name] = {
                'StudentID': row.get('StudentID', 'N/A'),
                'StudentEmailAddress': row['StudentEmailAddress'],
                'ParentEmailAddress': row['ParentEmailAddress']
            }
        print(f"Loaded data for {len(student_info_map)} students from CSV.")
except FileNotFoundError:
    print(f"Warning: {STUDENT_CSV} not found. Cannot look up StudentID/Email.")
    df_students = None
except Exception as e:
    print(f"Error loading student data from {STUDENT_CSV}: {e}")
    df_students = None

# --- Load OpenCV DNN Model (Optional, fallback only) ---
print("Loading OpenCV DNN face detector model (optional fallback)...")
try:
    face_detector_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("OpenCV DNN model loaded successfully.")
except cv2.error as e:
    print(f"Warning: Could not load OpenCV DNN model: {e}. Will rely on DeepFace internal detector if needed.")
    face_detector_net = None

# --- Initialize Attendance ---
attendance_status = {name: 'Absent' for name in unique_stripped_names}
recognition_duration = {name: 0.0 for name in unique_stripped_names}
print(f"Initialized attendance for {len(attendance_status)} unique students.")

# --- Initialize Snapshot Variables ---
if HR_SESSION_ACTIVE:
    last_snapshot_time = time.time()

# --- Initialize Video Capture ---
print(f"Attempting to open video capture device: {CAMERA_INDEX}")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open video capture device {CAMERA_INDEX}.")
    raise SystemExit(1)

# --- Set Desired Resolution ---
print(f"Requesting camera resolution: {DESIRED_WIDTH}x{DESIRED_HEIGHT}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

# --- Get Actual Resolution and FPS ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_fps = cap.get(cv2.CAP_PROP_FPS)
actual_recording_fps = RECORDING_FPS
if cam_fps > 0 and cam_fps < RECORDING_FPS:
    print(f"Warning: Camera FPS ({cam_fps}) < desired recording FPS ({RECORDING_FPS}). Using camera FPS.")
    actual_recording_fps = cam_fps
elif cam_fps == 0:
    print(f"Warning: Camera did not report FPS. Using configured recording FPS ({RECORDING_FPS}).")

print(f"Actual camera resolution set to: {frame_width}x{frame_height}")
print(f"Using recording FPS: {actual_recording_fps}")

# --- Initialize Video Writer (optional) ---
video_writer = None
video_filename = None
if RECORD_VIDEO:
    timestamp_vid = time.strftime("%Y-%m-%d_%I%M%S%p")
    video_filename = os.path.join(RECORDINGS_DIR, f"attendance_recording_{timestamp_vid}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, actual_recording_fps, (frame_width, frame_height))
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {video_filename} using mp4v codec.")
        print("Install H.264 support (ffmpeg/gstreamer) or set RECORD_VIDEO=False.")
        video_writer = None
    else:
        print(f"Recording video to {video_filename} at {actual_recording_fps} FPS.")

session_start_time = time.time()
previous_frame_time = session_start_time

# --- Helper: Distance Calculation ---
def find_distance(emb1, emb2, metric=DISTANCE_METRIC):
    try:
        if metric == 'cosine':
            return findCosineDistance(emb1, emb2)
        elif metric == 'euclidean':
            return findEuclideanDistance(emb1, emb2)
        elif metric == 'euclidean_l2':
            return findEuclideanDistance(l2_normalize(emb1), l2_normalize(emb2))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
    except Exception as e:
        print(f"\nError during distance calculation: {e}")
        return float('inf')

# --- Helper: Snapshot (already threaded) ---
def take_snapshot(frame_to_save):
    timestamp = time.strftime("%Y-%m-%d_%I%M%S%p")
    filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
    try:
        cv2.imwrite(filename, frame_to_save)
        print(f"\nSnapshot saved: {filename}")
    except Exception as e:
        print(f"\nError saving snapshot: {e}")

# NEW: Cache last detection boxes & names for skipped frames
last_boxes_xyxy = np.empty((0, 4), dtype=int)
last_names = []  # parallel to last_boxes_xyxy
last_colors = [] # to draw consistent colors
frame_count = 0

# --- Main Loop ---
print("Starting real-time attendance (optimized)...")
while True:
    current_frame_time = time.time()
    time_elapsed = current_frame_time - previous_frame_time

    ret, frame = cap.read()
    if not ret:
        if isinstance(CAMERA_INDEX, str):
            print("End of video file reached.")
            break
        else:
            print("Error: Failed to grab frame from camera.")
            break

    frame_count += 1
    recognized_names_this_frame = set()

    run_heavy = (frame_count % DETECT_EVERY_N_FRAMES == 0)

    if run_heavy:
        # --- YOLO Face Detection this cycle ---
        try:
            yolo_results = yolo_model(frame, imgsz=YOLO_IMG_SIZE, device=0 if DEVICE == "cuda" else "cpu", verbose=False)
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")
            yolo_results = []

        boxes = []
        confs = []

        if len(yolo_results) > 0 and hasattr(yolo_results[0], 'boxes') and yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy
            confs = yolo_results[0].boxes.conf
            try:
                boxes = boxes.detach().cpu().numpy().astype(int)
                confs = confs.detach().cpu().numpy()
            except Exception:
                boxes = np.array(boxes).astype(int)
                confs = np.array(confs)
        else:
            boxes = np.empty((0,4), dtype=int)
            confs = np.array([])

        # --- Prepare face crops ---
        crops = []
        crop_boxes = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            pad = 5
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(frame.shape[1], x2 + pad)
            crop_y2 = min(frame.shape[0], y2 + pad)

            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if face_crop.size == 0:
                continue
            crops.append(face_crop)
            crop_boxes.append((x1, y1, x2, y2))

        names_this_cycle = ["Unknown"] * len(crops)
        colors_this_cycle = [(0, 0, 255)] * len(crops)

        # --- DeepFace embeddings in small thread pool (per detect cycle) ---
        def embed_and_match(face_img):
            try:
                emb_objs = DeepFace.represent(
                    img_path=face_img,
                    model_name=DEEPFACE_MODEL_NAME,
                    enforce_detection=True,
                    detector_backend='opencv'  # fast
                )
                if emb_objs and isinstance(emb_objs, list) and 'embedding' in emb_objs[0]:
                    current_embedding = np.array(emb_objs[0]['embedding'])

                    min_dist = float('inf')
                    best_idx = -1
                    for j, known_embedding in enumerate(known_embeddings_np):
                        dist = find_distance(current_embedding, known_embedding, metric=DISTANCE_METRIC)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = j

                    if best_idx != -1 and min_dist < RECOGNITION_THRESHOLD:
                        return known_names[best_idx].strip(), (0, 255, 0)
            except ValueError:
                pass
            except Exception as e:
                print(f"[DeepFace] Error: {e}")
            return "Unknown", (0, 0, 255)

        if len(crops) > 0:
            with ThreadPoolExecutor(max_workers=MAX_EMBED_THREADS) as ex:
                futures = [ex.submit(embed_and_match, c) for c in crops]
                for idx, fut in enumerate(as_completed(futures)):
                    name, color = fut.result()
                    names_this_cycle[idx] = name
                    colors_this_cycle[idx] = color

        # Update last_* for skipped frames to draw
        last_boxes_xyxy = np.array(crop_boxes, dtype=int) if len(crop_boxes) > 0 else np.empty((0,4), dtype=int)
        last_names = names_this_cycle
        last_colors = colors_this_cycle

        # Attendance logic on detect frames
        for nm in names_this_cycle:
            if nm != "Unknown":
                recognized_names_this_frame.add(nm)
                if nm in recognition_duration:
                    recognition_duration[nm] += time_elapsed
                    if attendance_status[nm] == 'Absent' and recognition_duration[nm] >= REQUIRED_RECOGNITION_DURATION:
                        attendance_status[nm] = 'Present'
                        print(f"\n>> {nm} marked PRESENT.")
                else:
                    print(f"Warning: Recognized name '{nm}' not in duration tracking list.")

    # --- Draw last known boxes/names (for skipped frames we still show something smooth) ---
    for i in range(len(last_boxes_xyxy)):
        x1, y1, x2, y2 = last_boxes_xyxy[i]
        name = last_names[i] if i < len(last_names) else "Unknown"
        color = last_colors[i] if i < len(last_colors) else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10
        cv2.rectangle(frame, (x1, text_y - text_height - baseline), (x1 + text_width, text_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # --- Display Recognized Count ---
    count_text = f"Recognized Now: {len(set(last_names) - {'Unknown'}) if len(last_names)>0 else 0}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Display Frame (resized for UI only) ---
    aspect_ratio = frame_height / frame_width if frame_width > 0 else 1
    display_height = int(DISPLAY_WIDTH * aspect_ratio)
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
    cv2.imshow('Attendance System (GPU-Optimized)', display_frame)

    # --- Write the original frame to Video (optional) ---
    if video_writer is not None:
        try:
            video_writer.write(frame)
        except Exception as e:
            print(f"\nError writing frame to video: {e}")

    # --- Snapshot Logic ---
    current_time_snapshot = time.time()
    if HR_SESSION_ACTIVE and (current_time_snapshot - last_snapshot_time >= SNAPSHOT_INTERVAL):
        snapshot_thread = threading.Thread(target=take_snapshot, args=(frame.copy(),))
        snapshot_thread.daemon = True
        snapshot_thread.start()
        last_snapshot_time = current_time_snapshot

    previous_frame_time = current_frame_time

    # --- Session Timeout ---
    if time.time() - session_start_time > ATTENDANCE_SESSION_DURATION:
        print(f"\nAttendance session duration ({ATTENDANCE_SESSION_DURATION / 60.0:.1f} minutes) reached. Exiting...")
        break

    # --- Exit Key ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting application manually...")
        break

# --- Cleanup ---
cap.release()
if video_writer is not None:
    video_writer.release()
    print(f"Video recording saved to {video_filename}")
cv2.destroyAllWindows()
print("Video capture released and windows closed.")

# --- Generate Excel Report ---
print("\nGenerating attendance report...")
attendance_list = []
print(f"Generating report for {len(unique_stripped_names)} unique names found in encodings file.")

for name in unique_stripped_names:
    status = attendance_status.get(name, 'Absent')
    student_id = 'N/A'
    student_email = 'N/A'
    parent_email = 'N/A'
    found_in_csv = False

    if name in student_info_map:
        info = student_info_map[name]
        student_id = info.get('StudentID', 'N/A')
        student_email = info.get('StudentEmailAddress', 'N/A')
        parent_email = info.get('ParentEmailAddress', 'N/A')
        found_in_csv = True

    if not found_in_csv:
        print(f"  DEBUG: Could not find details for '{name}' in {STUDENT_CSV}. Using N/A.")

    duration_seen = recognition_duration.get(name, 0.0)
    duration_min = duration_seen / 60.0
    attendance_list.append({
        'StudentID': student_id,
        'Name': name,
        'StudentEmailAddress': student_email,
        'ParentEmailAddress': parent_email,
        'Status': status,
        'DurationSeen_min': round(duration_min, 2)
    })

df_report = pd.DataFrame(attendance_list)
timestamp = time.strftime("%Y-%m-%d_%I-%M-%S%p")
excel_filename = os.path.join(REPORTS_DIR, f"attendance_report_{timestamp}.xlsx")
try:
    df_report.to_excel(excel_filename, index=False)
    print(f"Attendance report saved to {excel_filename}")
except Exception as e:
    print(f"Error saving Excel report: {e}")

# --- Send Emails to Absentees ---
print("\nSending emails to absentees...")
absentees = df_report[df_report['Status'] == 'Absent']

if (not EMAIL_SENDER or "@" not in EMAIL_SENDER or not EMAIL_PASSWORD or EMAIL_PASSWORD == "your_app_password"):
    print("Email sender or password not configured correctly. Skipping email notifications.")
    print("Please configure EMAIL_SENDER and EMAIL_PASSWORD (use Gmail App Password).")
elif absentees.empty:
    print("No absentees found.")
else:
    sent_count_student = 0; sent_count_parent = 0
    error_count = 0; skipped_count = 0
    print(f"Attempting to send emails via {EMAIL_SERVER}:{EMAIL_PORT} as {EMAIL_SENDER}...")
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(EMAIL_SERVER, EMAIL_PORT, context=context, timeout=30) as server:
            print("Connecting to SMTP server...")
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            print("Logged into email server successfully.")
            for _, row in absentees.iterrows():
                student_email = row['StudentEmailAddress']
                parent_email = row['ParentEmailAddress']
                recipient_name = row['Name']
                current_time_str = time.strftime('%Y-%m-%d at %I:%M:%S %p')

                # Student email
                if pd.notna(student_email) and isinstance(student_email, str) and "@" in student_email:
                    subject_student = "Absence Notification - Attendance System"
                    body_student = (
                        f"Dear {recipient_name},\n\n"
                        f"You were marked as absent by the automated attendance system for the session on {current_time_str}.\n\n"
                        f"Please contact the instructor/administrator if you believe this is an error.\n\nRegards,\nAttendance System"
                    )
                    em_student = EmailMessage()
                    em_student['From'] = EMAIL_SENDER
                    em_student['To'] = student_email
                    em_student['Subject'] = subject_student
                    em_student.set_content(body_student)
                    try:
                        print(f"Sending email to student {recipient_name} ({student_email})...")
                        server.send_message(em_student)
                        print("  > Email to student sent successfully.")
                        sent_count_student += 1
                    except Exception as e_send_student:
                        print(f"  > Failed to send email to student {recipient_name}: {e_send_student}")
                        error_count += 1
                    time.sleep(0.5)
                else:
                    print(f"Skipping email for student {recipient_name} (invalid or missing student email).")
                    skipped_count += 1

                # Parent email
                if pd.notna(parent_email) and isinstance(parent_email, str) and "@" in parent_email:
                    subject_parent = f"Absence Notification for {recipient_name} - Attendance System"
                    body_parent = (
                        f"Dear Parent/Guardian of {recipient_name},\n\n"
                        f"This is to inform you that {recipient_name} was marked as absent by the automated attendance system for the class session on {current_time_str}.\n\n"
                        f"Please contact the instructor/administrator for further details if needed.\n\nRegards,\nAttendance System"
                    )
                    em_parent = EmailMessage()
                    em_parent['From'] = EMAIL_SENDER
                    em_parent['To'] = parent_email
                    em_parent['Subject'] = subject_parent
                    em_parent.set_content(body_parent)
                    try:
                        print(f"Sending email to parent of {recipient_name} ({parent_email})...")
                        server.send_message(em_parent)
                        print("  > Email to parent sent successfully.")
                        sent_count_parent += 1
                    except Exception as e_send_parent:
                        print(f"  > Failed to send email to parent of {recipient_name}: {e_send_parent}")
                        error_count += 1
                    time.sleep(1.0)
                else:
                    print(f"Skipping email for parent of {recipient_name} (invalid or missing parent email).")

    except smtplib.SMTPAuthenticationError:
        print("\nSMTP Authentication Error...")
        error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count
    except smtplib.SMTPConnectError:
        print(f"\nSMTP Connection Error...")
        error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count
    except ssl.SSLError as e:
        print(f"\nSSL Error during SMTP connection: {e}")
        error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count
    except Exception as e:
        print(f"\nFailed to connect to email server or send emails: {e}")
        error_count = len(absentees)*2 - sent_count_student - sent_count_parent - skipped_count

    print(f"\nEmail Summary: Student Emails Sent={sent_count_student}, Parent Emails Sent={sent_count_parent}, Failed={error_count}, Skipped={skipped_count}")

print("\nApplication finished.")
