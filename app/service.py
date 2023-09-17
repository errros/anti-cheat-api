import os
import shutil
import smtplib
import uuid
from email.message import EmailMessage
from enum import Enum

import cv2
import numpy as np
import pandas as pd
import redis as redis
import torch
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from torchvision.transforms import functional as F
from werkzeug.utils import secure_filename

import crud
from settings import app

store = redis.Redis(host='localhost', port=6379, db=0)

print("connected? = " + str(store.ping()))

resnet = InceptionResnetV1(pretrained='vggface2').eval()

net = cv2.dnn.readNet(app.config['YOLO_CONF_PATH'],
                      app.config['YOLO_MODEL_PATH'])

# List of classes for the YOLOv4 model
classes = ["phone-used"]


class ExamViolations(Enum):
    NOT_PRESENT = "No one was present in the camera"
    NOT_SAME_PERSON = "Not the student passing the exam"
    MORE_PEOPLE = "Other people appeared in your exam!"
    MOBILE_USAGE = "You appear to have used a phone!"


def is_cheating(duration=10):
    frame_counter = 1
    exam_taker_emb_path = store.get("user_emb_path")
    exam_taker_emb = np.load(exam_taker_emb_path)
    pubsub = store.pubsub()
    pubsub.subscribe("frame")
    # Create an empty DataFrame
    columns = ["Frame Number", "Violation", "Image Path", "Compare Score", "Phone Confidence"]
    df = pd.DataFrame(columns=columns)
    log_file_path = os.path.join(app.config["TMP_PATH"], "exam_log.csv")
    for message in pubsub.listen():
        if message["type"] == "message":
            violation = "Nothing"
            frame_bytes = message["data"]
            image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            save_path = os.path.join(app.config["TMP_PATH_LOGS"], f'original_{frame_counter}.jpg')
            cv2.imwrite(save_path, image)

            detector = MTCNN()
            # Detect faces in the image
            faces = detector.detect_faces(image)
            if len(faces) == 0:
                violation = ExamViolations.NOT_PRESENT.name
            elif len(faces) > 1:
                violation = ExamViolations.MORE_PEOPLE.name
            else:
                # Extract face only and turn it into grayscale image
                face_bbox = faces[0]['box']  # Bounding box of the detected face
                face_image = image[face_bbox[1]:face_bbox[1] + face_bbox[3], face_bbox[0]:face_bbox[0] + face_bbox[2]]
                # Convert the face image to grayscale
                gray_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
                # Save the grayscale detected face image
                save_path_gray = os.path.join(app.config["TMP_PATH"], f'detected_face_gray_{frame_counter}.jpg')
                cv2.imwrite(save_path_gray, gray_face_image)
                frame_counter = frame_counter + 1

                face_emb = generate_face_embedding(save_path_gray)

                score = compare_embeddings(face_emb, exam_taker_emb)
                detection_confidence = 0
                if (score < 0.6):
                    violation = ExamViolations.NOT_SAME_PERSON.name
                else:
                    phone_used_detected, detection_confidence = detect_phone_used(image)
                    if phone_used_detected:
                        violation = ExamViolations.MOBILE_USAGE.name
                # Append the log entry directly to the DataFrame
                log_entry = [frame_counter, violation, save_path, score, detection_confidence]

                if violation is not "Nothing":
                    df = df.append(pd.Series(log_entry, index=columns), ignore_index=True)
            if not pubsub.get_message():
                print("KEMELNA GOING TO END EVERYTHING")
                break
    df.to_csv(log_file_path, index=False)
    send_logs()


def send_logs():
    # Email configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'noreply.examlogs@gmail.com'
    smtp_password = 'vjseoyajruazqdes'

    sender_email = 'noreply.examlogs@gmail.com'
    recipient_email = 'a.merah@esi-sba.dz'

    # Create the email message
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = 'Exam logs'

    # Email body text
    student_email = store.get("user_email")
    body = f"Student with following email {student_email} passed the exam successfully. Check possible violations in the attached log file."
    msg.set_content(body)

    # Attach the log file
    log_file_path = os.path.join(app.config["TMP_PATH"], "exam_log.csv")
    with open(log_file_path, 'rb') as log_file:
        msg.add_attachment(log_file.read(), maintype='application', subtype='octet-stream',
                           filename=os.path.basename(log_file_path))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def detect_phone_used(image):
    # Get the height and width of the input image
    height, width, _ = image.shape

    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Initialize variables to store detection results
    phone_used_detected = False
    detection_confidence = 0.0

    # Forward pass through the network
    detections = net.forward(output_layer_names)

    # Iterate through the detections and check for phone-used class with confidence
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7 and class_id == 0:  # Class index 0 is for phone-used
                phone_used_detected = True
                detection_confidence = confidence

    return phone_used_detected, detection_confidence


def clear_tmp():
    # Define the directory path
    tmp_path = app.config['TMP_PATH']

    # Remove all files and directories in TMP_PATH
    try:
        for item in os.listdir(tmp_path):
            item_path = os.path.join(tmp_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print(f"Cleared directory: {tmp_path}")
    except Exception as e:
        print(f"An error occurred while clearing the directory: {str(e)}")

    # Create an empty directory called "logs"
    logs_directory = os.path.join(tmp_path, "logs")
    try:
        os.makedirs(logs_directory)
        print(f"Created empty 'logs' directory at: {logs_directory}")
    except Exception as e:
        print(f"An error occurred while creating the 'logs' directory: {str(e)}")


def validate_and_save_face(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    if len(faces) > 1 or len(faces) == 0 or faces[0]['confidence'] < 0.97:
        if len(faces) > 0:
            print(f'confidence on face detection is : {faces[0]["confidence"]}')
        raise RuntimeError("Please provide a clear picture of yourself only!")

    face_bbox = faces[0]['box']  # Bounding box of the detected face
    face_image = image[face_bbox[1]:face_bbox[1] + face_bbox[3], face_bbox[0]:face_bbox[0] + face_bbox[2]]

    # Save the detected face image
    # Convert the face image to grayscale
    gray_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    # Save the grayscale detected face image
    save_path = os.path.join(app.config["TMP_PATH"], 'detected_face_gray.jpg')
    cv2.imwrite(save_path, gray_face_image)
    return save_path


def save_image(image_file):
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        save_path = os.path.join(app.config["TMP_PATH"], filename)
        image_file.save(save_path)
        return save_path


def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def generate_face_embedding(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert grayscale image to RGB format (3 channels)
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # Convert NumPy array to PyTorch tensor
    tensor_image = F.to_tensor(rgb_image)
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)

    with torch.no_grad():
        return resnet(tensor_image)


def save_face_embedding(embedding):
    emb_filename = str(uuid.uuid4()) + '.npy'
    # Save the embedding as an .npy file
    emb_file_path = os.path.join(app.config['EMB_PATH'], emb_filename)
    np.save(emb_file_path, embedding.detach().numpy())
    return emb_file_path


def face_emb_retrieval(credentials):
    student = crud.get_student_by_credentials(credentials)
    if student is None:
        raise RuntimeError("there's no student with such credentials!")
    path = student.face_emb_path
    return np.load(path), path


def compare_embeddings(embedding1, embedding2):
    tensor1 = torch.tensor(embedding1[0])
    tensor2 = torch.tensor(embedding2[0])
    similarity = 1 - cosine(tensor1.detach().numpy(), tensor2.detach().numpy())
    return similarity
