import io
import os
import uuid
import cv2
import numpy as np
import torch

from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import functional as F

from mtcnn.mtcnn import MTCNN
from PIL import Image
from werkzeug.utils import secure_filename

import crud
from schemas import StudentCredentials
from settings import app


#from run import app


def validate_image(image_path):

    image = resize_image(image_path,300,300)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    if len(faces) > 1 or len(faces) == 0 or faces[0]['confidence'] < 0.9:
        print(f'confidence on face detection is : {faces[0]["confidence"]}')
        return False
    return True

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

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # Convert NumPy array to PyTorch tensor
    tensor_image = F.to_tensor(image)
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
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
    return np.load(path)


def resize_image(image_path, target_width, target_height):
    image = cv2.imread(image_path)

    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factors for width and height
    width_scale = target_width / original_width
    height_scale = target_height / original_height

    # Choose the smaller scaling factor to maintain the aspect ratio
    scale = min(width_scale, height_scale)

    # Calculate the new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image





def compare_embeddings(embedding1, embedding2):

    tensor1 = torch.tensor(embedding1[0])
    tensor2 = torch.tensor(embedding2[0])
    similarity = 1 - cosine(tensor1.detach().numpy(), tensor2.detach().numpy())
    return similarity

