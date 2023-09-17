from multiprocessing import Process

import redis
from flask import Blueprint, request, jsonify
from marshmallow import ValidationError

import crud
from schemas import StudentSchema, StudentCredentials
from service import save_image, generate_face_embedding, save_face_embedding, compare_embeddings, \
    face_emb_retrieval, validate_and_save_face, is_cheating, clear_tmp

bp = Blueprint('api', __name__)


@bp.route('/auth', methods=['GET'])
def authenticate():
    email = request.form.get('email')
    password = request.form.get('password')
    image_file = request.files.get('image')
    image_path = save_image(image_file)

    try:
        image_path = validate_and_save_face(image_path)
    except Exception as e:
        return jsonify("Failed authentifcation"), 400

    emb_auth = generate_face_embedding(image_path)

    # Create a StudentCredentials instance and validate the data
    credentials_data = {
        'email': email,
        'password': password
    }
    credentials_schema = StudentCredentials()

    try:
        credentials = credentials_schema.load(credentials_data)
    except ValidationError as error:
        return jsonify(error.messages), 400

    try:
        emb_original, emb_path = face_emb_retrieval(credentials_data)
    except RuntimeError as e:
        return jsonify("there's no student with such credentials!"), 404

    score = compare_embeddings(emb_auth, emb_original)
    print(f'score of face comparison for login is {score}')
    if (score < 0.75):
        return jsonify("Only the person could pass his exam"), 403

    store = redis.Redis(host='localhost', port=6379, db=0)
    store.set("user_emb_path", emb_path)
    store.set("user_email", email.replace("'", ""))

    return jsonify("Authenticated"), 200


@bp.route('/student', methods=['POST'])
def create_student():
    firstname = request.form.get('firstname')
    lastname = request.form.get('lastname')
    email = request.form.get('email')
    password = request.form.get('password')
    id_card = request.form.get('id_card')
    image_file = request.files.get('image')
    print(f'image_file = {image_file}')

    image_path = save_image(image_file)

    try:
        image_path = validate_and_save_face(image_path)
    except Exception as e:
        return jsonify(e.messages), 400

    emb = generate_face_embedding(image_path)
    face_emb_path = save_face_embedding(emb)

    # Validate the form student_schema=
    student_req = {
        'firstname': firstname,
        'lastname': lastname,
        'email': email,
        'password': password,
        'id_card': id_card,
        "face_emb_path": face_emb_path
    }

    try:
        student_schema = StudentSchema()
        crud.save_student(student_req, face_emb_path)

        # Validate the data using the schema instance
        validated_data = student_schema.load(student_req)
    except ValidationError as error:
        return jsonify(error.messages), 400

    return jsonify({'message': 'Student created successfully'}), 201


@bp.route('/monitor', methods=['POST'])
def take_exam():
    duration = request.form.get('exam_duration')

    process = Process(target=is_cheating, args=(duration,))
    process.start()
    clear_tmp()

    return jsonify({'message': 'Exam Monitoring started!'}), 201


@bp.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask!'})
