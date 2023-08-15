from flask import Blueprint, request, jsonify
from marshmallow import ValidationError

import crud
from models import Student, db
from schemas import StudentSchema, StudentCredentials
from service import validate_image, save_image , generate_face_embedding , save_face_embedding , compare_embeddings,face_emb_retrieval


bp = Blueprint('api', __name__)

@bp.route('/auth',methods=['GET'])
def authenticate():
    email = request.form.get('email')
    password = request.form.get('password')
    image_file = request.files.get('image')

    image_path = save_image(image_file)

    if not validate_image(image_path):
        return jsonify("please provide a right picture!"),400

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

    emb_original = face_emb_retrieval(credentials_data)
    if(compare_embeddings(emb_auth,emb_original)<0.75):
        return jsonify("Only the person could pass his exam"),403

    return jsonify("Authenticated"),200




@bp.route('/student', methods=['POST'])
def create_student():

    firstname = request.form.get('firstname')
    lastname = request.form.get('lastname')
    email = request.form.get('email')
    password = request.form.get('password')
    id_card = request.form.get('id_card')
    image_file = request.files.get('image')


    image_path = save_image(image_file)

    if not validate_image(image_path):
        return jsonify("please provide a right picture!"),400

    emb = generate_face_embedding(image_path)
    face_emb_path =save_face_embedding(emb)


    # Validate the form student_schema=
    student_req = {
        'firstname': firstname,
        'lastname': lastname,
        'email': email,
        'password': password,
        'id_card': id_card,
        "face_emb_path" : face_emb_path
    }

    student_schema = StudentSchema()
    try:
        # Validate the data using the schema instance
        validated_data = student_schema.load(student_req)
    except ValidationError as error:
        return jsonify(error.messages), 400



    crud.save_student(student_req,face_emb_path)
    return jsonify({'message': 'Student created successfully'}), 201


@bp.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask!'})
