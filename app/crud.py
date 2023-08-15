from models import Student, db
from schemas import StudentSchema, StudentCredentials


def save_student(student, face_emb_path):
    new_student = Student(
        firstname=student['firstname'],
        lastname=student['lastname'],
        email=student['email'],
        password=student['password'],
        id_card=student['id_card'],
        face_emb_path=face_emb_path
    )

    db.session.add(new_student)
    db.session.commit()


def get_student_by_credentials(credentials):
    return db.session.query(Student).\
        filter(Student.email == credentials['email'] ,Student.password == credentials['password']).one_or_none()

