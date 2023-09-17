from flask_sqlalchemy import SQLAlchemy

from settings import app

db = SQLAlchemy(app)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    id_card = db.Column(db.String(40), unique=True, nullable=False)
    face_emb_path = db.Column(db.String(255), nullable=False, unique=True)

    def __repr__(self):
        return f"Student(id={self.id}, firstname={self.firstname}, lastname={self.lastname}, email={self.email}, id_card={self.id_card})"
