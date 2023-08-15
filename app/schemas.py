# app/schemas.py

from marshmallow import Schema, fields, validate

class StudentSchema(Schema):
    firstname = fields.String(required=True, validate=validate.Length(min=1))
    lastname = fields.String(required=True, validate=validate.Length(min=1))
    email = fields.Email(required=True)
    password = fields.String(required=True, validate=validate.Length(min=7))
    id_card = fields.String(required=True)
    face_emb_path = fields.String(required=True)

class StudentCredentials(Schema):
    email = fields.Email(required=True)
    password = fields.String(required=True,validate=validate.Length(min=7))

