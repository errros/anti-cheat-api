import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from models import db
from settings import app


with app.app_context():
    db.create_all()

from routes import bp

app.register_blueprint(bp, url_prefix='/api')


if __name__ == '__main__':
    app
    app.run(debug=True)  # You can set debug to False for production
