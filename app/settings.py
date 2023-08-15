import os

from flask import Flask


app = Flask(__name__)

db_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../static', 'db', 'database.db')


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_file_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking for better performance
app.config['EMB_PATH'] = os.path.join(os.path.curdir, "../static", "embeddings")
app.config['YOLO_MODEL_PATH'] = os.path.join(os.path.curdir, "../static", "ml", "yolov4-custom-detector_last_pos_only.weights")
app.config['YOLO_CONF_PATH'] = os.path.join(os.path.curdir, "../static", "ml", "yolov4-custom-detector.cfg")
app.config['FACENET_PATH'] = os.path.join(os.path.curdir, "../static", "ml", "facenet_keras_weights.h5")
app.config['TMP_PATH'] = os.path.join(os.path.curdir,"../static","tmp")