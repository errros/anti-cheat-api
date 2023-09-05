import os

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
db_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), './static', 'db', 'database.db')


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_file_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking for better performance
app.config['EMB_PATH'] = os.path.join(os.path.curdir, "./static", "embeddings")
app.config['YOLO_MODEL_PATH'] = os.path.join(os.path.curdir, "./static", "ml", "yolov4-custom-detector_last.weights")
app.config['YOLO_CONF_PATH'] = os.path.join(os.path.curdir, "./static", "ml", "yolov4-custom-detector.cfg")
app.config['TMP_PATH'] = os.path.join(os.path.curdir,"./static","tmp")