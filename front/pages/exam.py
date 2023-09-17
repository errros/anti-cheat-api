import io  # Add this import

import av  # Add this import
import redis
import requests
import streamlit as st


class RedisConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisConnection, cls).__new__(cls)
            cls._instance.init_connection()
        return cls._instance

    def init_connection(self):
        # Connect to Redis server
        self.connection = redis.Redis(host='localhost', port=6379, db=0)
        print("Connected to Redis server.")

    def get_connection(cls):
        return cls.connection


from streamlit_webrtc import webrtc_streamer

global store
global sio

exam_started = False
exam_cont = True
a = 0


def captured_pic(frame: av.VideoFrame):
    global exam_started  # Declare the global variable
    global a
    if not exam_started:
        payload = {
            "exam_duration": 2
        }
        response = requests.post("http://localhost:5000/api/monitor", data=payload)
        if response.status_code == 201:
            exam_started = True
    if a <= 50:
        image = frame.to_image()
        # Convert the image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_data = image_bytes.getvalue()
        store.get_connection().publish("frame", image_data)
        a += 1
        print(f'frame number {str(a)}')
    else:
        exam_cont = False


def monitor():
    st.title("Exam Surveillance!")
    a = 0

    webrtc_streamer(key="stream",
                    desired_playing_state=exam_cont,
                    media_stream_constraints={"video": {"frameRate": 1.0}, "audio": False},
                    video_frame_callback=captured_pic
                    )


if __name__ == "__main__":
    store = RedisConnection()
    st.set_page_config(page_title="Exam", initial_sidebar_state="collapsed")
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    
    """,
        unsafe_allow_html=True,
    )

    monitor()
