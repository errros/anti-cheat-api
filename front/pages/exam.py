import os

import cv2
import numpy as np
import requests
import streamlit as st
import av  # Add this import
import io  # Add this import
import redis

from streamlit.runtime.uploaded_file_manager import UploadedFile
import redis

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


from streamlit_webrtc import webrtc_streamer, WebRtcMode
from _pickle import dumps, loads

global store
global sio
exam_started = False



def captured_pic(frame:av.VideoFrame):
    global exam_started  # Declare the global variable
    if not exam_started:
        payload={
            "exam_duration": 2
        }
        response = requests.post("http://localhost:5000/api/monitor",data=payload)
        if response.status_code == 201:
            exam_started = True

    image = frame.to_image()
    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_data = image_bytes.getvalue()
    store.get_connection().publish("frame",image_data)




def monitor():
    st.title("Exam Surveillance!")
    webrtc_streamer(key="stream",media_stream_constraints={"video": {"frameRate":1.0}, "audio": False},video_frame_callback=captured_pic)



if __name__ == "__main__":
    #sio = SocketConnection().sio
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

