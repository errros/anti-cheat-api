import time

import requests
import streamlit as st
from PIL import Image
from marshmallow import ValidationError, Schema, fields, validate
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu


class StudentInfosSchema(Schema):
    firstname = fields.String(required=True, validate=validate.Length(min=1))
    lastname = fields.String(required=True, validate=validate.Length(min=1))
    email = fields.Email(required=True)
    password = fields.String(required=True, validate=validate.Length(min=7))
    id_card = fields.String(required=True)


def validate_form(firstname, lastname, email, password, id_card, picture):
    # Validate the form student_schema=
    student_form = {
        'firstname': firstname,
        'lastname': lastname,
        'email': email,
        'password': password,
        'id_card': id_card
    }

    student_schema = StudentInfosSchema()
    validated_data = student_schema.load(student_form)
    if picture is None:
        raise ValidationError("Take a picture of yourself!")


def resize_image(image, new_size=(300, 300)):
    resized_image = image.resize(new_size, Image.ANTIALIAS)
    return resized_image


# st.set_page_config(page_title="Exam",initial_sidebar_state="collapsed")

def register():
    st.title("Student Registration")
    with st.form("Join"):
        # Initialize error variable
        alert = st.empty()
        # Get user inputs
        firstname = st.text_input("First Name", placeholder="Firstname")
        lastname = st.text_input("Last Name", placeholder="Lastname")
        email = st.text_input("Email", placeholder="mail@mail.com")
        password = st.text_input("Password", type="password", placeholder="password")
        id_card = st.text_input("ID Card Number", placeholder="Identity Card Number")
        picture = st.camera_input(label="Student Face", help="Please provide a clear picture of yourself!")

        # Use st.form_submit_button's on_click event handler
        if st.form_submit_button('Register', use_container_width=True, type="primary"):
            # Call the validation function and catch any validation error
            try:
                validate_form(firstname, lastname, email, password, id_card, picture)

                payload = {
                    'firstname': firstname,
                    'lastname': lastname,
                    'email': email,
                    'password': password,
                    'id_card': id_card
                }

                # Create a files dictionary with the captured image
                files = {
                    'image': ('image.jpg', picture.getvalue(), 'image/jpeg')
                }

                response = requests.post('http://localhost/api/student', data=payload, files=files)
                if response.status_code == 201:
                    alert.info(response.text)
                    time.sleep(2)
                    switch_page("login")

                else:
                    alert.error(
                        "you failed registration if you have an account login instead , take a good clear picture of yourself , check the form again!")

            except ValidationError as error:
                alert.error(error)

    if st.button(label="Login", key="green-button", use_container_width=True):
        switch_page("login")


if __name__ == "__main__":
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", 'Take Exam', 'Register'],
                               icons=['house', 'play-circle', "house-add"], menu_icon="cast", default_index=2,
                               key="navbar")

        if selected == "Home":
            switch_page("home")

        elif selected == "Take Exam":
            switch_page("login")

        st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

    register()
