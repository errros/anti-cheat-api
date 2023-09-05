import requests
import streamlit as st
from marshmallow import ValidationError, Schema, fields,validate
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu


class CredentialsSchema(Schema):
    email = fields.Email(required=True)
    password = fields.String(required=True, validate=validate.Length(min=7))

def validate_form(email,password):
    # Validate the form student_schema=
    credentials_form = {
        'email': email,
        'password': password
    }

    credentials_schema = CredentialsSchema()
    validated_data = credentials_schema.load(credentials_form)



def login():
    st.title("Student Login")
    with st.form("Welcome!"):
        # Initialize error variable
        alert = st.empty()
        email = st.text_input("Email", placeholder="mail@mail.com")
        password = st.text_input("Password", type="password", placeholder="password")
        picture = st.camera_input(label="Student Face", help="Please provide a clear picture of yourself!")

        # Use st.form_submit_button's on_click event handler
        if st.form_submit_button('Login', use_container_width=True, type="primary"):
            # Call the validation function and catch any validation error
            try:
                validate_form(email, password)

                payload = {
                    'email': email,
                    'password': password
                }
                # Create a files dictionary with the captured image
                files = {
                    'image': ('image.jpg', picture.getvalue(), 'image/jpeg')
                }

                response = requests.get('http://127.0.0.1:5000/api/auth', data=payload, files=files)
                alert.info(response.content)
                if response.status_code == 200:
                    switch_page("exam")



            except ValidationError as error:
                alert.error(error)

    if st.button(label="Register",key="green-button",use_container_width=True):
        switch_page("registration")


if __name__ == "__main__":
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", 'Take Exam', 'Register'],
                               icons=['house', 'play-circle', "house-add"], menu_icon="cast", default_index=1,
                               key="navbar")


        if selected == "Home":
            switch_page("home")
        elif selected == "Register":
            switch_page("registration")


        st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

    login()