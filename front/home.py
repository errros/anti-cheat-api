import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_option_menu import option_menu
from PIL import Image

home_image = Image.open('home_exam.jpg')
st.image(image=home_image)
st.button(key="login",label="Take your Exam!",use_container_width=True,type="primary")


with st.sidebar:


    selected = option_menu("Main Menu", ["Home", 'Take Exam', 'Register'],
                           icons=['house', 'play-circle', "house-add"], menu_icon="cast", default_index=0, key="navbar")



    if selected == "Take Exam":
        switch_page("login")
    elif selected == "Register":
        switch_page("registration")

    st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)
