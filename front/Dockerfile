FROM python:3.8.17
WORKDIR /src/
COPY requirements.txt .
RUN /bin/bash -c "pip install -r requirements.txt"
COPY home.py .
COPY home_exam.jpg .
RUN mkdir "pages"
COPY pages/ pages/
EXPOSE 8501:8501
ENTRYPOINT ["streamlit","run","home.py"]
