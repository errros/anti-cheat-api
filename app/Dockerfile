FROM python:3.7.16
WORKDIR /src/
COPY requirements.txt .
RUN /bin/bash -c "pip install -r requirements.txt"

COPY settings.py .
COPY service.py .
COPY schemas.py .
COPY run.py .
COPY routes.py .
COPY models.py .
COPY crud.py .

RUN mkdir "static"
RUN mkdir "static/db"
RUN mkdir "static/embeddings"
RUN mkdir "static/ml"
RUN mkdir "static/tmp"

EXPOSE 5000:5000

ENTRYPOINT ["python","run.py"]


