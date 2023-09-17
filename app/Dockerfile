FROM python:3.7.16
WORKDIR /src/
COPY requirements.txt .


# Install system dependencies (including the missing libgl1-mesa-glx)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN /bin/bash -c "pip install --no-cache-dir -r requirements.txt"


COPY settings.py .
COPY service.py .
COPY schemas.py .
COPY run.py .
COPY routes.py .
COPY models.py .
COPY crud.py .

ENV QT_X11_NO_MITSHM=1


RUN mkdir "static"

EXPOSE 5000:5000

ENTRYPOINT ["python","run.py"]


