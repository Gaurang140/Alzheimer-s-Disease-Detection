FROM python:3.8.15-slim-buster


RUN apt update -y && apt install awscli -y   
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt
# command line
CMD ["python3", "app.py"]