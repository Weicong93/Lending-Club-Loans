FROM python:3.11-slim

WORKDIR /flask-app

COPY ./requirements.txt ./requirements.txt
COPY ./src ./src
COPY ./model ./model

ENV PYTHONPATH /flask-app/src

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["gunicorn", "--bind=0.0.0.0:80", "src.app:app"]

