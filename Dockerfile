FROM python:3.9.1-slim-buster

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

RUN apt update

COPY . .
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["flask", "run"]