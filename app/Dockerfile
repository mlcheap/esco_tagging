# syntax=docker/dockerfile:1
FROM python:3.9.9-bullseye
WORKDIR /home
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt
CMD ["flask", "run"]
