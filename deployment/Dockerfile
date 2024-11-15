FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ["deployment/stroke_predictor.pkl", "deployment/app.py", "./"]
COPY utils/ /app/utils/

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8989", "app:app" ]
