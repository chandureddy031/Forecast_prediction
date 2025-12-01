FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt . 

RUN pip install -r requirements.txt

COPY .  .

RUN mkdir -p logs data/raw data/processed artifacts templates

EXPOSE 8000

CMD ["python", "main.py"]