FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir openai pydantic pyyaml

ENV PYTHONPATH=/app
CMD ["python", "inference.py"]
