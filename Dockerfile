FROM python:3.11-slim

WORKDIR /app

COPY . /app/burnwindow_env

RUN pip install --no-cache-dir openai pydantic pyyaml

ENV PYTHONPATH=/app
CMD ["python", "-m", "burnwindow_env.inference"]
