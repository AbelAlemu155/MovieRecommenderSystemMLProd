# Base image
FROM python:3.9-slim

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Setup working directory
WORKDIR /app


# Install system dependencies needed for building wheels
RUN apt-get update && apt-get install -y build-essential python3-dev git \
    # Clean up apt cache to keep image small
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY train.py ./train.py
COPY config.py ./config.py
COPY data_provenance.py ./data_provenance.py
COPY model_provenance.py ./model_provenance.py

# Store model in a shared volume
VOLUME ["/app/models", "/app/db"]


CMD ["python", "train.py"]

