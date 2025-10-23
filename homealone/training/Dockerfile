# Base image
FROM python:3.9-slim

# Setup working directory
WORKDIR /app


# Install system dependencies needed for building wheels
RUN apt-get update && apt-get install -y build-essential python3-dev \
    # Clean up apt cache to keep image small
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .  

RUN pip install -r requirements.txt

COPY train.py ./train.py

# Store model in a shared volume
VOLUME ["/app/models", "/app/db"]


CMD ["python", "train.py"]

