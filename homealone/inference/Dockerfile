# Base image
FROM python:3.9-slim

# Setup working directory
WORKDIR /app

# Install system dependencies needed for building wheels
RUN apt-get update && apt-get install -y build-essential python3-dev \
    # Clean up apt cache to keep image small
    && rm -rf /var/lib/apt/lists/*
    
# Copy requirements.txt and install dependencies
COPY requirements.txt .  

RUN pip install -r requirements.txt

# Copy the inference script to the working directory
COPY load_balancer.py ./load_balancer.py
COPY server.py ./server.py
COPY telemetry.py ./telemetry.py

# Shared volume for model
VOLUME ["/app/models", "/app/db"]

CMD ["python", "server.py"]

