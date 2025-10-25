# Base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    sshpass \
    openssh-client \
    procps \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/

# Setup working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./data_processing/

RUN mkdir -p /root/.ssh && \
    ssh-keyscan -H 128.2.220.241 >> /root/.ssh/known_hosts 2>/dev/null || true

VOLUME ["/app/db", "/app/data_queues"]

CMD ["python", "/app/data_processing/process.py"]