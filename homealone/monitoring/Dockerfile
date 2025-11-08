# Base image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all monitoring modules
COPY *.py ./monitoring/

# Create necessary directories
RUN mkdir -p /app/reports/health /app/reports/quality /app/reports/drift /app/reports/alerts /app/reports/hourly /app/reports/daily /app/reports/weekly

# Expose Prometheus metrics port
EXPOSE 8000

# Shared volumes for database and reports
VOLUME ["/app/db", "/app/reports"]

# Default command runs unified monitoring
CMD ["python", "/app/monitoring/unified_monitor.py"]