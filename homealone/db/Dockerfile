# Base image
FROM python:3.9-slim

# Setup working directory
WORKDIR /app

COPY home_alone.db /app/db/home_alone.db

# Shared volume for model
VOLUME ["/app/models", "/app/db"]


