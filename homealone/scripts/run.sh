#!/bin/bash
set -e

# Capture git commit
GIT_COMMIT=$(git rev-parse HEAD)
echo "Git commit captured: ${GIT_COMMIT:0:8}"

# Export for docker compose
export GIT_COMMIT

# Build with git commit
echo ""
echo "Building Docker images..."
docker compose build

# Run containers
echo ""
echo "Starting containers..."
docker compose up -d "$@"
