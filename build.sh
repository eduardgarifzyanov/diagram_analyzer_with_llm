#!/bin/bash
cd diagram-analyzer

set -e

echo "Stopping old containers..."
docker compose down -v || true

echo "Pulling latest images..."
docker compose pull

echo "Building and starting services..."
docker compose up --build
