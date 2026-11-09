#!/usr/bin/env bash
# Helper script to manage AckBot Docker simulation container on macOS

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${DIR}/docker-compose.yml"

case "$1" in
  build)
    echo "==> Building AckBot ROS 2 Jazzy & Gazebo container..."
    docker compose -f "${COMPOSE_FILE}" build
    ;;
  up)
    echo "==> Starting AckBot simulation container..."
    docker compose -f "${COMPOSE_FILE}" up -d
    echo "Container running!"
    echo "  • Web GUI (noVNC)       : http://localhost:6080"
    echo "  • Foxglove Studio Bridge: ws://localhost:8765"
    ;;
  bash|shell)
    echo "==> Opening bash shell inside AckBot container..."
    docker compose -f "${COMPOSE_FILE}" exec ackbot-sim bash
    ;;
  stop|down)
    echo "==> Stopping AckBot container..."
    docker compose -f "${COMPOSE_FILE}" down
    ;;
  logs)
    docker compose -f "${COMPOSE_FILE}" logs -f
    ;;
  *)
    echo "Usage: $0 {build|up|bash|stop|logs}"
    exit 1
    ;;
esac
