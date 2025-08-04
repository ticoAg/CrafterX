#!/bin/bash
# 使用 gunicorn + uvicorn 启动 CrafterX FastAPI 服务

APP_MODULE="crafterx.server.app:app"
HOST="0.0.0.0"
PORT="8000"
WORKERS=4

exec gunicorn \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "$HOST:$PORT" \
    "$APP_MODULE"
