#!/bin/bash
# 使用 uvicorn 单进程启动 CrafterX FastAPI 服务

APP_MODULE="crafterx.server.app:app"
HOST="127.0.0.1"
PORT="8000"

uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
