#!/bin/bash
# 使用 gunicorn + uvicorn 启动 CrafterX FastAPI 服务
# 获取脚本的二级父目录
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=$PROJECT_DIR

APP_MODULE="crafterx.server.app:app"
HOST="0.0.0.0"
PORT="8000"

# 动态计算 workers 数量: (CPU核心数 * 2) + 1，最大不超过 32
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
WORKERS=$((2 * CPU_CORES + 1))
if [ $WORKERS -gt 32 ]; then
    WORKERS=32
fi

gunicorn \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "$HOST:$PORT" \
    "$APP_MODULE"