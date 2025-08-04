#!/bin/bash

# 获取脚本的二级父目录
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=$PROJECT_DIR

streamlit run crafterx/web/loader_web.py --server.headless true
