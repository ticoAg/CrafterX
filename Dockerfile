# 基于 Python 3.12 的官方镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app


# 安装 uv 并同步依赖
RUN pip install --upgrade pip \
    && pip install uv \
    && uv sync --system

# 暴露端口（假设服务运行在8000端口，可根据实际情况修改）
EXPOSE 8000

# 启动命令，使用已配置的脚本
CMD ["/bin/bash", "scripts/run_server.sh"]
