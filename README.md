# CrafterX

CrafterX 是一个用于构建检索增强生成（RAG）系统的工具包，基于 Milvus 向量数据库。它提供了文档加载、解析、分块和向量存储等功能，帮助用户快速构建高性能的 RAG 应用。

## 功能特性

-   支持多种文档格式解析（Markdown、PDF、Word、PowerPoint、Excel）
-   基于 Milvus 的向量检索能力
-   文档内容分块处理
-   易于扩展的模块化设计

## 支持的文档格式

-   Markdown (.md, .markdown)
-   PDF (.pdf)
-   Word (.doc, .docx)
-   PowerPoint (.pptx)
-   Excel (.xls, .xlsx)

## 使用文档加载器

```python
from crafterx.core.loaders._loader_factory import DocumentLoaderFactory

# 创建加载器工厂
factory = DocumentLoaderFactory()

# 获取文档加载器 (默认模式)
loader = factory.get_loader("path/to/document.docx")

# 获取文档加载器 (增强模式，仅对PPT有效)
enhanced_loader = factory.get_loader("path/to/presentation.pptx", enhanced=True)

# 加载文档
documents = loader.load()

# 遍历加载结果
for doc in documents:
    print(f"内容: {doc.content}")
    print(f"元数据: {doc.metadata}")
```

### 增强型PPT加载器

CrafterX提供了增强型PPT加载器，可以保留PPT中的文本格式、图片和表格等富媒体信息。
使用方法简单，只需在`get_loader`时设置`enhanced=True`参数。

## 构建 RAG 系统

使用 `crafterx/exp/build_rag_with_milvus.py` 脚本可以构建基于 Milvus 的 RAG 系统。

### 前提条件

1. 确保已安装 Docker 和 Docker Compose
2. 启动 Milvus 服务（详见下方 Docker 部署部分）

### 运行脚本

```bash
# 运行 RAG 构建脚本
uv run python crafterx/exp/build_rag_with_milvus.py --data_dir path/to/your/documents
```

## Docker 部署

### 启动 Milvus 服务

```bash
# 进入 Milvus Docker 目录
cd docker/milvus

# 启动 Milvus 服务
docker-compose up -d
```

### 访问 Milvus 管理界面

启动成功后，可以通过以下地址访问 Milvus Attu 管理界面：
http://localhost:9092

## Usage

### 环境准备

```bash
# 同步依赖环境
uv sync
```

### 启动 API 服务

```bash
# 启动单进程 API 服务
uv run poe serve_single

# 或启动多进程 API 服务
uv run poe serve
```

### 访问 API 文档

服务启动后，可以通过以下地址访问 API 文档：
http://localhost:8000/docs

### 运行主程序

```bash
# 运行主程序
uv run python main.py
```

## 项目结构

```
CrafterX/
├── .gitignore
├── .python-version
├── Dockerfile
├── README.md
├── config/              # 配置文件
├── crafterx/
│   ├── __init__.py
│   ├── api/             # API接口定义
│   ├── core/            # 核心功能
│   │   ├── loaders/     # 文档加载器
│   │   ├── splitters/   # 文档分块器
│   │   └── kb_manager/  # 知识库管理器
│   ├── exp/             # 实验性代码
│   ├── server/          # 服务器代码
│   └── web/             # Web界面代码
├── data/                # 数据目录
├── docker/              # Docker配置
│   └── milvus/          # Milvus Docker配置
├── docs/                # 文档
├── examples/            # 示例代码
├── main.py              # 主程序
├── pyproject.toml       # 项目依赖
├── scripts/             # 脚本文件
└── uv.lock              # 依赖锁文件
```
