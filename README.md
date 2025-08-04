# CrafterX

CrafterX 是一个用于构建检索增强生成（RAG）系统的工具包，基于 Milvus 向量数据库。

## 功能特性

- 支持多种文档格式解析（Markdown、PDF、Word、PowerPoint、Excel）
- 基于 Milvus 的向量检索能力
- 文档内容分块处理
- 易于扩展的模块化设计

## 安装依赖

```bash
pip install -e .
```

或者安装所有可选依赖以支持所有文档格式：

```bash
pip install -e .[all]
```

## 支持的文档格式

- Markdown (.md, .markdown)
- PDF (.pdf)
- Word (.doc, .docx)
- PowerPoint (.pptx)
- Excel (.xls, .xlsx)

## 使用文档解析器

```python
from crafterx.core.parser import DocumentParserFactory

# 创建解析器工厂
factory = DocumentParserFactory()

# 解析任何支持的文档格式
documents = factory.parse_document("path/to/document.docx")

# 遍历解析结果
for doc in documents:
    print(f"内容: {doc.content}")
    print(f"元数据: {doc.metadata}")
```

## 构建RAG系统

使用 `crafterx/exp/build_rag_with_milvus.py` 脚本构建基于 Milvus 的 RAG 系统。



# Usage

```bash
# 环境同步
uv sync
# 启动api服务
uv run poe serve_single

uv run poe serve
```