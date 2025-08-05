# 文档加载器 (Loaders)

文档加载器模块提供了一套完整的解决方案，用于从各种文件格式中提取文本内容并将其转换为统一的[Document](base.py)对象。这些加载器支持多种常见文件格式，包括文本文件、办公文档、PDF等。

## 功能特点

- 支持多种文件格式：`.txt`、`.pdf`、`.doc`、`.docx`、`.xls`、`.xlsx`、`.ppt`、`.pptx`、`.md`
- 统一的接口设计，所有加载器都继承自[BaseLoader](base.py)
- 工厂模式实现，通过[DocumentLoaderFactory](factory.py)自动选择合适的加载器
- 增强型PPT加载器，支持视觉内容分析和QA对生成
- 客户端模式支持，可选择本地或远程处理模式

## 加载器类型

### 基础加载器

1. [TextLoader](base_loader/text.py) - 纯文本文件加载器
2. [PDFLoader](base_loader/pdf.py) - PDF文件加载器，基于PyMuPDF
3. [WordLoader](base_loader/word.py) - Word文档加载器（支持doc和docx）
4. [ExcelLoader](base_loader/excel.py) - Excel文件加载器，每个工作表生成一个Document
5. [PPTLoader](base_loader/ppt.py) - PowerPoint文件加载器
6. [MarkdownLoader](base_loader/md.py) - Markdown文件加载器

### 增强型加载器

1. [EnhancedPPTLoader](experimental/ppt_loader_enhanced.py) - 增强型PPT加载器，具有以下特性：
   - 使用视觉大模型分析幻灯片中的图片和图表
   - 自动生成高质量的问答对
   - 提供同步和异步两种处理模式
   - 适用于构建检索增强生成（RAG）系统

## 使用方法

### 基本用法

```python
from crafterx.core.loaders import DocumentLoaderFactory

# 获取适合文件的加载器
loader = DocumentLoaderFactory.get_loader("document.pdf")

# 加载文档
documents = loader.load("document.pdf")

# 处理结果
for doc in documents:
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
```

### 使用增强模式（仅适用于PPT）

```python
from crafterx.core.loaders import DocumentLoaderFactory

# 获取增强型PPT加载器
loader = DocumentLoaderFactory.get_loader("presentation.pptx", enhanced=True)

# 加载并分析PPT
documents = loader.load("presentation.pptx")
```

### 使用客户端模式

```python
from crafterx.core.loaders import LoaderClient

# 本地模式
client = LoaderClient(mode="local")
documents = client.load("document.pdf")

# 远程模式（需要配置远程服务） (暂未支持)
client = LoaderClient(mode="remote", base_url="http://localhost:8001")
documents = client.load("document.pdf")
```

## 核心类说明

### Document 类

表示加载后的文档内容，包含以下属性：
- `page_content`: 文档内容文本
- `metadata`: 文档元数据（如源文件、页码等）

### BaseLoader 类

所有加载器的基类，定义了加载文档的接口：
- `load(file_path)`: 加载文档并返回Document对象列表
- `validate_file(file_path)`: 验证文件是否存在且可访问

### DocumentLoaderFactory 类

文档加载器工厂类，根据文件扩展名自动选择合适的加载器：
- `get_loader(file_path, enhanced=False)`: 获取对应文件类型的加载器实例

## 支持的文件格式

| 扩展名 | 加载器 | 增强模式支持 |
|-------|--------|-------------|
| .txt | TextLoader | 否 |
| .pdf | PDFLoader | 否 |
| .doc | WordLoader | 否 |
| .docx | WordLoader | 否 |
| .xls | ExcelLoader | 否 |
| .xlsx | ExcelLoader | 否 |
| .ppt | PPTLoader/EnhancedPPTLoader | 是 |
| .pptx | PPTLoader/EnhancedPPTLoader | 是 |
| .md | MarkdownLoader | 否 |

## 扩展加载器

要添加新的文件格式支持，需要：

1. 在[base_loader](base_loader/)目录下创建新的加载器类
2. 继承[BaseLoader](base.py)类并实现load方法
3. 在[factory.py](factory.py)中的`_loaders`字典添加文件扩展名与加载器的映射关系