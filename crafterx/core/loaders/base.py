from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class Document:
    """文档类，用于存储加载后的文档内容"""

    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class BaseLoader(ABC):
    """文档加载器的基类"""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """加载文档并返回Document对象列表"""
        pass

    def validate_file(self, file_path: str) -> bool:
        """验证文件是否存在且可访问"""
        path = Path(file_path)
        return path.exists() and path.is_file()
