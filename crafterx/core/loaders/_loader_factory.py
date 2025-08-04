from pathlib import Path
from typing import Dict, Type

from ..Logger import logger
from ._base_base import BaseLoader
from .excel_loader import ExcelLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .ppt_loader import PPTLoader
from .text_loader import TextLoader
from .word_loader import WordLoader


class DocumentLoaderFactory:
    """文档加载器工厂类"""

    _loaders: Dict[str, Type[BaseLoader]] = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".doc": WordLoader,
        ".docx": WordLoader,
        ".xls": ExcelLoader,
        ".xlsx": ExcelLoader,
        ".ppt": PPTLoader,
        ".pptx": PPTLoader,
        ".md": MarkdownLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str) -> BaseLoader:
        """
        根据文件扩展名获取对应的加载器实例

        Args:
            file_path: 文件路径

        Returns:
            BaseLoader: 对应的加载器实例

        Raises:
            ValueError: 当文件类型不支持时抛出
        """
        ext = Path(file_path).suffix.lower()
        logger.debug(f"获取文件加载器: {file_path} (扩展名: {ext})")

        if ext not in cls._loaders:
            logger.error(f"不支持的文件类型: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")

        loader_class = cls._loaders[ext]
        logger.info(f"使用 {loader_class.__name__} 处理文件: {file_path}")
        return loader_class()
