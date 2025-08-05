from pathlib import Path
from typing import Dict, Type

from ..Logger import logger
from .base import BaseLoader, Document
from .base_loader.excel import ExcelLoader
from .base_loader.md import MarkdownLoader
from .base_loader.pdf import PDFLoader
from .base_loader.ppt import PPTLoader
from .base_loader.text import TextLoader
from .base_loader.word import WordLoader
from .experimental import EnhancedPPTLoader


class DocumentLoaderFactory:
    """文档加载器工厂类"""

    _loaders: dict[str, Type[BaseLoader]] = {
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

    _enhanced_loaders: dict[str, Type[BaseLoader]] = {
        ".ppt": EnhancedPPTLoader,
        ".pptx": EnhancedPPTLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str, enhanced: bool = False, **kwargs) -> BaseLoader:
        """
        根据文件扩展名获取对应的加载器实例

        Args:
            file_path: 文件路径
            enhanced: 是否使用增强型加载器（仅对PPT有效）
            **kwargs: 传递给加载器的额外参数

        Returns:
            BaseLoader: 对应的加载器实例

        Raises:
            ValueError: 当文件类型不支持时抛出
        """
        ext = Path(file_path).suffix.lower()
        logger.debug(f"获取文件加载器: {file_path} (扩展名: {ext}, 增强模式: {enhanced})")

        # 优先使用增强型加载器
        if enhanced and ext in cls._enhanced_loaders:
            loader_class = cls._enhanced_loaders[ext]
            logger.info(f"使用增强型 {loader_class.__name__} 处理文件: {file_path}")
            return loader_class(**kwargs)

        if ext not in cls._loaders:
            logger.error(f"不支持的文件类型: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")

        loader_class = cls._loaders[ext]
        logger.info(f"使用 {loader_class.__name__} 处理文件: {file_path}")
        return loader_class()
