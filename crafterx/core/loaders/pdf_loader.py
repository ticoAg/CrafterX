from typing import List

from langchain_community.document_loaders import PyMuPDFLoader

from ..Logger import logger
from ._base_base import BaseLoader, Document


class PDFLoader(BaseLoader):
    """PDF文件加载器"""

    def load(self, file_path: str) -> List[Document]:
        logger.info(f"[PDFLoader] 开始加载文件: {file_path}")
        if not self.validate_file(file_path):
            logger.error(f"[PDFLoader] 文件未找到: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"[PDFLoader] 加载完成，共 {len(documents)} 个文档")

        result = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        logger.debug(
            f"[PDFLoader] 文档内容示例: {result[0].page_content[:100] if result else '无内容'}"
        )
        return result
