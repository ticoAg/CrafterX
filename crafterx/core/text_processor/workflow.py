# text_processor/workflow.py
from dataclasses import dataclass
from typing import List

from ...core.loaders import DocumentLoaderFactory
from ...core.loaders._base_base import Document


@dataclass
class TextChunk:
    """文本块数据类"""

    content: str
    metadata: dict
    chunk_index: int


class TextProcessingWorkflow:
    """通用文本处理工作流"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, enhanced_ppt: bool = False):
        """
        初始化文本处理工作流

        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            enhanced_ppt: 是否使用增强型PPT加载器
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enhanced_ppt = enhanced_ppt

    def process_file(self, file_path: str) -> List[TextChunk]:
        """
        处理文件：加载 -> 提取文本 -> 切块

        Args:
            file_path: 文件路径

        Returns:
            文本块列表
        """
        # 1. 加载文档
        loader = DocumentLoaderFactory.get_loader(file_path, enhanced=self.enhanced_ppt)
        documents = loader.load(file_path)

        # 2. 提取文本并切块
        chunks = self._chunk_documents(documents)

        return chunks

    def _chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        将文档切块

        Args:
            documents: 文档列表

        Returns:
            文本块列表
        """
        chunks = []
        for doc in documents:
            doc_chunks = self._chunk_text(doc.page_content, doc.metadata)
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_text(self, text: str, metadata: dict) -> List[TextChunk]:
        """
        对文本进行切块处理

        Args:
            text: 输入文本
            metadata: 元数据

        Returns:
            文本块列表
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_content = text[start:end]

            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_content),
                    "text_start_position": start,
                    "text_end_position": min(end, len(text)),
                }
            )

            chunks.append(TextChunk(content=chunk_content, metadata=chunk_metadata, chunk_index=chunk_index))

            chunk_index += 1
            start = end - self.chunk_overlap

            # 如果剩余文本太短，直接作为最后一个块
            if start + self.chunk_size >= len(text):
                remaining_text = text[start:]
                if remaining_text.strip():  # 只有非空文本才创建块
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_index": chunk_index,
                            "chunk_size": len(remaining_text),
                            "text_start_position": start,
                            "text_end_position": len(text),
                        }
                    )

                    chunks.append(TextChunk(content=remaining_text, metadata=chunk_metadata, chunk_index=chunk_index))
                break

        return chunks
