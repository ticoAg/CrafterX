# text_processor/advanced_chunking.py
import re
from typing import Callable, List

from .workflow import TextChunk


class AdvancedTextSplitter:
    """高级文本分割器，支持按句子、段落等语义单元分割"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "!", "?", "。", "！"]

    def split_text(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """
        按语义单元分割文本

        Args:
            text: 输入文本
            metadata: 元数据

        Returns:
            文本块列表
        """
        if not text:
            return []

        metadata = metadata or {}
        chunks = []
        chunk_index = 0

        # 按照分隔符分割文本
        parts = self._split_by_separators(text)
        current_chunk = ""
        current_length = 0

        for part in parts:
            part_length = len(part)

            # 如果添加这个部分会超过块大小，则保存当前块并开始新块
            if current_length + part_length > self.chunk_size and current_chunk:
                # 保存当前块
                chunk_metadata = metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_size": len(current_chunk),
                    }
                )

                chunks.append(TextChunk(content=current_chunk.strip(), metadata=chunk_metadata, chunk_index=chunk_index))

                chunk_index += 1

                # 开始新块，考虑重叠
                if self.chunk_overlap > 0:
                    # 找到合适的重叠部分
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + part
                    current_length = len(overlap_text) + part_length
                else:
                    current_chunk = part
                    current_length = part_length
            else:
                # 添加到当前块
                current_chunk += part
                current_length += part_length

        # 处理最后一个块
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_size": len(current_chunk),
                }
            )

            chunks.append(TextChunk(content=current_chunk.strip(), metadata=chunk_metadata, chunk_index=chunk_index))

        return chunks

    def _split_by_separators(self, text: str) -> List[str]:
        """
        按分隔符分割文本

        Args:
            text: 输入文本

        Returns:
            分割后的文本部分列表
        """
        # 从最长的分隔符开始分割
        separators = sorted(self.separators, key=len, reverse=True)

        parts = [text]
        for separator in separators:
            new_parts = []
            for part in parts:
                if separator in part:
                    splits = part.split(separator)
                    # 重新添加分隔符（除了最后一个分割）
                    for i, split in enumerate(splits[:-1]):
                        new_parts.append(split + separator)
                    new_parts.append(splits[-1])
                else:
                    new_parts.append(part)
            parts = new_parts

        # 过滤掉空的部分
        return [part for part in parts if part]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        获取用于重叠的文本

        Args:
            text: 文本
            overlap_size: 重叠大小

        Returns:
            重叠文本
        """
        if len(text) <= overlap_size:
            return text

        # 从文本末尾获取指定长度的文本
        overlap_text = text[-overlap_size:]

        # 尝试在合理的位置截断（如句子边界）
        for separator in [". ", "! ", "? ", "。", "！", "？", "\n"]:
            last_separator_pos = overlap_text.rfind(separator)
            if last_separator_pos > 0:
                return overlap_text[last_separator_pos + len(separator) :]

        return overlap_text
