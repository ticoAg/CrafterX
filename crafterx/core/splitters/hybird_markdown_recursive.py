import copy
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import yaml


@dataclass
class Chunk:
    """用于存储文本片段及相关元数据的类。"""

    content: str = ""
    metadata: dict = field(default_factory=dict)

    def to_markdown(self, return_all: bool = False) -> str:
        """将块转换为 Markdown 格式。

        Args:
            return_all: 如果为 True，则在内容前包含 YAML 格式的元数据。

        Returns:
            Markdown 格式的字符串。
        """
        md_string = ""
        if return_all and self.metadata:
            metadata_yaml = yaml.dump(self.metadata, allow_unicode=True, sort_keys=False)
            md_string += f"---\n{metadata_yaml}---\n\n"
        md_string += self.content
        return md_string


class LineType(TypedDict):
    """行类型，使用类型字典定义。"""

    metadata: Dict[str, str]
    content: str


class HeaderType(TypedDict):
    """标题类型，使用类型字典定义。"""

    level: int
    name: str
    data: str


class MarkdownHeaderTextSplitter:
    """基于指定的标题分割 Markdown 文件，并可选地根据 chunk_size 进一步细分。"""

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ],
        strip_headers: bool = False,
        chunk_size: Optional[int] = None,
        length_function: Callable[[str], int] = len,
        chunk_overlap: int = 0,
        # 优化了默认分隔符顺序和内容
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = True,  # 默认改为 True，以更好地利用默认分隔符
    ):
        """创建一个新的 MarkdownHeaderTextSplitter。

        Args:
            headers_to_split_on: 用于分割的标题级别和名称元组列表。
            strip_headers: 是否从块内容中移除标题行。
            chunk_size: 块的最大非代码内容长度。如果设置，将进一步分割超出的块。
            length_function: 用于计算文本长度的函数。
            chunk_overlap: 块之间的重叠长度。
            separators: 用于分割的分隔符列表，优先级从高到低。
            is_separator_regex: 是否将分隔符视为正则表达式。
        """
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数或 None。")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 必须是非负整数。")
        if chunk_size is not None and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size。")

        self.headers_to_split_on = sorted(headers_to_split_on, key=lambda split: len(split[0]), reverse=True)
        self.strip_headers = strip_headers
        self._chunk_size = chunk_size
        self._length_function = length_function
        self._chunk_overlap = chunk_overlap
        # 优化了默认分隔符，使其更通用，并默认启用正则
        if separators is None:
            self._separators = [
                "\n\n",  # 首先尝试按段落分割
                "\n",  # 其次按行
                r"(?<=[。！？;\s])",  # 然后按中英文句子（使用正向后行断言保留分隔符）
                r"(?<=[，,\s])",  # 再次按逗号
                " ",  # 最后按空格
            ]
        else:
            self._separators = separators
        self._is_separator_regex = is_separator_regex

    def _calculate_length_excluding_code(self, text: str) -> int:
        """计算文本长度，不包括代码块内容。"""
        # 正则表达式查找 ```...``` 或 ~~~...~~~ 代码块
        # 使用 re.sub 将代码块替换为空字符串，然后计算剩余文本的长度
        non_code_text = re.sub(
            r"(?:```|~~~).*?\n(?:.*?\n)*?(?:```|~~~)\n?",
            "",
            text,
            flags=re.DOTALL | re.MULTILINE,
        )
        return self._length_function(non_code_text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并小的分割块，并处理重叠。"""
        if not splits:
            return []

        merged_splits = []
        current_split = ""
        for split in splits:
            # 如果当前块为空，或者加上新块（和分隔符）不会超长
            if (
                not current_split
                or self._calculate_length_excluding_code(current_split + separator + split) <= self._chunk_size
            ):
                current_split += separator + split
            else:
                # 当前块已经足够大，将其存入结果，并开始一个新块
                merged_splits.append(current_split.lstrip(separator))
                current_split = split

        merged_splits.append(current_split.lstrip(separator))
        return merged_splits

    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        """
        使用分隔符列表递归地分割文本。
        这是解决超长问题的核心。
        """
        final_chunks = []

        # 获取文本的非代码部分长度
        text_len = self._calculate_length_excluding_code(text)
        if text_len <= self._chunk_size and not text.isspace():  # 如果文本本身就不超长，直接返回
            return [text]
        elif not separators:  # 如果没有更多分隔符可用，但文本仍超长，则直接返回（这是基本情况）
            return [text]

        # 使用第一个分隔符进行分割
        current_separator = separators[0]
        next_separators = separators[1:]

        # 对正则表达式分隔符进行特殊处理
        if self._is_separator_regex:
            # 使用 re.split，它能处理更复杂的分隔符模式
            try:
                splits = re.split(f"({current_separator})", text)
                splits = [splits[i] + (splits[i + 1] if i + 1 < len(splits) else "") for i in range(0, len(splits), 2)]
                splits = [s for s in splits if s]  # 移除空字符串
            except re.error:
                # 如果不是有效的正则表达式，则按字面分割
                splits = text.split(current_separator)
        else:
            splits = text.split(current_separator)

        good_splits = []
        for s in splits:
            # 检查分割后的每个子块
            if self._calculate_length_excluding_code(s) <= self._chunk_size:
                good_splits.append(s)
            else:
                # 这个子块仍然太长，需要用下一级的分隔符继续分割
                if next_separators:
                    # 递归调用
                    recursive_splits = self._split_text_recursively(s, next_separators)
                    good_splits.extend(recursive_splits)
                else:
                    # 没有下一级分隔符了，只能保留这个超长的块
                    good_splits.append(s)

        # 合并小的块以更好地利用空间
        merged_splits = self._merge_splits(good_splits, current_separator if not self._is_separator_regex else "")

        return merged_splits

    def _aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Chunk]:
        """将具有共同元数据的行合并成块。"""
        aggregated_chunks: List[LineType] = []

        for line in lines:
            if aggregated_chunks and aggregated_chunks[-1]["metadata"] == line["metadata"]:
                aggregated_chunks[-1]["content"] += "\n" + line["content"]
            else:
                aggregated_chunks.append(copy.deepcopy(line))

        return [Chunk(content=chunk_data["content"], metadata=chunk_data["metadata"]) for chunk_data in aggregated_chunks]

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """基于标题分割 Markdown 文本，并根据 chunk_size 进一步细分。"""
        base_metadata = metadata or {}
        lines = text.split("\n")
        lines_with_metadata: List[LineType] = []
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        header_stack: List[HeaderType] = []

        in_code_block = False
        opening_fence = ""

        # --- 步骤 1: 按标题和代码块初步分割 ---
        for line in lines:
            stripped_line = line.strip()
            # --- 代码块处理 ---
            is_code_fence = False
            if not in_code_block:
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block, opening_fence, is_code_fence = True, "```", True
                elif stripped_line.startswith("~~~") and stripped_line.count("~~~") == 1:
                    in_code_block, opening_fence, is_code_fence = True, "~~~", True
            elif in_code_block and stripped_line.startswith(opening_fence):
                in_code_block, opening_fence, is_code_fence = False, "", True

            if in_code_block or is_code_fence:
                current_content.append(line)
                continue

            # --- 标题处理 ---
            found_header = False
            for sep, name in self.headers_to_split_on:
                if stripped_line.startswith(sep) and (len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "):
                    found_header = True
                    header_level = sep.count("#")
                    header_data = stripped_line[len(sep) :].strip()

                    if current_content:
                        lines_with_metadata.append({"content": "\n".join(current_content), "metadata": current_metadata.copy()})
                        current_content = []

                    while header_stack and header_stack[-1]["level"] >= header_level:
                        header_stack.pop()
                    header_stack.append({"level": header_level, "name": name, "data": header_data})
                    current_metadata = {h["name"]: h["data"] for h in header_stack}

                    if not self.strip_headers:
                        current_content.append(line)
                    break

            if not found_header:
                current_content.append(line)

        if current_content:
            lines_with_metadata.append({"content": "\n".join(current_content), "metadata": current_metadata.copy()})

        # --- 步骤 2: 合并具有相同元数据的行，形成初始大块 ---
        initial_chunks = self._aggregate_lines_to_chunks(lines_with_metadata)

        # --- 步骤 3: 细分超出 chunk_size 的块 ---
        if self._chunk_size is None:
            return initial_chunks

        final_chunks = []
        for chunk in initial_chunks:
            # 合并基础元数据和块元数据
            final_metadata = base_metadata.copy()
            final_metadata.update(chunk.metadata)

            # 检查非代码内容长度
            non_code_len = self._calculate_length_excluding_code(chunk.content)

            if non_code_len > self._chunk_size:
                # 如果块超长，则使用新的递归分割方法
                sub_splits = self._split_text_recursively(chunk.content, self._separators)
                for split_content in sub_splits:
                    # 为每个子块创建新的 Chunk 对象
                    if not split_content.isspace():  # 忽略只包含空白的块
                        final_chunks.append(Chunk(content=split_content, metadata=final_metadata.copy()))
            else:
                # 如果未超长，直接添加
                if not chunk.content.isspace():
                    final_chunks.append(Chunk(content=chunk.content, metadata=final_metadata))

        return final_chunks
