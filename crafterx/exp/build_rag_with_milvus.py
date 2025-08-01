# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-31 17:40:08
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pymilvus import MilvusClient
from transformers import AutoTokenizer

sys.path.append(Path(__file__).parents[2].as_posix())

from crafterx.core.splitters import MarkdownHeaderTextSplitter

load_dotenv()

_tokenizer = None
_last_used_time = None
_TIMEOUT_SECONDS = 300  # 5分钟超时
_lock = Lock()  # 线程安全
_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def qwen_length_func(text: str) -> int:
    global _tokenizer, _last_used_time

    current_time = time.time()

    with _lock:
        # 检查是否需要重新加载（超时或尚未加载）
        if _tokenizer is None or (
            _last_used_time and (current_time - _last_used_time) > _TIMEOUT_SECONDS
        ):
            print(f"Loading tokenizer from {_MODEL_NAME}...")
            _tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME, trust_remote_code=True
            )
            _last_used_time = current_time
        else:
            _last_used_time = current_time  # 更新最后使用时间

    # 执行编码
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def analyze_text_lengths(lengths: list[int]) -> None:
    """
    分析文本块长度分布并输出统计信息

    Args:
        lengths: 文本块长度列表
    """
    if not lengths:
        logger.warning("没有文本块需要统计")
        return

    # 按区间统计长度分布
    length_distribution = {}
    for length in lengths:
        if length < 128:
            bucket = "<128"
        elif length < 256:
            bucket = "128-256"
        elif length < 512:
            bucket = "256-512"
        elif length < 768:
            bucket = "512-768"
        elif length < 1024:
            bucket = "768-1024"
        else:
            bucket = ">=1024"

        length_distribution[bucket] = length_distribution.get(bucket, 0) + 1

    # 输出长度分布统计结果
    logger.info("文本块长度分布统计:")
    for bucket in ["<128", "128-256", "256-512", "512-768", "768-1024", ">=1024"]:
        count = length_distribution.get(bucket, 0)
        logger.info(f"  {bucket:8}: {count} 个文本块")

    # 显示最大、最小和平均长度
    logger.debug(f"最大文本块长度: {max(lengths)}")
    logger.debug(f"最小文本块长度: {min(lengths)}")
    logger.debug(f"平均文本块长度: {sum(lengths) / len(lengths):.2f}")


def load_docs() -> list[dict]:
    # ! wget https://github.com/milvus-io/milvus-docs/releases/download/v2.4.6-preview/milvus_docs_2.4.x_en.zip
    # ! unzip -q milvus_docs_2.4.x_en.zip -d milvus_docs
    doc_dir = Path("data")
    text_obj = []

    markdown_splitter = MarkdownHeaderTextSplitter(
        chunk_size=768, chunk_overlap=256, length_function=qwen_length_func
    )

    # 收集所有文本块长度用于后续统计
    all_lengths = []

    for file_path in doc_dir.rglob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                markdown_document = file.read()
        except UnicodeDecodeError:
            # 如果UTF-8解码失败，尝试使用其他编码
            with open(file_path, "r", encoding="latin-1") as file:
                markdown_document = file.read()

        text_lines = markdown_splitter.split_text(markdown_document)
        for idx, line in enumerate(text_lines):
            # 记录长度用于统计
            length = qwen_length_func(line.content)
            all_lengths.append(length)

            text_obj.append(
                {
                    "text": line,
                    "doc_remote_url": file_path.as_posix(),
                    "id": file_path.name,
                    "doc_idx": idx,
                }
            )

    # 处理完所有文档后进行统一统计
    analyze_text_lengths(all_lengths)

    return text_obj


async def aemb_text(text) -> list[float]:
    result = await aopenai_client.embeddings.create(
        input=text, model="netease-youdao/bce-embedding-base_v1"
    )
    return result.data[0].embedding


def create_collection(collection_name: str, embedding_dim: int):
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )


async def ainsert_data(text_obj, collection_name: str):
    from tqdm import tqdm

    # 使用异步并发处理向量化
    tasks = [aemb_text(line["text"].page_content) for line in text_obj]

    # 使用 tqdm 显示进度
    embeddings = []
    for task in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Creating embeddings"
    ):
        embedding = await task
        embeddings.append(embedding)

    data = [
        {"id": i, "vector": emb, "text": line}
        for i, (emb, line) in enumerate(zip(embeddings, text_obj))
    ]

    milvus_client.insert(collection_name=collection_name, data=data)


async def retrieve(query_text, collection_name: str):
    import json

    query_vec = await aemb_text(query_text)

    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            query_vec
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    print(json.dumps(retrieved_lines_with_distances, indent=4))
    return retrieved_lines_with_distances


async def achat_with_rag_response(question: str, knowledges: list[tuple[str, float]]):
    context = "\n".join([line_with_distance[0] for line_with_distance in knowledges])

    SYSTEM_PROMPT = "你是一个AI助手。你能够根据提供的上下文段落找到问题的答案"

    USER_PROMPT = (
        "请根据以下用<context>标签包裹的上下文内容，回答<question>标签中的问题：\n"
        "<context>\n"
        f"{context}\n"
        f"</context>\n"
        "<question>\n"
        f"{question}\n"
        "</question>"
    )
    response = await aopenai_client.chat.completions.create(
        model="qwen3-30b-a3b-instruct-2507",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print("Answer: \n", response.choices[0].message.content)


async def main():
    test_embedding = await aemb_text("This is a test")
    embedding_dim = len(test_embedding)
    text_lines = load_docs()
    create_collection(collection_name, embedding_dim)
    await ainsert_data(text_lines, collection_name)

    question = "数据怎么在miluvs中存储的?"
    knowledges = await retrieve(question, collection_name)
    await achat_with_rag_response(question, knowledges)


if __name__ == "__main__":
    aopenai_client = AsyncOpenAI()
    milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
    collection_name = "my_rag_collection"
    asyncio.run(main())
