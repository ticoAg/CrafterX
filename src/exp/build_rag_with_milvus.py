# -*- encoding: utf-8 -*-
"""
@Time    :   2025-07-31 17:40:08
@desc    :
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()


def load_docs() -> list[str]:
    doc_dir = Path("data")
    text_lines = []

    for file_path in doc_dir.rglob("en/faq/*.md"):
        with open(file_path, "r") as file:
            file_text = file.read()

        text_lines += file_text.split("# ")
    return text_lines


def emb_text(text):
    return (
        openai_client.embeddings.create(
            input=text, model="netease-youdao/bce-embedding-base_v1"
        )
        .data[0]
        .embedding
    )


def create_collection(collection_name: str):
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )


def insert_data(text_lines, collection_name: str):
    from tqdm import tqdm

    data = []

    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": emb_text(line), "text": line})

    milvus_client.insert(collection_name=collection_name, data=data)


def retrieve(query_text, collection_name: str):
    import json

    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            emb_text(query_text)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    print(json.dumps(retrieved_lines_with_distances, indent=4))


if __name__ == "__main__":
    openai_client = OpenAI()
    milvus_client = MilvusClient()
    test_embedding = emb_text("This is a test")
    embedding_dim = len(test_embedding)

    collection_name = "my_rag_collection"
    text_lines = load_docs()
    create_collection(collection_name)
    insert_data(text_lines, collection_name)

    question = "How is data stored in milvus?"
    retrieve(question, collection_name)
