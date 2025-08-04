from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger
from pymilvus import (
    Collection,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .doc_schema import get_doc_meta_schema


class AsyncMilvusManager:
    """
    Milvus 向量数据库异步管理类
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        connection_alias: str = "default",
    ):
        """
        初始化 Milvus 管理器

        Args:
            host: Milvus 服务器主机名
            port: Milvus 服务器端口
            user: 用户名（如果启用了认证）
            password: 密码（如果启用了认证）
            connection_alias: 连接别名
        """
        self.connection_alias = connection_alias
        self.connection_params = {
            "host": host,
            "port": port,
        }
        if user and password:
            self.connection_params.update({"user": user, "password": password})

        # 注意:__init__不能是异步的,所以我们只在这里配置连接参数
        # 实际连接会在第一次使用时建立

    async def _connect(self) -> None:
        """建立与 Milvus 服务器的连接"""
        try:
            connections.connect(
                alias=self.connection_alias,
                **self.connection_params,
                _async=True,  # 使用异步连接
            )
            logger.info(
                f"Successfully connected to Milvus server at {self.connection_params['host']}:{self.connection_params['port']}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    async def create_collection(
        self,
        collection_name: str,
        dim: int,
        primary_field: str = "id",
        vector_field: str = "embedding",
        description: str = "",
        auto_id: bool = False,
        timeout: Optional[float] = None,
    ) -> Collection:
        """
        创建新的集合

        Args:
            collection_name: 集合名称
            dim: 向量维度
            primary_field: 主键字段名称
            vector_field: 向量字段名称
            description: 集合描述
            auto_id: 是否自动生成ID
            timeout: 超时时间(秒)

        Returns:
            Collection: 创建的集合对象
        """
        try:
            if await self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} already exists")
                return Collection(collection_name)

            # 获取文档元数据schema
            doc_schema = get_doc_meta_schema()

            # 添加向量字段
            doc_schema.fields.append(
                FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim)
            )

            schema = doc_schema

            collection = Collection(
                name=collection_name, schema=schema, using=self.connection_alias
            )

            logger.info(f"Successfully created collection {collection_name}")
            return collection

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            raise

    async def has_collection(self, collection_name: str) -> bool:
        """
        检查集合是否存在

        Args:
            collection_name: 集合名称

        Returns:
            bool: 如果集合存在返回True，否则返回False
        """
        try:
            return utility.has_collection(collection_name, using=self.connection_alias)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {str(e)}")
            raise

    async def drop_collection(self, collection_name: str) -> None:
        """
        删除集合

        Args:
            collection_name: 要删除的集合名称
        """
        try:
            if not await self.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return

            utility.drop_collection(collection_name, using=self.connection_alias)
            logger.info(f"Successfully dropped collection {collection_name}")

        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {str(e)}")
            raise

    async def list_collections(self) -> List[str]:
        """
        列出所有集合

        Returns:
            List[str]: 集合名称列表
        """
        try:
            collections = utility.list_collections(using=self.connection_alias)
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            raise

    async def insert(
        self,
        collection_name: str,
        vectors: Union[List[List[float]], np.ndarray],
        ids: Optional[List[int]] = None,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> List[int]:
        """
        向集合中插入向量

        Args:
            collection_name: 集合名称
            vectors: 要插入的向量列表或numpy数组
            ids: 向量ID列表（如果auto_id=False则必须提供）
            partition_name: 分区名称（可选）
            timeout: 超时时间（秒）

        Returns:
            List[int]: 插入的向量ID列表
        """
        try:
            collection = Collection(collection_name)
            if not collection.is_empty:
                collection.load()

            # 准备数据
            entities = []
            if ids is not None:
                entities.append(ids)

            if isinstance(vectors, np.ndarray):
                vectors = vectors.tolist()
            entities.append(vectors)

            # 执行插入
            insert_result = collection.insert(
                entities, partition_name=partition_name, timeout=timeout
            )

            logger.info(
                f"Successfully inserted {len(vectors)} vectors into collection {collection_name}"
            )
            return insert_result.primary_keys

        except Exception as e:
            logger.error(f"Failed to insert vectors: {str(e)}")
            raise

    async def delete(
        self,
        collection_name: str,
        expr: str,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        从集合中删除向量

        Args:
            collection_name: 集合名称
            expr: 删除表达式，例如 "id in [1,2,3]"
            partition_name: 分区名称（可选）
            timeout: 超时时间（秒）
        """
        try:
            collection = Collection(collection_name)
            if not collection.is_empty:
                collection.load()

            collection.delete(expr, partition_name=partition_name, timeout=timeout)
            logger.info(f"Successfully deleted vectors with expression: {expr}")

        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            raise

    async def search(
        self,
        collection_name: str,
        vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        向量相似度搜索

        Args:
            collection_name: 集合名称
            vector: 查询向量
            top_k: 返回最相似的前k个结果
            expr: 过滤表达式
            output_fields: 返回的字段列表
            partition_names: 搜索的分区列表
            timeout: 超时时间（秒）
            **kwargs: 其他搜索参数

        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 确保已经连接
            if not connections.get_connection(self.connection_alias):
                await self._connect()

            collection = Collection(collection_name)
            collection.load()

            # 准备搜索参数
            search_params = {
                "metric_type": "L2",  # 默认使用L2距离
                "params": {"nprobe": 10},  # 默认nprobe值
            }
            search_params.update(kwargs)

            # 如果输入是numpy数组，转换为列表
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            # 确保vector是二维的
            if not isinstance(vector[0], (list, np.ndarray)):
                vector = [vector]

            # 执行异步搜索
            search_future = collection.search(
                data=vector,
                anns_field="embedding",  # 假设向量字段名为"embedding"
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                _async=True,  # 启用异步模式
            )

            # 等待搜索结果
            search_result = search_future.result()

            # 格式化返回结果
            formatted_results = []
            # 获取第一个查询的结果 (因为我们只搜索了一个向量)
            first_hits = search_result[0]  # Hits对象

            # 遍历命中结果
            for id, distance in zip(first_hits.ids, first_hits.distances):
                result = {
                    "id": id,
                    "distance": distance,
                }
                # 如果指定了output_fields并且结果中有entity
                if output_fields and hasattr(first_hits[0], "entity"):
                    result.update(first_hits[0].entity)
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to perform search: {str(e)}")
            raise

    def close(self) -> None:
        """关闭与Milvus的连接"""
        try:
            connections.disconnect(self.connection_alias)
            logger.info("Successfully disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {str(e)}")
            raise
