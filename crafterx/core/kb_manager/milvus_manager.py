from typing import Dict, List, Optional, Union, cast

import numpy as np
from loguru import logger
from pymilvus import MilvusClient
from pymilvus.orm.schema import CollectionSchema

from ..Logger import logger
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
        self.uri = f"http://{host}:{port}" if not user or not password else f"http://{user}:{password}@{host}:{port}"
        self.client = MilvusClient(uri=self.uri)
        logger.info(f"MilvusManager initialized for URI: {host}:{port}. Connection will be established on first use.")

    async def create_collection(
        self,
        collection_name: str,
        dim: int,
        description: str = "",
        timeout: Optional[float] = None,
    ) -> None:
        """
        创建新的集合

        Args:
            collection_name: 集合名称
            dim: 向量维度
            primary_field: 主键字段名称
            vector_field: 向量字段名称
            description: 集合描述
            timeout: 超时时间(秒)
        """
        try:
            if self.client.has_collection(collection_name=collection_name):
                logger.warning(f"Collection {collection_name} already exists")
                return

            # 使用定义的schema创建集合
            schema: CollectionSchema = get_doc_meta_schema()

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="AUTOINDEX",
                metric_type="IP",
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                description=description,
                timeout=timeout,
                dimension=dim,
            )

            logger.info(f"Successfully created collection {collection_name} with custom schema")
        except Exception as e:
            logger.exception(f"Failed to create collection {collection_name}:\n {e}")
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
            return self.client.has_collection(collection_name=collection_name)
        except Exception as e:
            logger.exception(f"Failed to check collection existence: {e}")
            raise

    async def drop_collection(self, collection_name: str) -> None:
        """
        删除集合

        Args:
            collection_name: 要删除的集合名称
        """
        try:
            if not self.client.has_collection(collection_name=collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return

            self.client.drop_collection(collection_name=collection_name)
            logger.info(f"Successfully dropped collection {collection_name}")

        except Exception as e:
            logger.exception(f"Failed to drop collection {collection_name}: \n{e}")
            raise

    async def list_collections(self) -> List[str]:
        """
        列出所有集合

        Returns:
            List[str]: 集合名称列表
        """
        try:
            collections = self.client.list_collections()
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            raise

    async def insert(
        self,
        collection_name: str,
        data: List[Dict],
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        向集合中插入数据

        Args:
            collection_name: 集合名称
            data: 要插入的数据列表
            timeout: 超时时间（秒）

        Returns:
            Dict: 插入结果
        """
        try:
            result = self.client.insert(collection_name=collection_name, data=data, timeout=timeout)

            logger.info(f"Successfully inserted {len(data)} records into collection {collection_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to insert data: {str(e)}")
            raise

    async def delete(
        self,
        collection_name: str,
        expr: str,
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        从集合中删除数据

        Args:
            collection_name: 集合名称
            expr: 删除表达式，例如 "id in [1,2,3]"
            timeout: 超时时间（秒）

        Returns:
            Dict: 删除结果
        """
        try:
            result = self.client.delete(collection_name=collection_name, filter=expr, timeout=timeout)

            logger.info(f"Successfully deleted records with expression: {expr}")
            return result

        except Exception as e:
            logger.error(f"Failed to delete records: {str(e)}")
            raise

    async def search(
        self,
        collection_name: str,
        data: Union[List[List[float]], np.ndarray],
        limit: int = 10,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        search_params: Optional[Dict] = None,
    ) -> List[List[Dict]]:
        """
        向量相似度搜索

        Args:
            collection_name: 集合名称
            data: 查询向量数据
            limit: 返回最相似的前k个结果
            expr: 过滤表达式
            output_fields: 返回的字段列表
            timeout: 超时时间（秒）
            search_params: 搜索参数

        Returns:
            List[List[Dict]]: 搜索结果列表
        """
        try:
            # 转换为列表格式
            if isinstance(data, np.ndarray):
                processed_data: List[List[float]] = data.tolist()
            else:
                processed_data = list(data)

            # 确保data是二维列表
            if len(processed_data) > 0 and not isinstance(processed_data[0], list):
                processed_data = [cast(List[float], processed_data)]

            # 执行搜索
            search_result = self.client.search(
                collection_name=collection_name,
                data=processed_data,
                limit=limit,
                filter=expr or "",
                output_fields=output_fields,
                timeout=timeout,
                search_params=search_params or {"metric_type": "L2", "params": {"nprobe": 10}},
            )

            logger.info(f"Successfully performed search in collection {collection_name}")
            return search_result

        except Exception as e:
            logger.error(f"Failed to perform search: {str(e)}")
            raise

    async def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict]:
        """
        查询数据

        Args:
            collection_name: 集合名称
            expr: 查询表达式
            output_fields: 返回的字段列表
            timeout: 超时时间（秒）

        Returns:
            List[Dict]: 查询结果列表
        """
        try:
            result = self.client.query(
                collection_name=collection_name, filter=expr, output_fields=output_fields, timeout=timeout
            )

            logger.info(f"Successfully queried records with expression: {expr}")
            return result

        except Exception as e:
            logger.error(f"Failed to query records: {str(e)}")
            raise
