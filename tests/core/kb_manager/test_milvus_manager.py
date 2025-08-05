# tests/core/kb_manager/test_milvus_manager.py
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from pymilvus import MilvusClient

from crafterx.core.kb_manager.milvus_manager import AsyncMilvusManager


class TestAsyncMilvusManager(unittest.TestCase):
    def setUp(self):
        """在每个测试之前设置环境"""
        self.host = "localhost"
        self.port = "19530"
        self.collection_name = "test_collection"
        self.dim = 128

        with patch("crafterx.core.kb_manager.milvus_manager.MilvusClient") as mock_client:
            self.milvus_manager = AsyncMilvusManager(host=self.host, port=self.port)
            self.mock_client = mock_client.return_value
            self.milvus_manager.client = self.mock_client

    def test_init(self):
        """测试初始化"""
        with patch("crafterx.core.kb_manager.milvus_manager.MilvusClient") as mock_client:
            manager = AsyncMilvusManager(host="test_host", port="19540")
            mock_client.assert_called_once_with(uri="http://test_host:19540")

    @patch("crafterx.core.kb_manager.milvus_manager.get_doc_meta_schema")
    def test_create_collection_success(self, mock_get_schema):
        """测试成功创建集合"""
        mock_schema = MagicMock()
        mock_get_schema.return_value = mock_schema
        self.mock_client.has_collection.return_value = False

        mock_index_params = MagicMock()
        self.mock_client.prepare_index_params.return_value = mock_index_params

        async def run_test():
            await self.milvus_manager.create_collection(
                collection_name=self.collection_name, dim=self.dim, description="Test collection"
            )

            self.mock_client.has_collection.assert_called_once_with(collection_name=self.collection_name)
            mock_get_schema.assert_called_once()
            self.mock_client.prepare_index_params.assert_called_once()
            mock_index_params.add_index.assert_called_once_with(
                field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX", metric_type="IP"
            )
            self.mock_client.create_collection.assert_called_once()

        asyncio.run(run_test())

    def test_create_collection_already_exists(self):
        """测试创建已存在的集合"""
        self.mock_client.has_collection.return_value = True

        async def run_test():
            await self.milvus_manager.create_collection(collection_name=self.collection_name, dim=self.dim)

            self.mock_client.has_collection.assert_called_once_with(collection_name=self.collection_name)
            self.mock_client.create_collection.assert_not_called()

        asyncio.run(run_test())

    def test_create_collection_exception(self):
        """测试创建集合时发生异常"""
        self.mock_client.has_collection.return_value = False
        self.mock_client.create_collection.side_effect = Exception("Connection failed")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.create_collection(collection_name=self.collection_name, dim=self.dim)

        asyncio.run(run_test())

    def test_has_collection_success(self):
        """测试检查集合是否存在"""
        self.mock_client.has_collection.return_value = True

        async def run_test():
            result = await self.milvus_manager.has_collection(self.collection_name)
            self.assertTrue(result)
            self.mock_client.has_collection.assert_called_once_with(collection_name=self.collection_name)

        asyncio.run(run_test())

    def test_has_collection_exception(self):
        """测试检查集合时发生异常"""
        self.mock_client.has_collection.side_effect = Exception("Connection error")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.has_collection(self.collection_name)

        asyncio.run(run_test())

    def test_drop_collection_success(self):
        """测试成功删除集合"""
        self.mock_client.has_collection.return_value = True

        async def run_test():
            await self.milvus_manager.drop_collection(self.collection_name)
            self.mock_client.has_collection.assert_called_once_with(collection_name=self.collection_name)
            self.mock_client.drop_collection.assert_called_once_with(collection_name=self.collection_name)

        asyncio.run(run_test())

    def test_drop_collection_not_exists(self):
        """测试删除不存在的集合"""
        self.mock_client.has_collection.return_value = False

        async def run_test():
            await self.milvus_manager.drop_collection(self.collection_name)
            self.mock_client.has_collection.assert_called_once_with(collection_name=self.collection_name)
            self.mock_client.drop_collection.assert_not_called()

        asyncio.run(run_test())

    def test_drop_collection_exception(self):
        """测试删除集合时发生异常"""
        self.mock_client.has_collection.return_value = True
        self.mock_client.drop_collection.side_effect = Exception("Delete failed")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.drop_collection(self.collection_name)

        asyncio.run(run_test())

    def test_list_collections_success(self):
        """测试列出所有集合"""
        expected_collections = ["collection1", "collection2", "collection3"]
        self.mock_client.list_collections.return_value = expected_collections

        async def run_test():
            result = await self.milvus_manager.list_collections()
            self.assertEqual(result, expected_collections)
            self.mock_client.list_collections.assert_called_once()

        asyncio.run(run_test())

    def test_list_collections_exception(self):
        """测试列出集合时发生异常"""
        self.mock_client.list_collections.side_effect = Exception("List failed")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.list_collections()

        asyncio.run(run_test())

    def test_insert_success(self):
        """测试成功插入数据"""
        test_data = [{"id": 1, "vector": [0.1] * self.dim}, {"id": 2, "vector": [0.2] * self.dim}]
        expected_result = {"insert_count": 2}
        self.mock_client.insert.return_value = expected_result

        async def run_test():
            result = await self.milvus_manager.insert(collection_name=self.collection_name, data=test_data)
            self.assertEqual(result, expected_result)
            self.mock_client.insert.assert_called_once_with(collection_name=self.collection_name, data=test_data, timeout=None)

        asyncio.run(run_test())

    def test_insert_exception(self):
        """测试插入数据时发生异常"""
        test_data = [{"id": 1, "vector": [0.1] * self.dim}]
        self.mock_client.insert.side_effect = Exception("Insert failed")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.insert(collection_name=self.collection_name, data=test_data)

        asyncio.run(run_test())

    def test_delete_success(self):
        """测试成功删除数据"""
        expr = "id in [1, 2, 3]"
        expected_result = {"delete_count": 3}
        self.mock_client.delete.return_value = expected_result

        async def run_test():
            result = await self.milvus_manager.delete(collection_name=self.collection_name, expr=expr)
            self.assertEqual(result, expected_result)
            self.mock_client.delete.assert_called_once_with(collection_name=self.collection_name, filter=expr, timeout=None)

        asyncio.run(run_test())

    def test_delete_exception(self):
        """测试删除数据时发生异常"""
        expr = "id in [1, 2, 3]"
        self.mock_client.delete.side_effect = Exception("Delete failed")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.delete(collection_name=self.collection_name, expr=expr)

        asyncio.run(run_test())

    def test_search_with_list_data(self):
        """测试使用列表数据进行搜索"""
        query_vectors = [[0.1] * self.dim, [0.2] * self.dim]
        expected_result = [[{"id": 1, "distance": 0.9}]]
        self.mock_client.search.return_value = expected_result

        async def run_test():
            result = await self.milvus_manager.search(collection_name=self.collection_name, data=query_vectors, limit=5)
            self.assertEqual(result, expected_result)
            self.mock_client.search.assert_called_once()

        asyncio.run(run_test())

    def test_search_with_numpy_data(self):
        """测试使用numpy数组进行搜索"""
        query_vectors = np.array([[0.1] * self.dim])
        expected_result = [[{"id": 1, "distance": 0.9}]]
        self.mock_client.search.return_value = expected_result

        async def run_test():
            result = await self.milvus_manager.search(collection_name=self.collection_name, data=query_vectors, limit=5)
            self.assertEqual(result, expected_result)
            self.mock_client.search.assert_called_once()

        asyncio.run(run_test())

    def test_search_exception(self):
        """测试搜索时发生异常"""
        query_vectors = [[0.1] * self.dim]
        self.mock_client.search.side_effect = Exception("Search failed")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.search(collection_name=self.collection_name, data=query_vectors)

        asyncio.run(run_test())

    def test_query_success(self):
        """测试成功查询数据"""
        expr = "id > 0"
        output_fields = ["id", "vector"]
        expected_result = [{"id": 1, "vector": [0.1] * self.dim}]
        self.mock_client.query.return_value = expected_result

        async def run_test():
            result = await self.milvus_manager.query(
                collection_name=self.collection_name, expr=expr, output_fields=output_fields
            )
            self.assertEqual(result, expected_result)
            self.mock_client.query.assert_called_once_with(
                collection_name=self.collection_name, filter=expr, output_fields=output_fields, timeout=None
            )

        asyncio.run(run_test())

    def test_query_exception(self):
        """测试查询数据时发生异常"""
        expr = "id > 0"
        self.mock_client.query.side_effect = Exception("Query failed")

        async def run_test():
            with self.assertRaises(Exception):
                await self.milvus_manager.query(collection_name=self.collection_name, expr=expr)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
