import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pymilvus import Collection

sys.path.append(Path(__file__).parents[3].as_posix())
from dotenv import load_dotenv

from crafterx.core.kb_manager.milvus_manager import AsyncMilvusManager

load_dotenv()


@pytest.mark.asyncio
class TestAsyncMilvusManager:
    """AsyncMilvusManager 类的单元测试"""

    def setup_method(self):
        """测试前设置"""
        self.host = os.environ.get("MILVUS_HOST", "localhost")
        self.port = os.environ.get("MILVUS_PORT", "19530")
        self.test_collection_name = f"test_collection_{random.randint(1000, 9999)}"
        self.dim = 128
        self.manager = AsyncMilvusManager(host=self.host, port=self.port, connection_alias="test_alias")

    async def teardown_method(self):
        """测试后清理"""
        # 删除测试集合
        if await self.manager.has_collection(self.test_collection_name):
            await self.manager.drop_collection(self.test_collection_name)

    async def test_connect(self):
        """测试连接Milvus服务器"""
        # 测试成功连接
        with patch("pymilvus.connections.connect") as mock_connect:
            mock_connect.return_value = True
            await self.manager._connect()
            mock_connect.assert_called_once_with(alias="test_alias", host=self.host, port=self.port, _async=True)

        # 测试连接失败
        with patch("pymilvus.connections.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            with pytest.raises(Exception, match="Connection failed"):
                await self.manager._connect()

    async def test_create_collection(self):
        """测试创建集合"""
        # 测试创建新集合
        collection = await self.manager.create_collection(collection_name=self.test_collection_name, dim=self.dim)
        assert isinstance(collection, Collection)
        assert collection.name == self.test_collection_name

        # 测试创建已存在的集合
        with patch("loguru.logger.warning") as mock_warning:
            collection = await self.manager.create_collection(collection_name=self.test_collection_name, dim=self.dim)
            mock_warning.assert_called_once_with(f"Collection {self.test_collection_name} already exists")

    async def test_has_collection(self):
        """测试检查集合是否存在"""
        # 测试不存在的集合
        assert not await self.manager.has_collection("non_existent_collection")

        # 测试存在的集合
        await self.manager.create_collection(collection_name=self.test_collection_name, dim=self.dim)
        assert await self.manager.has_collection(self.test_collection_name)

    async def test_drop_collection(self):
        """测试删除集合"""
        # 测试删除不存在的集合
        with patch("loguru.logger.warning") as mock_warning:
            await self.manager.drop_collection("non_existent_collection")
            mock_warning.assert_called_once_with("Collection non_existent_collection does not exist")

        # 测试删除存在的集合
        await self.manager.create_collection(collection_name=self.test_collection_name, dim=self.dim)
        assert await self.manager.has_collection(self.test_collection_name)
        await self.manager.drop_collection(self.test_collection_name)
        assert not await self.manager.has_collection(self.test_collection_name)

    async def test_list_collections(self):
        """测试列出所有集合"""
        # 先创建一个测试集合
        await self.manager.create_collection(collection_name=self.test_collection_name, dim=self.dim)

        # 测试列出集合
        collections = await self.manager.list_collections()
        assert self.test_collection_name in collections

    async def test_insert(self):
        """测试插入向量"""
        # 先创建集合
        await self.manager.create_collection(collection_name=self.test_collection_name, dim=self.dim)

        # 准备测试数据
        num_vectors = 10
        vectors = np.random.random((num_vectors, self.dim)).tolist()

        # 测试插入向量
        inserted_ids = await self.manager.insert(collection_name=self.test_collection_name, vectors=vectors)
        assert len(inserted_ids) == num_vectors

        # 测试插入带ID的向量
        vectors_with_ids = np.random.random((num_vectors, self.dim)).tolist()
        ids = [i + 1000 for i in range(num_vectors)]

        inserted_ids_with_ids = await self.manager.insert(
            collection_name=self.test_collection_name, vectors=vectors_with_ids, ids=ids
        )
        assert len(inserted_ids_with_ids) == num_vectors
        assert inserted_ids_with_ids == ids

    async def test_insert_with_numpy_array(self):
        """测试使用numpy数组插入向量"""
        # 先创建集合
        await self.manager.create_collection(collection_name=self.test_collection_name, dim=self.dim)

        # 准备测试数据
        num_vectors = 5
        vectors = np.random.random((num_vectors, self.dim))

        # 测试插入numpy数组
        inserted_ids = await self.manager.insert(collection_name=self.test_collection_name, vectors=vectors)
        assert len(inserted_ids) == num_vectors


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__])
