from pymilvus import CollectionSchema, DataType, FieldSchema

"""
{
    "activate": "是否在用, if false, 检索排除"
    "u_id": "资源上传用户id",
    "u_department": "用户部门",
    "upload_time": "上传时间", 
    "update_time": "知识更新时间, 创建时=upload_time", 
    "doc_id": "文档id",
    "doc_idx": "当前片段在所属doc的位置",
    "doc_rel_ids": ["相关的文档id"]
    "doc_remote_storage_platform": "文档存储平台 minIO & oss & etc",
    "doc_remote_url": "文档资源地址"
}
"""


def get_doc_meta_schema() -> CollectionSchema:
    """获取文档元数据的Schema定义
    Returns:
        CollectionSchema: 文档元数据的Milvus集合schema
    """
    fields = [
        # 文档ID作为主键
        FieldSchema(
            name="doc_id",
            dtype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
            description="文档唯一标识符",
        ),
        # 文档是否激活
        FieldSchema(
            name="activate",
            dtype=DataType.BOOL,
            description="文档是否可用于检索",
        ),
        # 用户相关信息
        FieldSchema(
            name="u_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="上传用户ID",
        ),
        FieldSchema(
            name="u_department",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="用户所属部门",
        ),
        # 时间信息
        FieldSchema(
            name="upload_time",
            dtype=DataType.INT64,
            description="文档上传时间戳",
        ),
        FieldSchema(
            name="update_time",
            dtype=DataType.INT64,
            description="文档更新时间戳",
        ),
        # 文档位置信息
        FieldSchema(
            name="doc_idx",
            dtype=DataType.INT64,
            description="文档片段在原文档中的位置索引",
        ),
        # 文档关联信息
        FieldSchema(
            name="doc_rel_ids",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length_per_element=128,
            max_capacity=50,
            description="相关文档ID列表",
        ),
        # 文档存储信息
        FieldSchema(
            name="doc_remote_storage_platform",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="文档存储平台(minIO/oss等)",
        ),
        FieldSchema(
            name="doc_remote_url",
            dtype=DataType.VARCHAR,
            max_length=1024,
            description="文档远程存储URL",
        ),
    ]

    return CollectionSchema(
        fields=fields,
        description="文档元数据schema",
        enable_dynamic_field=False,
    )
