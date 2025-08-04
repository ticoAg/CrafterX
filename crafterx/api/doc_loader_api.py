import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from crafterx.core.loaders import Document
from crafterx.core.Logger import logger

from ..core.loaders import (
    DocumentLoaderFactory
)

doc_loader_router = APIRouter()


@doc_loader_router.post("/parse", summary="解析文档")
async def parse_document(file: UploadFile = File(...)):
    """
    上传并解析文档

    Args:
        file: 要解析的文档文件

    Returns:
        解析后的文档内容
    """
    # 检查文件扩展名是否支持
    if not file.filename:
        logger.error("文件名为空")
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_ext = Path(file.filename).suffix.lower()
    logger.info(f"接收到文件上传请求: {file.filename}")
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)

        # 获取对应的加载器并解析文档
        try:
            loader = DocumentLoaderFactory.get_loader(str(tmp_path))
            documents: List[Document] = loader.load(str(tmp_path))
            return {"content": [doc.page_content for doc in documents]}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            # 清理临时文件
            tmp_path.unlink()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析文档时发生错误: {str(e)}")
    finally:
        file.file.close()
