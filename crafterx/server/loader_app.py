import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile

from ..core.loaders import Document, DocumentLoaderFactory
from ..core.Logger import logger

loader_api_router = FastAPI()


@loader_api_router.post("/parse")
async def parse_document(file: UploadFile = File(...), enhanced: bool = False) -> Dict[str, Any]:
    """
    解析上传的文档

    Args:
        file: 上传的文件
        enhanced: 是否使用增强模式（针对PPT）

    Returns:
        解析后的文档内容
    """
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
            loader = DocumentLoaderFactory.get_loader(str(tmp_path), enhanced=enhanced)
            documents: List[Document] = loader.load(str(tmp_path))

            result = {
                "filename": file.filename,
                "content": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            }

            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            # 清理临时文件
            tmp_path.unlink()

    except Exception as e:
        logger.error(f"解析文档时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"解析文档时发生错误: {str(e)}")
    finally:
        file.file.close()
