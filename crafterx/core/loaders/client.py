import requests
from typing import List, Optional, Union
from pathlib import Path
import os

from . import DocumentLoaderFactory, Document
# from .loader_api import loader_api_router


class LoaderClient:
    """
    Loader客户端，支持本地和远程两种调用模式
    
    使用方法:
    1. 本地模式:
       client = LoaderClient(mode="local")
       documents = client.load("path/to/document.pdf")
       
    2. 远程模式:
       client = LoaderClient(mode="remote", base_url="http://localhost:8001")
       documents = client.load("path/to/document.pdf")
    """
    
    def __init__(self, mode: str = "local", base_url: Optional[str] = None):
        """
        初始化Loader客户端
        
        Args:
            mode: 调用模式，"local" 或 "remote"
            base_url: 远程服务的基础URL，仅在mode为"remote"时需要
        """
        self.mode = mode
        self.base_url = base_url
        
        if mode == "remote" and not base_url:
            raise ValueError("远程模式需要提供base_url参数")
    
    def load(self, file_path: str, enhanced: bool = False) -> List[Document]:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            enhanced: 是否使用增强模式（针对PPT）
            
        Returns:
            Document对象列表
        """
        if self.mode == "local":
            return self._load_local(file_path, enhanced)
        elif self.mode == "remote":
            return self._load_remote(file_path, enhanced)
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def _load_local(self, file_path: str, enhanced: bool = False) -> List[Document]:
        """
        本地加载文档
        
        Args:
            file_path: 文件路径
            enhanced: 是否使用增强模式
            
        Returns:
            Document对象列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        loader = DocumentLoaderFactory.get_loader(file_path, enhanced=enhanced)
        return loader.load(file_path)
    
    
    def _load_remote(self, file_path: str, enhanced: bool = False) -> List[Document]:
        """
        远程加载文档 (暂时不启用)
        
        Args:
            file_path: 文件路径
            enhanced: 是否使用增强模式
            
        Returns:
            Document对象列表
        """
        # TODO: 这部分代码暂时不想启用，以后考虑再要不要写
        # 暂时返回空列表表示未启用
        return []
        
        # 以下为原始实现代码，暂不启用
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        file_path_obj = Path(file_path)
        
        with open(file_path, "rb") as f:
            files = {
                "file": (file_path_obj.name, f, self._get_mime_type(file_path_obj.suffix))
            }
            data = {
                "enhanced": enhanced
            }
            
            response = requests.post(
                f"{self.base_url}{loader_api_router.prefix}/parse",
                files=files,
                data=data
            )
            
        if response.status_code != 200:
            raise Exception(f"远程服务调用失败: {response.text}")
            
        result = response.json()
        documents = []
        for item in result["content"]:
            doc = Document(
                page_content=item["page_content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
            
        return documents
        """
    def _get_mime_type(self, extension: str) -> str:
        """
        根据文件扩展名获取MIME类型
        
        Args:
            extension: 文件扩展名
            
        Returns:
            MIME类型字符串
        """
        mime_types = {
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".md": "text/markdown",
        }
        return mime_types.get(extension.lower(), "application/octet-stream")