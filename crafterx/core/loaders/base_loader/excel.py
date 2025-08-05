from typing import List

import pandas as pd

from ...Logger import logger
from ..base import BaseLoader, Document


class ExcelLoader(BaseLoader):
    """Excel文件加载器"""

    def load(self, file_path: str) -> List[Document]:
        logger.info(f"[ExcelLoader] 开始加载文件: {file_path}")
        if not self.validate_file(file_path):
            logger.error(f"[ExcelLoader] 文件未找到: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        excel_file = pd.ExcelFile(file_path)
        logger.info(f"[ExcelLoader] 发现 {len(excel_file.sheet_names)} 个sheet: {excel_file.sheet_names}")
        documents = []

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            content = df.to_string()
            metadata = {
                "source": file_path,
                "sheet_name": sheet_name,
                "row_count": len(df),
                "column_count": len(df.columns),
            }
            documents.append(Document(page_content=content, metadata=metadata))
            logger.debug(f"[ExcelLoader] sheet: {sheet_name}, 行数: {len(df)}, 列数: {len(df.columns)}")

        logger.info(f"[ExcelLoader] 加载完成，共 {len(documents)} 个文档")
        return documents
