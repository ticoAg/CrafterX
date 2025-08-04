import streamlit as st
from fastapi.testclient import TestClient

from crafterx.server.app import app


def main():
    st.set_page_config(page_title="CrafterX 文档解析器", page_icon="📄", layout="wide")

    st.title("CrafterX 文档解析器")
    st.write("支持解析多种文档格式，包括: PDF, Word, Excel, PPT, Markdown, TXT等")

    # 文件上传区域
    uploaded_file = st.file_uploader(
        "选择要解析的文档",
        type=["txt", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "md"],
    )

    if uploaded_file is not None:
        # 显示文件信息
        file_details = {
            "文件名": uploaded_file.name,
            "文件类型": uploaded_file.type,
            "文件大小": f"{uploaded_file.size / 1024:.2f} KB",
        }
        st.write("### 文件信息")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")

        # 创建解析按钮
        if st.button("开始解析"):
            try:
                with st.spinner("正在解析文档..."):
                    # 将文件通过API发送到后端解析
                    client = TestClient(app)
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    response = client.post("/api/parse", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.write("### 解析结果")
                        content_list = result.get("content", [])
                        if content_list:
                            for i, page_content in enumerate(content_list, 1):
                                with st.expander(f"第 {i} 页/部分"):
                                    st.markdown(page_content)
                        else:
                            st.info("未解析到内容。")
                    else:
                        try:
                            error_detail = response.json().get("detail", "")
                        except Exception:
                            error_detail = response.text
                        st.error(f"解析失败: {error_detail}")
            except Exception as e:
                st.error(f"处理文件时发生错误: {str(e)}")


if __name__ == "__main__":
    main()
