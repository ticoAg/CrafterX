from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from crafterx.core.Logger import logger

from ..api.doc_loader_api import doc_loader_router


@asynccontextmanager
async def lifespan(app):
    logger.info("服务启动")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="CrafterX Document Parser API",
    description="文档解析服务API",
    version="0.1.0",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api", doc_loader_router)


if __name__ == "__main__":
    import uvicorn

    logger.info("启动服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
