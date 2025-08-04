import sys
from pathlib import Path

from loguru import logger as _logger


class LoggerManager:
    """日志管理器类"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not LoggerManager._initialized:
            self._setup_logger()
            LoggerManager._initialized = True

    def _setup_logger(self):
        """配置日志系统"""
        # 获取项目根目录
        self.project_root = Path(__file__).parents[2]
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # 日志格式
        self.log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # 移除默认的处理器
        _logger.remove()

        # 添加处理器
        self._add_handlers()

    def _add_handlers(self):
        """添加日志处理器"""
        # 控制台处理器
        self.add_console_handler("DEBUG")

        # 文件处理器
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in log_levels:
            self.add_file_handler(level)

    def add_console_handler(self, level: str = "INFO"):
        """添加控制台处理器"""
        return _logger.add(
            sys.stdout, format=self.log_format, level=level, enqueue=True
        )

    def add_file_handler(
        self, level: str, rotation: str = "00:00", retention: str = "30 days"
    ):
        """添加文件处理器"""
        return _logger.add(
            self.log_dir / f"{level.lower()}.log",
            format=self.log_format,
            level=level,
            rotation=rotation,
            retention=retention,
            enqueue=True,
        )

    @staticmethod
    def get_logger():
        """获取logger实例"""
        return _logger

    def set_global_level(self, level: str):
        """设置全局日志级别"""
        _logger.remove()
        self._add_handlers()
        _logger.level(level)

    @staticmethod
    def debug(message: str, *args, **kwargs):
        """记录调试日志"""
        _logger.debug(message, *args, **kwargs)

    @staticmethod
    def info(message: str, *args, **kwargs):
        """记录信息日志"""
        _logger.info(message, *args, **kwargs)

    @staticmethod
    def warning(message: str, *args, **kwargs):
        """记录警告日志"""
        _logger.warning(message, *args, **kwargs)

    @staticmethod
    def error(message: str, *args, **kwargs):
        """记录错误日志"""
        _logger.error(message, *args, **kwargs)

    @staticmethod
    def critical(message: str, *args, **kwargs):
        """记录严重错误日志"""
        _logger.critical(message, *args, **kwargs)


# 创建全局日志管理器实例
log_manager: LoggerManager = LoggerManager()

logger = log_manager.get_logger()

if __name__ == "__main__":
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
