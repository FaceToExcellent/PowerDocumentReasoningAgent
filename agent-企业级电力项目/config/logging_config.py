"""结构化日志配置 — loguru + 标准 logging 桥接"""
import logging
import sys
from pathlib import Path

from loguru import logger as _loguru


# 标准 logging 处理器：将日志记录转接到 loguru
class _InterceptHandler(logging.Handler):
    """把标准 logging 的日志转接到 loguru"""

    # 转发单条日志记录到 loguru，保留日志级别与异常堆栈
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# 初始化 loguru（控制台 + 应用文件 + 审计日志），并接管标准 logging
def setup_logging(level: str = "INFO", log_dir: str = "./logs"):
    """初始化 loguru，同时接管标准 logging"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _loguru.remove()
    _loguru.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
               "<cyan>{module}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    _loguru.add(
        f"{log_dir}/app.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        encoding="utf-8",
    )
    _loguru.add(
        f"{log_dir}/audit.log",
        level="INFO",
        filter=lambda record: record["extra"].get("audit", False),
        encoding="utf-8",
        rotation="20 MB",
    )

    # 标准 logging → loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


# 全局 loguru logger（各模块 from config.logging_config import logger）
logger = _loguru
