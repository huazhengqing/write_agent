from functools import lru_cache
from loguru import logger
import logging
from utils.file import log_dir



@lru_cache(maxsize=None)
def init_logger(file_name):
    logger.remove()
    log_path = log_dir / f"{file_name}.log"
    if log_path.exists():
        log_path.unlink()
    sink_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        level="DEBUG",
        enqueue=False,  # 改为False, 使用同步日志记录, 在Prefect中更可靠
        backtrace=True,
        diagnose=True,
    )
    return sink_id



@lru_cache(maxsize=None)
def init_logger_by_runid(file_name):
    logger.remove()
    log_path = log_dir / f"{file_name}.log"
    if log_path.exists():
        log_path.unlink()
    # class InterceptHandler(logging.Handler):
    #     def emit(self, record: logging.LogRecord):
    #         try:
    #             level = logger.level(record.levelname).name
    #         except ValueError:
    #             level = record.levelno
    #         # 找到调用栈的正确深度
    #         frame, depth = logging.currentframe(), 2
    #         while frame and frame.f_code.co_filename == logging.__file__:
    #             frame = frame.f_back
    #             depth += 1
    #         logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    # logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    # logging.getLogger("llama_index").setLevel(logging.DEBUG)
    logger.add(
        log_path,
        filter=lambda record: not record["extra"].get("run_id"), 
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        level="DEBUG", 
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )



@lru_cache(maxsize=None)
def ensure_task_logger(run_id: str):
    log_path = log_dir / f"{run_id}.log"
    if log_path.exists():
        log_path.unlink()
    logger.add(
        log_path,
        filter=lambda record: record["extra"].get("run_id") == run_id,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
