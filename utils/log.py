import threading
from loguru import logger
import logging
from utils.file import log_dir


class PropagateHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        logging.getLogger(record.name).handle(record)


_pytest_logger_initialized = False

def init_logger(file_name):
    global _pytest_logger_initialized
    if not _pytest_logger_initialized:
        try:
            logger.remove()
        except ValueError:
            pass
        # 在测试期间, 注释此行可将所有日志输出重定向到文件
        # logger.add(PropagateHandler(), format="{message}", level="DEBUG")
        _pytest_logger_initialized = True

    sink_id = logger.add(
        log_dir / f"{file_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        level="DEBUG",
        enqueue=False,  # 改为False, 使用同步日志记录, 在Prefect中更可靠
        backtrace=True,
        diagnose=True,
    )
    return sink_id


def init_logger_by_runid(file_name):
    logger.remove()
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            # 找到调用栈的正确深度
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("llama_index").setLevel(logging.DEBUG)
    logger.add(
        log_dir / f"{file_name}.log",
        filter=lambda record: not record["extra"].get("run_id"), 
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        level="DEBUG", 
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


_SINK_IDS = {}
_SINK_LOCK = threading.Lock()

def ensure_task_logger(run_id: str):
    if run_id in _SINK_IDS:
        return

    with _SINK_LOCK:
        if run_id in _SINK_IDS:
            return
        sink_id = logger.add(
            log_dir / f"{run_id}.log",
            filter=lambda record: record["extra"].get("run_id") == run_id,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        _SINK_IDS[run_id] = sink_id
