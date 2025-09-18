import sqlite3
import threading
from loguru import logger
from typing import List, Optional, Dict, Any
from utils.models import Task
from utils.file import data_dir


"""
Table: t_book_meta
存储书/报告的元信息。

字段:
- run_id: TEXT (PRIMARY KEY) - 整个流程运行的唯一ID。
- category: TEXT - 任务类别 ('story', 'book', 'report')。
- language: TEXT - 任务语言 ('cn', 'en')。
- goal: TEXT - 任务的核心目标。
- root_name: TEXT - 根任务的名字, 书名, 报告名。
- length: TEXT - 预估总字数。
- day_wordcount_goal: INTEGER - 每日字数目标。
- instructions: TEXT - 任务的【具体指令】。
- input_brief: TEXT - 任务的【输入指引】。
- constraints: TEXT - 任务的【限制和禁忌】。
- acceptance_criteria: TEXT - 任务的【验收标准】。
- created_at: TIMESTAMP - 记录创建时的时间戳。
- updated_at: TIMESTAMP - 记录最后更新时的时间戳。
"""


class BookMetaDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"SQLite BookMetaDB 连接已建立: {self.db_path}")
        self.cursor = self.conn.cursor()
        self._lock = threading.Lock()
        self._create_table()


    def _create_table(self):
        with self._lock:
            logger.debug("正在检查并创建 t_book_meta 表...")
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS t_book_meta (
                run_id TEXT PRIMARY KEY,
                category TEXT,
                language TEXT,
                goal TEXT,
                root_name TEXT,
                length TEXT,
                day_wordcount_goal INTEGER,
                instructions TEXT,
                input_brief TEXT,
                constraints TEXT,
                acceptance_criteria TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_book_meta_updated_at ON t_book_meta (updated_at)
            """)
            self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS t_book_meta_auto_update_timestamp
            AFTER UPDATE ON t_book_meta
            FOR EACH ROW
            BEGIN
                UPDATE t_book_meta SET updated_at = CURRENT_TIMESTAMP WHERE run_id = OLD.run_id;
            END;
            """)
            logger.debug("t_book_meta 表及相关索引、触发器已准备就绪。")
            self.conn.commit()


    def add_or_update_book_meta(self, task: Task):
        """
        根据根任务信息，添加或更新书的元数据。
        """
        meta_data = {
            "run_id": task.run_id,
            "category": task.category,
            "language": task.language,
            "goal": task.goal,
            "root_name": task.root_name,
            "length": task.length,
            "day_wordcount_goal": task.day_wordcount_goal,
            "instructions": "\n".join(task.instructions),
            "input_brief": "\n".join(task.input_brief),
            "constraints": "\n".join(task.constraints),
            "acceptance_criteria": "\n".join(task.acceptance_criteria),
        }

        columns = ", ".join(meta_data.keys())
        placeholders = ", ".join([f":{key}" for key in meta_data.keys()])
        update_clause = ", ".join([f"{key} = excluded.{key}" for key in meta_data if key != 'run_id'])

        with self._lock:
            logger.debug(f"正在添加或更新书籍元数据: run_id='{task.run_id}'")
            self.cursor.execute(
                f"""
                INSERT INTO t_book_meta ({columns})
                VALUES ({placeholders})
                ON CONFLICT(run_id) DO UPDATE SET
                    {update_clause}
                """,
                meta_data
            )
            self.conn.commit()
            logger.info(f"书籍元数据添加/更新成功: run_id='{task.run_id}'")


    def get_book_meta(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 run_id 获取单本书的元数据。
        """
        with self._lock:
            logger.debug(f"正在查询书籍元数据: run_id='{run_id}'")
            self.cursor.execute(
                "SELECT * FROM t_book_meta WHERE run_id = ?",
                (run_id,)
            )
            row = self.cursor.fetchone()
        logger.info(f"书籍元数据查询 {'成功' if row else '失败'}: run_id='{run_id}'")
        return dict(row) if row else None


    def get_all_book_meta(self) -> List[Dict[str, Any]]:
        """
        获取所有书的元数据列表，按最后更新时间降序排列。
        """
        with self._lock:
            logger.debug("正在查询所有书籍的元数据...")
            self.cursor.execute(
                "SELECT * FROM t_book_meta ORDER BY updated_at DESC"
            )
            rows = self.cursor.fetchall()
        logger.info(f"共查询到 {len(rows)} 本书的元数据。")
        return [dict(row) for row in rows]


    def close(self):
        with self._lock:
            if self.conn:
                logger.info(f"正在关闭 SQLite BookMetaDB 连接: {self.db_path}")
                self.conn.close()


###############################################################################


_db_instance: Optional[BookMetaDB] = None
_db_lock = threading.Lock()

def get_meta_db() -> BookMetaDB:
    global _db_instance
    with _db_lock:
        if _db_instance is None:
            logger.info("正在创建全局 BookMetaDB 实例...")
            db_path = data_dir / "books.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _db_instance = BookMetaDB(db_path=str(db_path))
        return _db_instance


###############################################################################


if __name__ == '__main__':
    import tempfile
    import shutil
    import os
    from utils.log import init_logger

    init_logger("sqlite_meta_test")

    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test_meta.db")

    # 模拟一个 Task 对象
    book_task = Task(id="1", parent_id="0", task_type="design", goal="写一本伟大的小说", category="story", language="cn", root_name="我的第一本书", run_id="book_run_1", day_wordcount_goal=1000)

    try:
        logger.info("--- 开始 SQLite BookMetaDB 测试 ---")
        db = BookMetaDB(db_path)

        logger.info("--- 测试: 添加/更新元数据 ---")
        db.add_or_update_book_meta(book_task)

        logger.info("--- 测试: 获取元数据 ---")
        meta = db.get_book_meta("book_run_1")
        logger.info(f"获取到的元数据: {meta}")
        assert meta is not None
        assert meta["root_name"] == "我的第一本书"
        assert meta["day_wordcount_goal"] == 1000

        db.close()
        logger.success("--- SQLite BookMetaDB 测试通过 ---")

    finally:
        shutil.rmtree(test_dir)
        logger.info(f"测试目录已删除: {test_dir}")
