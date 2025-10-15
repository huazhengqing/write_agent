import hashlib
import re
from loguru import logger
from typing import List, Optional, Dict, Any
from functools import lru_cache
from utils.file import sanitize_filename
from utils.models import Task


"""
Table: t_book_meta
存储书/报告的元信息。

字段:
- run_id: TEXT (PRIMARY KEY) - 整个流程运行的唯一ID。
- category: TEXT - 任务类别 ('story', 'book', 'report')。
- language: TEXT - 任务语言 ('cn', 'en')。
- goal: TEXT - 任务的核心目标。
- name: TEXT - 根任务的名字, 书名, 报告名。
- length: TEXT - 预估总字数。
- day_wordcount_goal: INTEGER - 每日字数目标。
- instructions: TEXT - 任务的[具体指令]。
- input_brief: TEXT - 任务的[输入指引]。
- constraints: TEXT - 任务的[限制和禁忌]。
- acceptance_criteria: TEXT - 任务的[验收标准]。
- title: TEXT - 书名
- synopsis: TEXT - 简介。
- style: TEXT - 叙事风格。
- book_level_design: TEXT - 全书设计方案
- global_state_summary: TEXT - 全局状态摘要
- status: TEXT - 项目状态 ('idle', 'running', 'paused', 'completed', 'archived')。
- created_at: TIMESTAMP - 记录创建时的时间戳。
- updated_at: TIMESTAMP - 记录最后更新时的时间戳。
"""


class BookMetaDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        import sqlite3
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        import threading
        self._lock = threading.Lock()
        self._create_table()

    def _create_table(self):
        with self._lock:
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS t_book_meta (
                run_id TEXT PRIMARY KEY,
                category TEXT,
                language TEXT,
                goal TEXT,
                name TEXT,
                length TEXT,
                day_wordcount_goal INTEGER,
                instructions TEXT,
                input_brief TEXT,
                constraints TEXT,
                acceptance_criteria TEXT,
                title TEXT,
                synopsis TEXT,
                style TEXT,
                book_level_design TEXT,
                global_state_summary TEXT,
                status TEXT DEFAULT 'idle',
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
            self.conn.commit()

    def add_book(self, book_info: Dict[str, Any]) -> str:
        if "run_id" not in book_info or not book_info["run_id"]:
            if "name" not in book_info or not book_info["name"]:
                raise ValueError("创建新书时，'name' 字段不能为空。")
            book_info["run_id"] = sanitize_filename(book_info["name"])
        if "status" not in book_info:
            book_info["status"] = "idle"
        run_id = book_info["run_id"]
        columns = ", ".join(book_info.keys())
        placeholders = ", ".join([f":{key}" for key in book_info.keys()])
        update_clause = ", ".join([f"{key} = :{key}" for key in book_info if key != 'run_id'])
        with self._lock:
            self.cursor.execute(
                f"""
                INSERT INTO t_book_meta ({columns})
                VALUES ({placeholders})
                ON CONFLICT(run_id) DO UPDATE SET
                    {update_clause}, updated_at = CURRENT_TIMESTAMP
                """,
                book_info
            )
            self.conn.commit()
        return run_id


    def update_title(self, run_id: str, title: str):
        with self._lock:
            self.cursor.execute(
                "UPDATE t_book_meta SET title = ? WHERE run_id = ?",
                (title, run_id)
            )
            self.conn.commit()

    def update_synopsis(self, run_id: str, synopsis: str):
        with self._lock:
            self.cursor.execute(
                "UPDATE t_book_meta SET synopsis = ? WHERE run_id = ?",
                (synopsis, run_id)
            )
            self.conn.commit()

    def update_style(self, run_id: str, style: str):
        with self._lock:
            self.cursor.execute(
                "UPDATE t_book_meta SET style = ? WHERE run_id = ?",
                (style, run_id)
            )
            self.conn.commit()

    def update_book_level_design(self, run_id: str, book_level_design: str):
        with self._lock:
            self.cursor.execute(
                "UPDATE t_book_meta SET book_level_design = ? WHERE run_id = ?",
                (book_level_design, run_id)
            )
            self.conn.commit()

    def update_global_state_summary(self, run_id: str, global_state_summary: str):
        with self._lock:
            self.cursor.execute(
                "UPDATE t_book_meta SET global_state_summary = ? WHERE run_id = ?",
                (global_state_summary, run_id)
            )
            self.conn.commit()

    def update_status(self, run_id: str, status: str):
        with self._lock:
            self.cursor.execute(
                "UPDATE t_book_meta SET status = ? WHERE run_id = ?",
                (status, run_id)
            )
            self.conn.commit()

    def delete_book_meta(self, run_id: str):
        with self._lock:
            self.cursor.execute(
                "DELETE FROM t_book_meta WHERE run_id = ?",
                (run_id,)
            )
            self.conn.commit()

    def get_book_meta(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self.cursor.execute(
                "SELECT * FROM t_book_meta WHERE run_id = ?",
                (run_id,)
            )
            row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_all_book_meta(self) -> List[Dict[str, Any]]:
        with self._lock:
            self.cursor.execute(
                "SELECT * FROM t_book_meta ORDER BY updated_at DESC"
            )
            rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self):
        with self._lock:
            if self.conn:
                self.conn.close()



###############################################################################



@lru_cache(maxsize=None)
def get_meta_db() -> BookMetaDB:
    import atexit
    from utils.file import data_dir
    db_path = data_dir / "books.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    instance = BookMetaDB(db_path=str(db_path))
    atexit.register(instance.close)
    return instance
