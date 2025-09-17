import sqlite3
import json
import collections
import threading
from typing import List, Optional, Dict
from utils.models import Task, natural_sort_key
from utils.file import data_dir


"""
Table: t_tasks
将任务信息和结果存储在本地 SQLite 数据库中。

字段:
- id: TEXT (PRIMARY KEY) - 唯一的层级任务ID。父任务id.子任务序号。(例如, '1', '1.1', '1.2.1')。
- parent_id: TEXT - 父任务的ID (已索引, 用于快速查找子任务)。
- task_type: TEXT - 任务类型 ('write', 'design', 'search')。
- hierarchical_position: TEXT - 任务的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。
- goal: TEXT - 任务的具体目标。
- length: TEXT - 预估产出字数 (用于 'write' 任务)。
- dependency: TEXT - JSON 格式的、执行前必须完成的同级任务ID列表。
- plan: TEXT - 任务分解结果
- plan_reasoning: TEXT - 任务分解的推理过程
- plan_reflection: TEXT - 任务分解结果的反思结果
- plan_reflection_reasoning: TEXT - 任务分解结果的反思的推理过程
- design: TEXT - 设计方案
- design_reasoning: TEXT - 设计方案的推理过程
- design_reflection: TEXT - 设计方案的反思结果
- design_reflection_reasoning: TEXT -  设计方案的反思的推理过程
- search: TEXT - 搜索结果
- search_reasoning: TEXT - 搜索结果的推理过程
- write: TEXT - 正文
- write_reasoning: TEXT - 正文的推理过程
- write_reflection: TEXT - 正文的反思结果
- write_reflection_reasoning: TEXT - 正文的反思的推理过程
- summary: TEXT - 正文摘要
- summary_reasoning: TEXT - 正文摘要的推理过程
- atom: TEXT - 判断原子任务的完整JSON结果
- atom_reasoning: TEXT - 判断原子任务的推理过程
- atom_result: TEXT - 判断原子任务的结果 ('atom' 或 'complex')
- created_at: TIMESTAMP - 记录创建时的时间戳。
- review_design: TEXT - 设计评审结果
- review_design_reasoning: TEXT - 设计评审的推理过程
- review_write: TEXT - 正文评审结果
- review_write_reasoning: TEXT - 正文评审的推理过程
- updated_at: TIMESTAMP - 记录最后更新时的时间戳。
"""


class TaskDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._lock = threading.Lock()
        self._create_table()


    def _create_table(self):
        with self._lock:
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS t_tasks (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                task_type TEXT,
                hierarchical_position TEXT,
                goal TEXT,
                length TEXT,
                dependency TEXT,
                plan TEXT,
                plan_reasoning TEXT,
                plan_reflection TEXT,
                plan_reflection_reasoning TEXT,
                design TEXT,
                design_reasoning TEXT,
                design_reflection TEXT,
                design_reflection_reasoning TEXT,
                search TEXT,
                search_reasoning TEXT,
                write TEXT,
                write_reasoning TEXT,
                write_reflection TEXT,
                write_reflection_reasoning TEXT,
                summary TEXT,
                summary_reasoning TEXT,
                atom TEXT,
                atom_reasoning TEXT,
                atom_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                review_design TEXT,
                review_design_reasoning TEXT,
                review_write TEXT,
                review_write_reasoning TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent_id ON t_tasks (parent_id)
            """)
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated_at ON t_tasks (updated_at)
            """)
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent_id_task_type ON t_tasks (parent_id, task_type)
            """)
            self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS t_tasks_auto_update_timestamp
            AFTER UPDATE ON t_tasks
            FOR EACH ROW
            BEGIN
                UPDATE t_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
            """)
            self.conn.commit()


    def add_task(self, task: Task):
        dependency = json.dumps(task.dependency, ensure_ascii=False)
        with self._lock:
            self.cursor.execute(
                """
                INSERT INTO t_tasks 
                (id, parent_id, task_type, hierarchical_position, goal, length, dependency) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    parent_id = excluded.parent_id,
                    task_type = excluded.task_type,
                    hierarchical_position = excluded.hierarchical_position,
                    goal = excluded.goal,
                    length = excluded.length,
                    dependency = excluded.dependency
                """,
                (
                    task.id, 
                    task.parent_id, 
                    task.task_type, 
                    task.hierarchical_position, 
                    task.goal, 
                    task.length, 
                    dependency
                )
            )
            self.conn.commit()


    def add_sub_tasks(self, task: Task):
        tasks_to_process = collections.deque(task.sub_tasks or [])
        while tasks_to_process:
            current_task = tasks_to_process.popleft()
            self.add_task(current_task)

            if current_task.sub_tasks:
                tasks_to_process.extend(current_task.sub_tasks)


    def add_result(self, task: Task):
        update_fields = {
            "plan_reasoning": task.results.get("plan_reasoning"),
            "plan_reflection_reasoning": task.results.get("plan_reflection_reasoning"),
            "design": task.results.get("design"),
            "design_reasoning": task.results.get("design_reasoning"),
            "design_reflection": task.results.get("design_reflection"),
            "design_reflection_reasoning": task.results.get("design_reflection_reasoning"),
            "search": task.results.get("search"),
            "search_reasoning": task.results.get("search_reasoning"),
            "write": task.results.get("write"),
            "write_reasoning": task.results.get("write_reasoning"),
            "write_reflection": task.results.get("write_reflection"),
            "write_reflection_reasoning": task.results.get("write_reflection_reasoning"),
            "summary": task.results.get("summary"),
            "summary_reasoning": task.results.get("summary_reasoning"),
            "atom_reasoning": task.results.get("atom_reasoning"),
            "atom_result": task.results.get("atom_result"),
            "review_design": task.results.get("review_design"),
            "review_design_reasoning": task.results.get("review_design_reasoning"),
            "review_write": task.results.get("review_write"),
            "review_write_reasoning": task.results.get("review_write_reasoning"),
        }
        # 统一处理可能需要序列化的字段 (plan, plan_reflection, atom)
        for field_name in ["plan", "plan_reflection", "atom"]:
            data = task.results.get(field_name)
            if isinstance(data, dict):
                update_fields[field_name] = json.dumps(data, ensure_ascii=False)
            else:
                update_fields[field_name] = data

        # 过滤掉None值和空字符串
        fields_to_update = {k: v for k, v in update_fields.items() if v}
        if not fields_to_update:
            return

        set_clause = ", ".join([f"{key} = ?" for key in fields_to_update.keys()])
        values = list(fields_to_update.values())
        values.append(task.id)
        with self._lock:
            self.cursor.execute(
                f"""
                UPDATE t_tasks 
                SET {set_clause}
                WHERE id = ?
                """,
                tuple(values)
            )
            self.conn.commit()


    def get_task_list(self, task: Task) -> str:
        """
        高效获取任务上下文列表 (父任务链 + 兄弟任务), 仅查询必要的字段。
        """
        all_task_data = {}
        with self._lock:
            id_parts = task.id.split('.')
            parent_chain_ids = []
            if len(id_parts) > 1:
                parent_chain_ids = [".".join(id_parts[:i]) for i in range(1, len(id_parts))]
            query_parts = []
            params = []
            if parent_chain_ids:
                placeholders = ','.join(['?'] * len(parent_chain_ids))
                query_parts.append(f"id IN ({placeholders})")
                params.extend(parent_chain_ids)
            if task.parent_id:
                query_parts.append("parent_id = ?")
                params.append(task.parent_id)
            if query_parts:
                query = f"SELECT id, hierarchical_position, task_type, goal, length FROM t_tasks WHERE {' OR '.join(query_parts)}"
                self.cursor.execute(query, tuple(params))
                rows = self.cursor.fetchall()
                for row in rows:
                    all_task_data[row[0]] = (row[1], row[2], row[3], row[4])
        all_task_data[task.id] = (task.hierarchical_position, task.task_type, task.goal, task.length)
        sorted_ids = sorted(all_task_data.keys(), key=natural_sort_key)
        output_lines = []
        for task_id in sorted_ids:
            hierarchical_position, task_type, goal, length = all_task_data[task_id]
            line_parts = [task_id, hierarchical_position, task_type, goal, length]
            output_lines.append(" ".join(filter(None, map(str, line_parts))))
        return "\n".join(output_lines)


    def get_dependent_design(self, task: Task) -> str:
        if not task.parent_id:
            return ""
        id_parts = task.id.split('.')
        if len(id_parts) < 2:
            return ""
        try:
            current_seq = int(id_parts[-1])
        except ValueError:
            return ""

        # 生成需要查询的兄弟任务及当前任务自身的ID列表
        # 例如, 当前是 1.5, 则查询 1.1, 1.2, 1.3, 1.4, 1.5
        dependent_ids = [f"{task.parent_id}.{i}" for i in range(1, current_seq + 1)]
        if not dependent_ids:
            return ""

        with self._lock:
            placeholders = ','.join(['?'] * len(dependent_ids))
            self.cursor.execute(
                f"SELECT id, design, design_reflection, review_design, review_write FROM t_tasks WHERE id IN ({placeholders})",
                tuple(dependent_ids)
            )
            rows = self.cursor.fetchall()
        if not rows:
            return ""
        # 按 task id 进行自然排序 (例如, '1.10' 会在 '1.2' 之后)
        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = []
        for row in sorted_rows:
            # 逻辑：组合多个设计相关字段
            # 1. 设计内容: design_reflection 优先于 design
            # 2. 设计评审: review_design
            # 3. 正文评审: review_write
            parts = []
            design_content = row[2] or row[1]  # design_reflection or design
            if design_content:
                parts.append(design_content)
            if row[3]:  # review_design
                parts.append(row[3])
            if row[4]:  # review_write
                parts.append(row[4])
            content = "\n\n".join(parts)
            if content:
                content_list.append(content)
        return "\n\n".join(content_list)


    def get_dependent_search(self, task: Task) -> str:
        if not task.parent_id:
            return ""
        id_parts = task.id.split('.')
        if len(id_parts) < 2:
            return ""
        try:
            current_seq = int(id_parts[-1])
        except ValueError:
            return ""
        dependent_ids = [f"{task.parent_id}.{i}" for i in range(1, current_seq)]
        if not dependent_ids:
            return ""
        with self._lock:
            placeholders = ','.join(['?'] * len(dependent_ids))
            self.cursor.execute(
                f"SELECT id, search FROM t_tasks WHERE id IN ({placeholders}) AND task_type = 'search'",
                tuple(dependent_ids)
            )
            rows = self.cursor.fetchall()
        if not rows:
            return ""
        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = [row[1] for row in sorted_rows if row[1]]
        return "\n\n".join(content_list)


    def get_subtask_design(self, parent_id: str) -> str:
        with self._lock:
            self.cursor.execute(
                "SELECT id, design, design_reflection, review_design, review_write FROM t_tasks WHERE parent_id = ? AND task_type = 'design'",
                (parent_id,)
            )
            rows = self.cursor.fetchall()

        if not rows:
            return ""

        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = []
        for row in sorted_rows:
            # 逻辑：组合多个设计相关字段
            # 1. 设计内容: design_reflection 优先于 design
            # 2. 设计评审: review_design
            # 3. 正文评审: review_write
            parts = []
            design_content = row[2] or row[1]  # design_reflection or design
            if design_content:
                parts.append(design_content)
            if row[3]:  # review_design
                parts.append(row[3])
            if row[4]:  # review_write
                parts.append(row[4])
            content = "\n\n".join(parts)
            if content:
                content_list.append(content)
        return "\n\n".join(content_list)


    def get_subtask_search(self, parent_id: str) -> str:
        with self._lock:
            self.cursor.execute(
                "SELECT id, search FROM t_tasks WHERE parent_id = ? AND task_type = 'search'",
                (parent_id,)
            )
            rows = self.cursor.fetchall()

        if not rows:
            return ""

        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = [row[1] for row in sorted_rows if row[1]]
        return "\n\n".join(content_list)


    def get_subtask_summary(self, parent_id: str) -> str:
        with self._lock:
            self.cursor.execute(
                "SELECT id, summary FROM t_tasks WHERE parent_id = ? AND task_type = 'write'",
                (parent_id,)
            )
            rows = self.cursor.fetchall()

        if not rows:
            return ""

        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = [row[1] for row in sorted_rows if row[1]]
        
        return "\n\n".join(content_list)


    def get_latest_write_reflection(self, length: int = 500) -> str:
        # 通过 updated_at 倒序获取最近更新的内容, 这是最高效的方式。
        # 此查询利用新创建的 idx_updated_at 索引, 避免了全表扫描, 显著提升性能。
        # 原有实现是基于任务ID进行自然排序, 必须加载所有数据到内存, 性能较差。
        # 此处将“最新”的定义从“任务ID最大”调整为“时间上最近更新”, 更符合直觉且高效。
        with self._lock:
            self.cursor.execute(
                """
                SELECT write_reflection 
                FROM t_tasks 
                WHERE write_reflection IS NOT NULL AND write_reflection != '' 
                ORDER BY updated_at DESC
                """
            )
            rows = self.cursor.fetchall()

        if not rows:
            return ""

        # 累积内容直到达到指定长度
        content_parts = []
        total_length = 0
        for row in rows:
            content = row[0]
            if content:
                content_parts.append(content)
                total_length += len(content)
                if total_length >= length:
                    break
        
        # 因为我们是按时间倒序获取的(最新在前), 所以需要反转列表以恢复正确的时序
        return "\n\n".join(reversed(content_parts))


    def get_word_count_last_24h(self) -> int:
        """
        获取最近24小时内 'write_reflection' 字段的总字数。
        """
        with self._lock:
            self.cursor.execute(
                """
                SELECT write_reflection 
                FROM t_tasks 
                WHERE updated_at >= datetime('now', '-24 hours') 
                AND write_reflection IS NOT NULL AND write_reflection != ''
                """
            )
            rows = self.cursor.fetchall()
            if not rows:
                return 0
            return sum(len(row[0]) for row in rows)


    def get_write_text(self, task: Task) -> str:
        with self._lock:
            self.cursor.execute(
                """
                SELECT id, write 
                FROM t_tasks 
                WHERE (id = ? OR id LIKE ?) 
                AND write IS NOT NULL AND write != ''
                """,
                (task.id, f"{task.id}.%")
            )
            rows = self.cursor.fetchall()
        if not rows:
            return ""
        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = [row[1] for row in sorted_rows if row[1]]
        return "\n\n".join(content_list)


    def close(self):
        with self._lock:
            if self.conn:
                self.conn.close()


###############################################################################


_stores: Dict[str, TaskDB] = {}
_lock = threading.Lock()
def get_task_db(run_id: str) -> TaskDB:
    with _lock:
        if run_id in _stores:
            return _stores[run_id]
        db_path = data_dir / run_id / "task.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = TaskDB(db_path=str(db_path))
        _stores[run_id] = store
        return store

