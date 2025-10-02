import json
from loguru import logger
from functools import lru_cache
from utils.models import Task, natural_sort_key


"""
Table: t_tasks
将任务信息和结果存储在本地 SQLite 数据库中。

字段:
- id: TEXT (PRIMARY KEY) - 唯一的层级任务ID。父任务id.子任务序号。(例如, '1', '1.1', '1.2.1')。
- parent_id: TEXT - 父任务的ID (已索引, 用于快速查找子任务)。
- task_type: TEXT - 任务类型 ('write', 'design', 'search')。
- hierarchical_position: TEXT - 任务的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。
- goal: TEXT - 任务的具体目标。
- instructions: TEXT - JSON 格式的、任务的具体指令和要求。
- input_brief: TEXT - JSON 格式的、任务执行时需要参考的关键上下文摘要。
- constraints: TEXT - JSON 格式的、任务的限制和禁忌。
- acceptance_criteria: TEXT - JSON 格式的、任务完成的验收标准。
- length: TEXT - 预估产出字数 (用于 'write' 任务)。
- atom: TEXT - 判断原子任务的完整JSON结果
- atom_reasoning: TEXT - 判断原子任务的推理过程
- atom_result: TEXT - 判断原子任务的结果 ('atom' 或 'complex')
- plan: TEXT - 任务分解结果
- plan_reasoning: TEXT - 任务分解的推理过程
- design: TEXT - 设计方案
- design_reasoning: TEXT - 设计方案的推理过程
- search: TEXT - 搜索结果
- search_reasoning: TEXT - 搜索结果的推理过程
- hierarchy: TEXT - 结构划分结果
- hierarchy_reasoning: TEXT - 结构划分的推理过程
- write: TEXT - 正文
- write_reasoning: TEXT - 正文的推理过程
- translation: TEXT - 翻译结果
- translation_reasoning: TEXT - 翻译的推理过程
- summary: TEXT - 正文摘要
- summary_reasoning: TEXT - 正文摘要的推理过程
- write_review: TEXT - 正文评审结果
- write_review_reasoning: TEXT - 正文评审的推理过程
- created_at: TIMESTAMP - 记录创建时的时间戳。
- updated_at: TIMESTAMP - 记录最后更新时的时间戳。
"""


class TaskDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        import sqlite3
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        import threading
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
                instructions TEXT,
                input_brief TEXT,
                constraints TEXT,
                acceptance_criteria TEXT,
                length TEXT,
                atom TEXT,
                atom_reasoning TEXT,
                atom_result TEXT,
                plan TEXT,
                plan_reasoning TEXT,
                design TEXT,
                design_reasoning TEXT,
                search TEXT,
                search_reasoning TEXT,
                hierarchy TEXT,
                hierarchy_reasoning TEXT,
                write TEXT,
                write_reasoning TEXT,
                translation TEXT,
                translation_reasoning TEXT,
                summary TEXT,
                summary_reasoning TEXT,
                write_review TEXT,
                write_review_reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        task_data = {
            "id": task.id,
            "parent_id": task.parent_id,
            "task_type": task.task_type,
            "hierarchical_position": task.hierarchical_position,
            "goal": task.goal,
            "length": task.length,
            "instructions": json.dumps(task.instructions, ensure_ascii=False),
            "input_brief": json.dumps(task.input_brief, ensure_ascii=False),
            "constraints": json.dumps(task.constraints, ensure_ascii=False),
            "acceptance_criteria": json.dumps(task.acceptance_criteria, ensure_ascii=False),
        }

        columns = ", ".join(task_data.keys())
        placeholders = ", ".join(["?"] * len(task_data))
        update_clause = ", ".join([f"{key} = excluded.{key}" for key in task_data if key != 'id'])

        with self._lock:
            # 使用 ON CONFLICT...DO UPDATE 语句实现原子性的插入或更新
            self.cursor.execute(
                f"""
                INSERT INTO t_tasks ({columns}) 
                VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET
                    {update_clause}
                """,
                tuple(task_data.values())
            )
            self.conn.commit()



    def add_sub_tasks(self, task: Task):
        import collections
        tasks_to_process = collections.deque(task.sub_tasks or [])
        while tasks_to_process:
            current_task = tasks_to_process.popleft()
            self.add_task(current_task)
            if current_task.sub_tasks:
                tasks_to_process.extend(current_task.sub_tasks)



    def add_result(self, task: Task):
        update_fields = {
            "atom_reasoning": task.results.get("atom_reasoning"),
            "atom_result": task.results.get("atom_result"),
            "plan_reasoning": task.results.get("plan_reasoning"),
            "design": task.results.get("design"),
            "design_reasoning": task.results.get("design_reasoning"),
            "search": task.results.get("search"),
            "search_reasoning": task.results.get("search_reasoning"),
            "hierarchy_reasoning": task.results.get("hierarchy_reasoning"),
            "write": task.results.get("write"),
            "write_reasoning": task.results.get("write_reasoning"),
            "translation": task.results.get("translation"),
            "translation_reasoning": task.results.get("translation_reasoning"),
            "summary": task.results.get("summary"),
            "summary_reasoning": task.results.get("summary_reasoning"),
            "write_review": task.results.get("write_review"),
            "write_review_reasoning": task.results.get("write_review_reasoning"),
        }
        # 统一处理可能需要序列化的字段 (plan, hierarchy, atom)
        for field_name in ["plan", "hierarchy", "atom"]:
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
            logger.info(f"任务 {task.id} 的结果已成功更新。更新字段: {list(fields_to_update.keys())}")



    def get_task_list(self, task: Task) -> str:
        """
        高效获取任务上下文列表 (父任务链 + 每个父任务的所有兄弟任务 + 当前任务的兄弟任务), 仅查询必要的字段。
        """
        all_task_data = {}
        with self._lock:
            # 1. 找出所有需要查询的 parent_id
            # 包括: 根 (parent_id IS NULL), 父任务链中每个任务的 parent_id, 以及当前任务的 parent_id
            # 根任务的 parent_id 可能是 '' (空字符串) 或 NULL, 为了健壮性, 两者都考虑。
            # 使用一个特殊值来代表根任务查询条件, 这里用空字符串 ''。
            parent_ids_to_query = {''} 
            if task.parent_id:
                parent_ids_to_query.add(task.parent_id)

            id_parts = task.id.split('.')
            if len(id_parts) > 1:
                # 例如, task.id '1.2.3', id_parts ['1', '2', '3']
                # 父任务链的 parent_id 是 '1' (1.2的父)
                for i in range(1, len(id_parts) - 1):
                    parent_ids_to_query.add(".".join(id_parts[:i]))

            # 2. 构建查询
            # 使用一个查询获取所有相关任务, 提高效率
            query_conditions = []
            params = []
            
            # 处理非 NULL 的 parent_id
            non_null_parent_ids = [pid for pid in parent_ids_to_query if pid != '']
            if non_null_parent_ids:
                placeholders = ','.join(['?'] * len(non_null_parent_ids))
                query_conditions.append(f"parent_id IN ({placeholders})")
                params.extend(non_null_parent_ids)
            
            # 处理根任务 (parent_id = '' 或 parent_id IS NULL)
            if '' in parent_ids_to_query:
                query_conditions.append("(parent_id = '' OR parent_id IS NULL)")

            query = f"SELECT id, hierarchical_position, task_type, goal, length FROM t_tasks WHERE {' OR '.join(query_conditions)}"
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
            output_lines.append(" ".join(str(part) for part in line_parts if part is not None))
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
                f"""
                SELECT id, design, write_review
                FROM t_tasks 
                WHERE id IN ({placeholders}) AND (design IS NOT NULL OR write_review IS NOT NULL)
                """,
                tuple(dependent_ids)
            )
            rows = self.cursor.fetchall()
        if not rows:
            return ""
        # 按 task id 进行自然排序 (例如, '1.10' 会在 '1.2' 之后)
        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = []
        for row in sorted_rows:
            design_content = row[1]
            write_review_content = row[2]
            if design_content:
                content_list.append(design_content)
            if write_review_content:
                content_list.append(write_review_content)
        return "\n\n".join(content_list)



    def has_preceding_sibling_design_tasks(self, task: Task) -> bool:
        """
        检查任务是否有同级的前置设计任务。
        """
        if not task.parent_id:
            return False
        id_parts = task.id.split('.')
        if len(id_parts) < 2:
            return False
        try:
            current_seq = int(id_parts[-1])
        except ValueError:
            return False

        # 如果是第一个子任务, 就没有前置兄弟任务
        if current_seq <= 1:
            return False

        # 生成需要查询的前置兄弟任务ID列表
        # 例如, 当前是 1.5, 则查询 1.1, 1.2, 1.3, 1.4
        preceding_sibling_ids = [f"{task.parent_id}.{i}" for i in range(1, current_seq)]

        with self._lock:
            placeholders = ','.join(['?'] * len(preceding_sibling_ids))
            # 使用 SELECT 1 ... LIMIT 1 来高效地检查存在性
            self.cursor.execute(
                f"SELECT 1 FROM t_tasks WHERE id IN ({placeholders}) AND task_type = 'design' LIMIT 1",
                tuple(preceding_sibling_ids)
            )
            row = self.cursor.fetchone()
        
        exists = row is not None
        return exists



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
                f"""
                SELECT id, search 
                FROM t_tasks 
                WHERE id IN ({placeholders}) AND task_type = 'search' 
                AND search IS NOT NULL AND search != ''
                """,
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
                """
                SELECT id, design
                FROM t_tasks 
                WHERE parent_id = ? AND task_type = 'design'
                """,
                (parent_id,)
            )
            rows = self.cursor.fetchall()

        if not rows:
            return ""

        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = []
        for row in sorted_rows:
            design_content = row[1]  # design
            if design_content:
                content_list.append(design_content)
        return "\n\n".join(content_list)



    def get_subtask_search(self, parent_id: str) -> str:
        with self._lock:
            self.cursor.execute(
                """
                SELECT id, search 
                FROM t_tasks 
                WHERE parent_id = ? AND task_type = 'search' 
                AND search IS NOT NULL AND search != ''
                """,
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
                """
                SELECT id, summary 
                FROM t_tasks 
                WHERE parent_id = ? AND task_type = 'write' 
                AND summary IS NOT NULL AND summary != ''
                """,
                (parent_id,)
            )
            rows = self.cursor.fetchall()

        if not rows:
            return ""

        sorted_rows = sorted(rows, key=lambda row: natural_sort_key(row[0]))
        content_list = [row[1] for row in sorted_rows if row[1]]
        
        return "\n\n".join(content_list)



    def get_latest_write(self, length: int = 500) -> str:
        with self._lock:
            self.cursor.execute(
                """
                SELECT write 
                FROM t_tasks 
                WHERE write IS NOT NULL AND write != '' 
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
        获取最近24小时内 'write' 字段的总字数。
        """
        with self._lock:
            self.cursor.execute(
                """
                SELECT write 
                FROM t_tasks 
                WHERE updated_at >= datetime('now', '-24 hours') 
                AND write IS NOT NULL AND write != ''
                """
            )
            rows = self.cursor.fetchall()
            if not rows:
                return 0
            total_words = sum(len(row[0]) for row in rows)
            return total_words



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



    def get_oldest_task_to_translate(self) -> dict | None:
        """
        获取最早一个需要翻译的任务。
        条件: 'write' 字段有内容, 但 'translation' 字段为空或NULL。
        返回: 单个任务的字典, 如果找不到则返回 None。
        """
        with self._lock:
            import sqlite3
            original_factory = self.conn.row_factory
            self.conn.row_factory = sqlite3.Row
            self.cursor.execute(
                """
                SELECT * FROM t_tasks 
                WHERE write IS NOT NULL AND write != '' 
                AND (translation IS NULL OR translation = '')
                ORDER BY id
                LIMIT 1
                """
            )
            row = self.cursor.fetchone()
            self.conn.row_factory = original_factory
        return dict(row) if row else None



    def get_task_type(self, task_id: str) -> str | None:
        """
        根据 task_id 高效获取单个任务的类型。
        """
        if not task_id:
            return None
        with self._lock:
            self.cursor.execute(
                "SELECT task_type FROM t_tasks WHERE id = ?",
                (task_id,)
            )
            row = self.cursor.fetchone()
        return row[0] if row else None



    def close(self):
        with self._lock:
            if self.conn:
                self.conn.close()


###############################################################################



@lru_cache(maxsize=None)
def get_task_db(run_id: str) -> TaskDB:
    from utils.file import data_dir
    db_path = data_dir / run_id / "task.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = TaskDB(db_path=str(db_path))
    return store
