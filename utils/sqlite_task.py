from datetime import datetime
import json
from typing import Any, Dict, List, Optional
from loguru import logger
from functools import lru_cache
from utils.file import get_text_file_path, text_file_append
from utils.models import Task, natural_sort_key


"""
Table: t_tasks
将任务信息和结果存储在本地 SQLite 数据库中。

字段:
- id: TEXT (PRIMARY KEY) - 唯一的层级任务ID。父任务id.子任务序号。(例如, '1', '1.1', '1.2.1')。
- parent_id: TEXT - 父任务的ID (已索引, 用于快速查找子任务)。
- task_type: TEXT - 任务类型 ('write', 'design', 'search')。
- status: TEXT - 任务状态 ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused')。
- hierarchical_position: TEXT - 任务的层级和位置。例如: '全书', '第1卷', '第2幕', '第3章'。
- goal: TEXT - 任务的具体目标。
- instructions: TEXT - JSON 格式的、任务的具体指令和要求。
- input_brief: TEXT - JSON 格式的、任务执行时需要参考的关键上下文摘要。
- constraints: TEXT - JSON 格式的、任务的限制和禁忌。
- acceptance_criteria: TEXT - JSON 格式的、任务完成的验收标准。
- length: TEXT - 预估产出字数 (用于 'write' 任务)。
- results: TEXT - JSON格式, 存储未被提取到独立字段的其他结果数据。
- expert: TEXT - 路由到哪个专家
- atom: TEXT - 判断原子任务的结果 ('atom' 或 'complex')
- atom_reasoning: TEXT - 判断原子任务的推理过程
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
- summary: TEXT - 正文摘要
- summary_reasoning: TEXT - 正文摘要的推理过程
- book_level_design: TEXT - 全书设计方案
- global_state: TEXT - 全局状态摘要
- write_review: TEXT - 正文评审结果
- write_review_reasoning: TEXT - 正文评审的推理过程
- translation: TEXT - 翻译结果
- translation_reasoning: TEXT - 翻译的推理过程
- context_design TEXT - 上下文,
- context_summary TEXT - 上下文,
- context_search TEXT - 上下文,
- kg_design TEXT - 知识图谱提取,
- kg_write TEXT - 知识图谱提取,
- inquiry_design TEXT - 检索词,
- inquiry_summary TEXT - 检索词,
- inquiry_search TEXT - 检索词,
- x-litellm-cache-key TEXT - llm 缓存 key,
- created_at: TIMESTAMP - 记录创建时的时间戳。
- updated_at: TIMESTAMP - 记录最后更新时的时间戳。
"""


class TaskDB:
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
            CREATE TABLE IF NOT EXISTS t_tasks (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                task_type TEXT,
                status TEXT,
                hierarchical_position TEXT,
                goal TEXT,
                instructions TEXT,
                input_brief TEXT,
                constraints TEXT,
                acceptance_criteria TEXT,
                results TEXT,
                length TEXT,
                expert TEXT,
                atom TEXT,
                atom_reasoning TEXT,
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
                summary TEXT,
                summary_reasoning TEXT,
                book_level_design TEXT,
                global_state TEXT,
                write_review TEXT,
                write_review_reasoning TEXT,
                translation TEXT,
                translation_reasoning TEXT,
                context_design TEXT,
                context_summary TEXT,
                context_search TEXT,
                kg_design TEXT,
                kg_write TEXT,
                inquiry_design TEXT,
                inquiry_summary TEXT,
                inquiry_search TEXT,
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

    def _get_dedicated_result_columns(self) -> List[str]:
        """获取在t_tasks表中作为独立列存在的结果字段名"""
        # 这些字段存在于 t_tasks 表中，但其数据源于 Task.results
        return [
            "expert", "atom", "atom_reasoning", "plan", "plan_reasoning",
            "design", "design_reasoning", "search", "search_reasoning",
            "hierarchy", "hierarchy_reasoning", "write", "write_reasoning",
            "summary", "summary_reasoning", "book_level_design", "global_state",
            "write_review", "write_review_reasoning", "translation",
            "translation_reasoning", "context_design", "context_summary",
            "context_search", "kg_design", "kg_write",
            "inquiry_design", "inquiry_summary", "inquiry_search"
        ]

    def _prepare_remaining_results(self, task_results: Dict[str, Any]) -> str:
        """从任务结果中分离出需要存入JSON字段的数据"""
        dedicated_cols = self._get_dedicated_result_columns()
        remaining_results = {
            k: v for k, v in task_results.items()
            if k not in dedicated_cols and v is not None
        }
        return json.dumps(remaining_results, ensure_ascii=False, indent=2) if remaining_results else "{}"

    def add_task(self, task: Task):
        task_data = {
            "id": task.id,
            "parent_id": task.parent_id,
            "task_type": task.task_type,
            "status": task.status,
            "hierarchical_position": task.hierarchical_position,
            "goal": task.goal,
            "length": task.length,
            "instructions": task.instructions,
            "input_brief": task.input_brief,
            "constraints": task.constraints,
            "acceptance_criteria": task.acceptance_criteria,
            "results": self._prepare_remaining_results(task.results),
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
        # 1. 准备要更新到独立列的字段
        update_fields = {
            "atom": task.results.get("atom"),
            "atom_reasoning": task.results.get("atom_reasoning"),
            "plan": task.results.get("plan"),
            "plan_reasoning": task.results.get("plan_reasoning"),
            "design": task.results.get("design"),
            "design_reasoning": task.results.get("design_reasoning"),
            "search": task.results.get("search"),
            "search_reasoning": task.results.get("search_reasoning"),
            "hierarchy": task.results.get("hierarchy"),
            "hierarchy_reasoning": task.results.get("hierarchy_reasoning"),
            "write": task.results.get("write"),
            "write_reasoning": task.results.get("write_reasoning"),
            "summary": task.results.get("summary"),
            "summary_reasoning": task.results.get("summary_reasoning"),
            "translation": task.results.get("translation"),
            "translation_reasoning": task.results.get("translation_reasoning"),
            "book_level_design": task.results.get("book_level_design"),
            "global_state": task.results.get("global_state"),
            "write_review": task.results.get("write_review"),
            "write_review_reasoning": task.results.get("write_review_reasoning"),
        }
        # 过滤掉None值
        fields_to_update = {k: v for k, v in update_fields.items() if v is not None}

        # 2. 准备要更新到 'results' JSON列的剩余数据
        fields_to_update["results"] = self._prepare_remaining_results(task.results)

        if not fields_to_update:
            return

        set_clause = ", ".join([f"{key} = ?" for key in fields_to_update])
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

        self.save_write_file(task)



    def save_write_file(self, task: Task):
        final_content = task.results.get("write")
        if final_content:
            header_parts = [task.id, task.hierarchical_position, task.goal, task.length]
            header = " ".join(filter(None, header_parts))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = f"## 任务\n{header}\n{timestamp}\n\n{final_content}"
            text_file_append(get_text_file_path(task), content)



    def get_task_by_id(self, task_id: str) -> dict | None:
        """
        根据 task_id 获取单个任务的完整信息, 并返回一个字典。
        """
        if not task_id:
            return None
        
        row = None
        with self._lock:
            self.cursor.execute("SELECT * FROM t_tasks WHERE id = ?", (task_id,))
            row = self.cursor.fetchone()

        if not row:
            return None
        task_data = dict(row)
        
        # 将存储在 'results' 字段的JSON字符串解码并合并
        if task_data.get('results'):
            try:
                remaining_results = json.loads(task_data['results'])
                task_data.update(remaining_results)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"无法解析任务 {task_id} 的 'results' JSON 字段: {task_data['results']}")
        return task_data



    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取数据库中的所有任务, 并返回一个字典列表。
        """
        rows = []
        with self._lock:
            self.cursor.execute("SELECT * FROM t_tasks")
            rows = self.cursor.fetchall()

        if not rows:
            return []

        tasks_data = [dict(row) for row in rows]
        for task_data in tasks_data:
            # 将存储在 'results' 字段的JSON字符串解码并合并
            if task_data.get('results'):
                try:
                    remaining_results = json.loads(task_data['results'])
                    task_data.update(remaining_results)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"无法解析任务 {task_data.get('id', '未知ID')} 的 'results' JSON 字段: {task_data['results']}")
        return tasks_data



    def get_subtask_ids(self, parent_id: str, task_type: Optional[str] = None) -> list[str]:
        """
        获取指定父任务的所有直接子任务的ID列表, 并按自然顺序排序。
        例如, 对于 parent_id '1.3', 返回 ['1.3.1', '1.3.2', '1.3.3', ...]。
        如果提供了 task_type, 则只返回该类型的子任务ID。
        """
        if not parent_id:
            return []
        
        query = "SELECT id FROM t_tasks WHERE parent_id = ?"
        params = [parent_id]

        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)

        with self._lock:
            self.cursor.execute(
                query,
                tuple(params)
            )
            rows = self.cursor.fetchall()
        if not rows:
            return []
        subtask_ids = [row[0] for row in rows]
        return sorted(subtask_ids, key=natural_sort_key)



    def get_overall_planning(self, task: Task) -> str:
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



    def get_dependent_design(self, task: Optional[Task]) -> str:
        if not task:
            return ""
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



    def get_dependent_search(self, task: Optional[Task]) -> str:
        if not task:
            return ""
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





    def get_write_latest(self, length: int = 500) -> str:
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


    def get_text_latest(self, length: int = 500) -> str:
        full_content = self.get_write_latest(length)
        if len(full_content) <= length:
            result = full_content
        else:
            # 为了避免截断一个完整的段落或句子, 我们尝试从一个换行符后开始截取
            start_pos = len(full_content) - length
            # 1. 尝试在截取点之后找到第一个换行符, 从该换行符后开始截取
            first_newline_after_start = full_content.find('\n', start_pos)
            if first_newline_after_start != -1:
                result = full_content[first_newline_after_start + 1:]
            else:
                # 2. 如果后面没有换行符(说明我们在最后一段), 则尝试在截取点之前找到最后一个换行符
                last_newline_before_start = full_content.rfind('\n', 0, start_pos)
                if last_newline_before_start != -1:
                    result = full_content[last_newline_before_start + 1:]
                else:
                    # 3. 如果全文都没有换行符, 或者只有一段很长的内容, 则直接硬截取
                    result = full_content[-length:]
        return result


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

    def get_total_write_word_count(self) -> int:
        """
        获取所有 'write' 字段的总字数。
        """
        with self._lock:
            self.cursor.execute(
                """
                SELECT SUM(LENGTH(write)) 
                FROM t_tasks 
                WHERE write IS NOT NULL AND write != ''
                """
            )
            row = self.cursor.fetchone()
            if not row or row[0] is None:
                return 0
            return int(row[0])



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



    def update_task_status(self, task_id: str, status: str):
        """
        更新指定任务的状态。
        """
        if not task_id or not status:
            return
        with self._lock:
            self.cursor.execute(
                """
                UPDATE t_tasks 
                SET status = ?
                WHERE id = ?
                """,
                (status, task_id)
            )
            self.conn.commit()

    def update_task_expert(self, task_id: str, expert: str):
        """
        更新指定任务的 expert 字段。
        """
        if not task_id or not expert:
            return
        with self._lock:
            self.cursor.execute(
                """
                UPDATE t_tasks 
                SET expert = ?
                WHERE id = ?
                """,
                (expert, task_id)
            )
            self.conn.commit()

    def update_task_inquiry(self, task_id: str, inquiry_type: str, inquiry_content: str):
        """
        更新指定任务的 inquiry 字段。
        """
        if not task_id or not inquiry_type or inquiry_content is None:
            return
        
        field_name = f"inquiry_{inquiry_type}"
        # 简单的白名单校验，防止SQL注入
        if field_name not in ["inquiry_design", "inquiry_summary", "inquiry_search"]:
            logger.error(f"无效的 inquiry_type: {inquiry_type}")
            return

        with self._lock:
            self.cursor.execute(
                f"UPDATE t_tasks SET {field_name} = ? WHERE id = ?",
                (inquiry_content, task_id)
            )
            self.conn.commit()



    def update_task_context(self, task_id: str, context_type: str, context_content: str):
        """
        更新指定任务的 context 字段。
        """
        if not task_id or not context_type or context_content is None:
            return

        field_name = f"context_{context_type}"
        # 简单的白名单校验，防止SQL注入
        if field_name not in ["context_design", "context_summary", "context_search"]:
            logger.error(f"无效的 context_type: {context_type}")
            return

        with self._lock:
            self.cursor.execute(
                f"UPDATE t_tasks SET {field_name} = ? WHERE id = ?",
                (context_content, task_id)
            )
            self.conn.commit()



    def delete_task_and_subtasks(self, task_id: str):
        """
        删除指定任务及其所有子任务（级联删除）。
        """
        if not task_id:
            return

        tasks_to_delete = [task_id]
        queue = [task_id]

        # 使用广度优先搜索找到所有子孙任务
        while queue:
            current_id = queue.pop(0)
            sub_ids = self.get_subtask_ids(current_id)
            if sub_ids:
                tasks_to_delete.extend(sub_ids)
                queue.extend(sub_ids)

        with self._lock:
            placeholders = ','.join(['?'] * len(tasks_to_delete))
            self.cursor.execute(
                f"DELETE FROM t_tasks WHERE id IN ({placeholders})",
                tuple(tasks_to_delete)
            )
            self.conn.commit()
        logger.info(f"已级联删除任务: {task_id} 及其所有子任务。共删除 {len(tasks_to_delete)} 个任务。")


    def close(self):
        with self._lock:
            if self.conn:
                self.conn.close()


###############################################################################


def dict_to_task(task_data: dict) -> Task | None:
    """将从数据库查询出的字典转换为 Task 模型对象"""
    if not task_data:
        return None

    # 将存储在 'results' 字段的JSON字符串解码并合并到主字典中
    if task_data.get('results') and isinstance(task_data['results'], str):
        try:
            remaining_results = json.loads(task_data['results'])
            # 合并时，数据库中的独立列数据优先级更高
            task_data = {**remaining_results, **task_data}
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"无法解析任务 {task_data.get('id')} 的 'results' JSON 字段: {task_data['results']}")

    # 2. 准备 Task 模型的构造函数参数，并处理 results 字段
    task_args = {}
    results = {}
    for key, value in task_data.items():
        if key in Task.model_fields:
            task_args[key] = value
        elif value is not None:
            results[key] = value
    task_args['results'] = results
    
    # 3. 创建 Task 实例
    return Task(**task_args)


###############################################################################


@lru_cache(maxsize=None)
def get_task_db(run_id: str) -> TaskDB:
    from utils.file import data_dir
    db_path = data_dir / run_id / "task.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = TaskDB(db_path=str(db_path))
    return store
