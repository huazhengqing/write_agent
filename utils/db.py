#coding: utf8
import os
import sqlite3
import asyncio
import json
import collections
import threading
from loguru import logger
from typing import List, Optional, Dict
from utils.models import Task


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
- design: TEXT - 设计结果
- design_reasoning: TEXT - 设计结果的推理过程
- design_reflection: TEXT - 设计结果的反思结果
- design_reflection_reasoning: TEXT -  设计结果的反思的推理过程
- search: TEXT - 搜索结果
- search_reasoning: TEXT - 搜索结果的推理过程
- write: TEXT - 正文
- write_reasoning: TEXT - 正文的推理过程
- write_reflection: TEXT - 正文的反思结果
- write_reflection_reasoning: TEXT - 正文的反思的推理过程
- summary: TEXT - 正文摘要
- summary_reasoning: TEXT - 正文摘要的推理过程
- atom: TEXT - 判断原子任务的结果
- atom_reasoning: TEXT - 判断原子任务的推理过程
- created_at: TIMESTAMP - 记录创建/更新时的时间戳。
"""


class DB:
    def __init__(self, db_path: str):
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_parent_id ON t_tasks (parent_id)
        """)
        self.conn.commit()

    def add_task(self, task: Task):
        dependency = json.dumps(task.dependency, ensure_ascii=False)
        self.cursor.execute(
            """
            INSERT INTO t_tasks 
            (id, parent_id, task_type, hierarchical_position, goal, length, dependency) 
            VALUES (?, ?, ?, ?, ?, ?)
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
            "plan": task.results.get("plan"),
            "plan_reasoning": task.results.get("plan_reasoning"),
            "plan_reflection": task.results.get("plan_reflection"),
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
            "atom": task.results.get("atom"),
            "atom_reasoning": task.results.get("atom_reasoning"),
        }
        
        # 过滤掉None值和空字符串
        fields_to_update = {k: v for k, v in update_fields.items() if v}
        
        if not fields_to_update:
            return

        set_clause = ", ".join([f"{key} = ?" for key in fields_to_update.keys()])
        values = list(fields_to_update.values())
        values.append(task.id)

        self.cursor.execute(
            f"""
            UPDATE t_tasks 
            SET {set_clause}
            WHERE id = ?
            """,
            tuple(values)
        )
        self.conn.commit()

    """
    获取任务上下文, 包括两部分:
    1. 父任务链: 从根任务到当前任务的父任务的完整任务信息。
    2. 当前层级任务: 与当前任务同层级的所有任务(兄弟任务)的信息。

    任务ID格式为 "父任务ID.子任务序号", 根任务ID为 "1"。
    例如, 对于任务ID "1.3.5", 此函数将:
    - 获取父任务链 "1" 和 "1.3" 的信息。
    - 获取所有父ID为 "1.3" 的任务信息。
    返回的格式：
    1. goal length
    1.3 goal length
    1.3.1 goal  length
    1.3.2 goal  length
    1.3.3 goal  length
    1.3.4 goal  length
    1.3.5 goal  length
    如果 没有 length ，就不显示 
    """
    def get_context_task_list(self, task: Task) -> str:
        """
        高效获取任务上下文列表 (父任务链 + 兄弟任务), 仅查询必要的字段。
        """
        all_task_data = {}  # 使用字典存储

        # 1. 获取父任务链的 ID
        id_parts = task.id.split('.')
        if len(id_parts) > 1:
            parent_chain_ids = [".".join(id_parts[:i]) for i in range(1, len(id_parts))]
            
            # 为 SQL IN 子句创建占位符
            placeholders = ','.join(['?'] * len(parent_chain_ids))
            query = f"SELECT id, hierarchical_position, goal, length FROM t_tasks WHERE id IN ({placeholders})"
            
            self.cursor.execute(query, parent_chain_ids)
            rows = self.cursor.fetchall()
            for row in rows:
                all_task_data[row[0]] = (row[1], row[2], row[3])

        # 2. 获取兄弟任务 (与当前任务同属一个父任务)
        if task.parent_id:
            self.cursor.execute(
                "SELECT id, hierarchical_position, goal, length FROM t_tasks WHERE parent_id = ?",
                (task.parent_id,)
            )
            rows = self.cursor.fetchall()
            for row in rows:
                all_task_data[row[0]] = (row[1], row[2], row[3])
        
        # 3. 确保当前任务也被包含 (使用最新的内存中信息)
        all_task_data[task.id] = (task.hierarchical_position, task.goal, task.length)

        # 4. 按ID进行自然排序
        def natural_sort_key(task_id_str: str):
            return [int(p) for p in task_id_str.split('.')]

        sorted_ids = sorted(all_task_data.keys(), key=natural_sort_key)

        # 5. 格式化为指定的字符串输出
        output_lines = []
        for task_id in sorted_ids:
            hierarchical_position, goal, length = all_task_data[task_id]
            line = f"{task_id}"
            if hierarchical_position:
                line += f" {hierarchical_position}"
            line += f" {goal}"
            if length:
                line += f" {length}"
            output_lines.append(line)
            
        return "\n".join(output_lines)

    def get_dependent_design(self, task: Task) -> str:
        if not task.parent_id:
            return ""

        # 不能有 task_type = 'design'，因为 write 任务会有 design_reflection
        self.cursor.execute(
            "SELECT id, design, design_reflection FROM t_tasks WHERE parent_id = ?",
            (task.parent_id,)
        )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        # 按 task id 进行自然排序 (例如, '1.10' 会在 '1.2' 之后)
        def natural_sort_key(row):
            return [int(p) for p in row[0].split('.')]

        sorted_rows = sorted(rows, key=natural_sort_key)

        content_list = []
        for row in sorted_rows:
            # row[0] is id, row[1] is design, row[2] is design_reflection
            design_content = row[1]
            reflection_content = row[2]
            
            combined_content = []
            if design_content:
                combined_content.append(design_content)
            if reflection_content:
                combined_content.append(reflection_content)
            
            if combined_content:
                content_list.append("\n\n".join(combined_content))
                
        return "\n\n".join(content_list)

    def get_dependent_search(self, task: Task) -> str:
        if not task.parent_id:
            return ""

        self.cursor.execute(
            "SELECT id, search FROM t_tasks WHERE parent_id = ? AND task_type = 'search'",
            (task.parent_id,)
        )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        # 按 task id 进行自然排序
        def natural_sort_key(row):
            return [int(p) for p in row[0].split('.')]

        sorted_rows = sorted(rows, key=natural_sort_key)

        content_list = [row[1] for row in sorted_rows if row[1]]
        
        return "\n\n".join(content_list)

    def get_subtask_design(self, parent_id: str) -> str:
        self.cursor.execute(
            "SELECT id, design FROM t_tasks WHERE parent_id = ? AND task_type = 'design'",
            (parent_id,)
        )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        def natural_sort_key(row):
            return [int(p) for p in row[0].split('.')]
        sorted_rows = sorted(rows, key=natural_sort_key)

        content_list = [row[1] for row in sorted_rows if row[1]]
        return "\n\n".join(content_list)

    def get_subtask_search(self, parent_id: str) -> str:
        self.cursor.execute(
            "SELECT id, search FROM t_tasks WHERE parent_id = ? AND task_type = 'search'",
            (parent_id,)
        )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        def natural_sort_key(row):
            return [int(p) for p in row[0].split('.')]
        sorted_rows = sorted(rows, key=natural_sort_key)

        content_list = [row[1] for row in sorted_rows if row[1]]
        return "\n\n".join(content_list)

    def get_subtask_summary(self, parent_id: str) -> str:
        self.cursor.execute(
            "SELECT id, summary FROM t_tasks WHERE parent_id = ? AND task_type = 'write'",
            (parent_id,)
        )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        def natural_sort_key(row):
            return [int(p) for p in row[0].split('.')]
        sorted_rows = sorted(rows, key=natural_sort_key)

        content_list = [row[1] for row in sorted_rows if row[1]]
        
        return "\n\n".join(content_list)

    def get_latest_write_reflection(self, length: int) -> str:
        self.cursor.execute(
            "SELECT id, write_reflection FROM t_tasks WHERE write_reflection IS NOT NULL AND write_reflection != ''"
        )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        # 按 task id 进行自然排序
        def natural_sort_key(row):
            """为任务ID提供健壮的自然排序键，处理空或格式错误的ID。"""
            try:
                # 过滤掉拆分后可能产生的空字符串（如 '1.'），并转换为整数列表
                return [int(p) for p in row[0].split('.') if p]
            except (AttributeError, ValueError):
                # 如果ID为None、空字符串或格式错误，返回[]。
                # 在降序排序中，这会使无效ID排在最后。
                return []

        # 按任务ID倒序排列，从最新到最旧
        sorted_rows = sorted(rows, key=natural_sort_key, reverse=True)

        # 累积内容直到达到指定长度
        content_parts = []
        total_length = 0
        for row in sorted_rows:
            content = row[1]
            if content:
                content_parts.append(content)
                total_length += len(content)
                if total_length >= length:
                    break
        
        # 因为我们是倒序添加的（最新在前），所以需要反转列表以恢复正确的章节顺序
        return "\n\n".join(reversed(content_parts))

    def close(self):
        if self.conn:
            self.conn.close()


###############################################################################


_stores: Dict[str, DB] = {}
_lock = threading.Lock()

def get_db(run_id: str, category: str) -> DB:
    with _lock:
        if run_id in _stores:
            return _stores[run_id]
        
        db_path = os.path.join("output", category, f"{run_id}.db")
        store = DB(db_path=db_path)
        _stores[run_id] = store
        return store
