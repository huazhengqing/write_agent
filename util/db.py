#coding: utf8
import os
import sqlite3
import asyncio
import json
import collections
import threading
from loguru import logger
from typing import List, Optional, Dict
from util.models import Task


"""
Table: t_tasks
将任务信息和结果存储在本地 SQLite 数据库中。

字段:
- id: TEXT (PRIMARY KEY) - 唯一的层级任务ID (例如, '1', '1.1', '1.2.1')。
- parent_id: TEXT - 父任务的ID。
- task_type: TEXT - 任务类型 ('write', 'design', 'search')。
- goal: TEXT - 任务的具体目标。
- length: TEXT - 预估产出字数 (用于 'write' 任务)。
- dependency: TEXT - JSON 格式的、执行前必须完成的同级任务ID列表。
- reasoning: TEXT - 任务结果背后的推理过程。
- result: TEXT - 来自 llm 的完整结果
- design_reflection_reasoning: TEXT -  设计结果的反思原因。
- design_reflection: TEXT - 设计结果的反思结果
- reasoning_reflection: TEXT - 任务结果再反思背后的推理过程。
- result_reflection: TEXT - 来自 llm 的再反思完整结果
- summary: TEXT - 正文的摘要
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
            goal TEXT,
            length TEXT,
            dependency TEXT,
            reasoning TEXT,
            result TEXT,
            design_reflection_reasoning TEXT,
            design_reflection TEXT,
            reasoning_reflection TEXT,
            result_reflection TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.conn.commit()

    def add_task(self, task: Task):
        dependency = json.dumps(task.dependency, ensure_ascii=False)
        self.cursor.execute(
            """
            INSERT INTO t_tasks 
            (id, parent_id, task_type, goal, length, dependency) 
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                parent_id = excluded.parent_id,
                task_type = excluded.task_type,
                goal = excluded.goal,
                length = excluded.length,
                dependency = excluded.dependency
            """,
            (
                task.id, 
                task.parent_id, 
                task.task_type, 
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
            "reasoning": task.results.get("reasoning"),
            "result": task.results.get("result"),
            "design_reflection_reasoning": task.results.get("design_reflection_reasoning"),
            "design_reflection": task.results.get("design_reflection"),
            "reasoning_reflection": task.results.get("reasoning_reflection"),
            "result_reflection": task.results.get("result_reflection"),
            "summary": task.results.get("summary"),
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

    def add_text(self, task: Task):
        result = task.results.get("result") # 在rag.py中, 这是摘要
        self.cursor.execute(
            """
            UPDATE t_tasks 
            SET result = ?, summary = ?
            WHERE id = ?
            """,
            (result, result, task.id)
        )
        self.conn.commit()

    def get_task(self, id: str) -> Optional[Task]:
        self.cursor.execute("SELECT * FROM t_tasks WHERE id = ?", (id,))
        row = self.cursor.fetchone()
        if not row:
            logger.warning(f"未能在数据库中找到任务, DB: '{self.db_path}', Task ID: {id}")
            return None

        columns = [description[0] for description in self.cursor.description]
        task_data = dict(zip(columns, row))

        dependency = json.loads(task_data.get('dependency') or '[]')
        results = {
            "reasoning": task_data.get('reasoning'),
            "result": task_data.get('result'),
            "design_reflection_reasoning": task_data.get('design_reflection_reasoning'),
            "design_reflection": task_data.get('design_reflection'),
            "reasoning_reflection": task_data.get('reasoning_reflection'),
            "result_reflection": task_data.get('result_reflection'),
            "summary": task_data.get('summary'),
        }

        task_params = task_data.copy()
        task_params['dependency'] = dependency
        task_params['results'] = results
        task_params['sub_tasks'] = []
        
        task_params.setdefault('category', 'story')
        task_params.setdefault('language', 'cn')
        task_params.setdefault('root_name', 'default_root')
        task_params.setdefault('run_id', 'default_run')
        
        return Task(**task_params)

    def get_tasks_by_parent_id(self, parent_id: str, task_type: Optional[str] = None) -> List[Task]:
        query = "SELECT * FROM t_tasks WHERE parent_id = ?"
        params = [parent_id]
        
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)

        self.cursor.execute(query, tuple(params))
        rows = self.cursor.fetchall()
        if not rows:
            return []

        columns = [description[0] for description in self.cursor.description]
        tasks = []
        for row in rows:
            task_data = dict(zip(columns, row))

            dependency = json.loads(task_data.get('dependency') or '[]')
            results = {
                "reasoning": task_data.get('reasoning'),
                "result": task_data.get('result'),
                "design_reflection_reasoning": task_data.get('design_reflection_reasoning'),
                "design_reflection": task_data.get('design_reflection'),
                "reasoning_reflection": task_data.get('reasoning_reflection'),
                "result_reflection": task_data.get('result_reflection'),
                "summary": task_data.get('summary'),
            }

            task_params = task_data.copy()
            task_params['dependency'] = dependency
            task_params['results'] = results
            task_params['sub_tasks'] = []
            
            task_params.setdefault('category', 'story')
            task_params.setdefault('language', 'cn')
            task_params.setdefault('root_name', 'default_root')
            task_params.setdefault('run_id', 'default_run')
            
            tasks.append(Task(**task_params))
        
        return tasks

    """
    获取父任务链: 从根任务到当前任务的父任务的完整任务信息。
    任务ID格式为 "父任务ID.子任务序号", 根任务ID为 "1"。
    例如, 对于任务ID "1.3.5", 此函数将:
    - 获取父任务链 "1" 和 "1.3" 的信息。
    """
    def get_parent_tasks(self, task: Task) -> List[Task]:
        id_parts = task.id.split('.')
        if len(id_parts) <= 1:
            return []  # 根任务没有父任务

        # 生成父任务ID链, 例如, 对于 '1.3.5', 生成 ['1', '1.3']
        parent_chain_ids = [".".join(id_parts[:i]) for i in range(1, len(id_parts))]

        # 为 SQL IN 子句创建占位符
        placeholders = ','.join(['?'] * len(parent_chain_ids))
        query = f"SELECT * FROM t_tasks WHERE id IN ({placeholders})"
        
        self.cursor.execute(query, parent_chain_ids)
        rows = self.cursor.fetchall()

        if not rows:
            logger.warning(f"未能为任务 {task.id} 找到任何父任务。DB: '{self.db_path}'")
            return []

        columns = [description[0] for description in self.cursor.description]
        
        found_tasks = {}
        for row in rows:
            task_data = dict(zip(columns, row))

            dependency = json.loads(task_data.get('dependency') or '[]')
            results = {
                "reasoning": task_data.get('reasoning'),
                "result": task_data.get('result'),
                "design_reflection_reasoning": task_data.get('design_reflection_reasoning'),
                "design_reflection": task_data.get('design_reflection'),
                "reasoning_reflection": task_data.get('reasoning_reflection'),
                "result_reflection": task_data.get('result_reflection'),
                "summary": task_data.get('summary'),
            }

            task_params = task_data.copy()
            task_params['dependency'] = dependency
            task_params['results'] = results
            task_params['sub_tasks'] = []  # 父任务的子任务不在此处递归加载

            # 从当前任务继承元数据
            task_params['category'] = task.category
            task_params['language'] = task.language
            task_params['root_name'] = task.root_name
            task_params['run_id'] = task.run_id
            
            parent_task_obj = Task(**task_params)
            found_tasks[parent_task_obj.id] = parent_task_obj

        # 按父任务链的顺序对结果进行排序
        sorted_tasks = [found_tasks[pid] for pid in parent_chain_ids if pid in found_tasks]
        
        return sorted_tasks

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
        all_tasks = {}

        # 1. 获取父任务链
        parent_tasks = self.get_parent_tasks(task)
        for p_task in parent_tasks:
            all_tasks[p_task.id] = p_task

        # 2. 获取兄弟任务 (包括当前任务自身)
        if task.parent_id:
            sibling_tasks = self.get_tasks_by_parent_id(task.parent_id)
            for s_task in sibling_tasks:
                all_tasks[s_task.id] = s_task
        
        # 确保当前任务也被包含 (以防它尚未保存到数据库或需要使用最新的信息)
        all_tasks[task.id] = task

        # 3. 按ID进行层级化排序
        # 排序键将 '1.10.2' 转换为 [1, 10, 2] 以实现正确的数字排序
        sorted_tasks = sorted(all_tasks.values(), key=lambda t: [int(p) for p in t.id.split('.')])

        # 4. 格式化为指定的字符串格式
        output_lines = []
        for t in sorted_tasks:
            line = f"{t.id} {t.goal}"
            if t.length:
                line += f" {t.length}"
            output_lines.append(line)
            
        return "\n".join(output_lines)

    def get_dependent(self, task: Task, task_type: str) -> str:
        if not task.parent_id:
            return ""

        self.cursor.execute(
            "SELECT result FROM t_tasks WHERE parent_id = ? AND task_type = ?",
            (task.parent_id, task_type)
        )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        # rows 是一个元组列表, 例如 [('result1',), ('result2',)]
        # 我们需要提取每个元组的第一个元素, 并过滤掉 None/空字符串。
        content_list = [row[0] for row in rows if row[0]]
        
        return "\n\n".join(content_list)

    def get_subtask_results(self, parent_id: str, task_type: str) -> str:
        if task_type == "summary":
            self.cursor.execute(
                "SELECT summary FROM t_tasks WHERE parent_id = ? AND task_type = ?",
                (parent_id, "write")
            )
        else:
            self.cursor.execute(
                "SELECT result FROM t_tasks WHERE parent_id = ? AND task_type = ?",
                (parent_id, task_type)
            )
        rows = self.cursor.fetchall()

        if not rows:
            return ""

        content_list = [row[0] for row in rows if row[0]]
        
        return "\n\n".join(content_list)

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info(f"原文层数据库连接已关闭: {self.db_path}")


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
