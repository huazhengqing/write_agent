import streamlit as st
import json
import os
import sys
import asyncio # type: ignore
from typing import List, Dict, Any, Callable, Coroutine
from loguru import logger
import shutil

# --- 项目根目录设置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.models import Task, natural_sort_key
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import get_task_db, dict_to_task
from story.task import do_write, do_design, do_search, create_root_task
from story.project import generate_idea


# --- 数据获取函数 ---

@st.cache_data(ttl=5)
def get_all_books():
    """从 BookMetaDB 获取所有书籍元数据"""
    meta_db = get_meta_db()
    return meta_db.get_all_book_meta()

@st.cache_data(ttl=5)
def get_all_tasks_for_book(run_id: str) -> List[Task]:
    """根据 run_id 从对应的 TaskDB 获取所有任务"""
    if not run_id:
        return []
    task_db = get_task_db(run_id)
    meta_db = get_meta_db()
    
    # 1. 获取书籍的元数据，这是创建完整Task对象所必需的上下文
    book_meta = meta_db.get_book_meta(run_id)
    if not book_meta:
        logger.warning(f"在 get_all_tasks_for_book 中未找到 run_id '{run_id}' 对应的书籍元数据。")
        return []

    # 2. 获取该书籍的所有原始任务数据
    tasks_data = task_db.get_all_tasks()
    
    # 3. 将书籍元数据与每个任务数据合并，然后创建Task对象
    return [dict_to_task({**t, "run_id": run_id}) for t in tasks_data if t] # type: ignore

# --- 任务执行相关 ---

TASK_EXECUTORS: Dict[str, Callable[[Task], Coroutine[Any, Any, Any]]] = {
    "write": do_write,
    "design": do_design,
    "search": do_search,
}

async def do_task(task: Task):
    """根据任务类型分发并执行任务"""
    st.info(f"开始执行任务: {task.id} ({task.task_type})")
    executor = TASK_EXECUTORS.get(task.task_type)
    if not executor:
        st.error(f"未知的任务类型: {task.task_type}")
        return

    await executor(task) # type: ignore
    st.success(f"任务 {task.id} 执行完成！")
    st.rerun() # 刷新页面以显示最新状态

# --- 文本处理工具 ---

def list_to_text(data: List[str]) -> str:
    return "\n".join(data)

def text_to_list(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]

def delete_project(run_id: str, root_name: str):
    """删除整个项目，包括元数据和相关文件。"""
    st.toast(f"正在删除项目: {root_name}...")
    logger.info(f"请求删除项目: {root_name} (run_id: {run_id})")

    meta_db = get_meta_db()
    meta_db.delete_book_meta(run_id)
    logger.info(f"已从 BookMetaDB 删除元数据: {run_id}")

    from utils.file import data_dir
    project_path = data_dir / run_id
    if project_path.exists() and project_path.is_dir():
        shutil.rmtree(project_path)
        logger.info(f"已删除项目文件夹: {project_path}")
    
    st.success(f"项目 {root_name} 已被彻底删除。")

def sync_book_to_task_db(run_id: str):
    """将单个书籍元数据同步到其 TaskDB 创建根任务"""
    st.toast(f"正在同步项目 {run_id}...")
    create_root_task(run_id)
    st.success(f"项目 {run_id} 已同步到任务库！")

# --- 通用UI渲染函数 ---

def _get_all_db_fields() -> List[str]:
    """从 TaskDB 定义中获取所有字段名，用于动态生成表单"""
    # 这些是 Task 模型的核心字段
    task_model_fields = list(Task.model_fields.keys())    
    # 'results' 字段本身是一个容器，但在UI上我们希望它作为一个可编辑的JSON文本区

    # 这些是存储在 results 字典中，但在数据库里是独立列的字段
    result_fields_in_db = [
        "reasoning", 
        "expert", 
        "atom", 
        "atom_reasoning", 
        "plan", 
        "plan_reasoning", 
        "design", 
        "design_reasoning", 
        "search",
        "search_reasoning", 
        "hierarchy", 
        "hierarchy_reasoning", 
        "write",
        "write_reasoning", 
        "summary", 
        "summary_reasoning", 
        "book_level_design",
        "global_state", 
        "write_review", 
        "write_review_reasoning", 
        "translation",
        "translation_reasoning", 
        "context_design", 
        "context_summary",
        "context_search", 
        "kg_design", 
        "kg_write",
        "inquiry_design", 
        "inquiry_summary", 
        "inquiry_search", 
    ]
    
    # 排除一些不应在UI中直接编辑的字段
    excluded_fields = ['run_id', 'root_name', 'category', 'language', 'sub_tasks']
    
    # 定义希望优先显示在表单顶部的核心字段
    primary_fields_order = [
        'id',                     # 任务ID
        'parent_id',              # 父任务ID
        'hierarchical_position',  # 层级位置
        'task_type',              # 任务类型
        'status',                 # 状态
        'goal',                   # 核心目标
        'length',                 # 预估长度
        'instructions',           # 具体指令
        'input_brief',            # 输入指引
        'constraints',            # 限制和禁忌
        'acceptance_criteria',    # 验收标准
        'reasoning',              # 推理过程
        'expert',                 # 执行专家
        'results',                # 剩余结果 (JSON)
    ]
    
    # 1. 获取所有不应被排除的字段，并去重
    all_available_fields = {f for f in (task_model_fields + result_fields_in_db) if f not in excluded_fields}
    
    # 2. 将主要字段按预定顺序排列
    ordered_fields = [f for f in primary_fields_order if f in all_available_fields]
    
    # 3. 获取剩余字段，并按字母顺序排序
    remaining_fields = sorted(list(all_available_fields - set(ordered_fields)))
    
    # 4. 合并列表，返回最终的字段顺序
    return ordered_fields + remaining_fields

def render_task_details_and_actions(task_obj: Task):
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(task_obj.run_id)
    root_name = book_meta.get('root_name', '未知项目') if book_meta else '未知项目'

    st.header("任务详情")
    run_id = task_obj.run_id
    selected_id = task_obj.id

    st.subheader(f"编辑任务: {task_obj.id} ({task_obj.hierarchical_position})")
    st.caption(f"项目: {root_name}")

    # 将操作按钮移动到顶部
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button(f"▶️ 执行此任务 ({task_obj.task_type})", key=f"run_{run_id}_{selected_id}", use_container_width=True):
            asyncio.run(do_task(task_obj))
    with action_cols[1]:
        if st.button(f"🗑️ 删除此任务及子任务", key=f"delete_task_{run_id}_{selected_id}", use_container_width=True, type="primary"):
            get_task_db(run_id).delete_task_and_subtasks(selected_id)
            st.session_state.selected_composite_id = None
            st.rerun()

    with st.form(key=f"form_{run_id}_{selected_id}"):
        all_fields = _get_all_db_fields()
        form_inputs = {}

        # 将 Task 对象和其 results 字典合并，方便统一取值
        full_task_data = task_obj.model_dump()
        full_task_data.update(task_obj.results)

        # 动态生成表单字段
        for field in all_fields:
            value = full_task_data.get(field)
            
            if field == 'id':
                st.text_input(f"任务ID (Id)", value=str(value or ''), key=f"form_{run_id}_{selected_id}_{field}", disabled=True)
            elif field == 'status':
                status_options = ["pending", "running", "completed", "failed", "cancelled", "paused"]
                current_status = value if value in status_options else "pending"
                form_inputs[field] = st.selectbox(f"状态 (Status)", options=status_options, index=status_options.index(current_status), key=f"form_{run_id}_{selected_id}_{field}")
            elif field == 'results':
                # 将非独立列的 results 字典转换为格式化的 JSON 字符串进行显示和编辑
                dedicated_cols = [f for f in _get_all_db_fields() if f != 'results']
                remaining_results = {k: v for k, v in task_obj.results.items() if k not in dedicated_cols}
                json_text = json.dumps(remaining_results, indent=2, ensure_ascii=False)
                form_inputs[field] = st.text_area("剩余结果 (Results JSON)", value=json_text, height=200, key=f"form_{run_id}_{selected_id}_{field}")
            # 优先按字段名判断类型，确保即使值为None也能正确处理
            elif field in ['instructions', 'input_brief', 'constraints', 'acceptance_criteria']:
                text_value = list_to_text(value or [])
                form_inputs[field] = st.text_area(f"{field.replace('_', ' ').title()}", value=text_value, height=100, key=f"form_{run_id}_{selected_id}_{field}")
            elif field in ['plan', 'hierarchy', 'design', 'write', 'summary', 'search', 'reasoning', 'expert',
                           'atom', 'atom_reasoning', 'plan_reasoning', 'design_reasoning', 'search_reasoning',
                           'hierarchy_reasoning', 'write_reasoning', 'summary_reasoning', 'book_level_design',
                           'global_state', 'write_review', 'write_review_reasoning', 'translation', 'translation_reasoning']:
                # 为较长的文本字段提供更大的输入框
                text_value = str(value or '')
                form_inputs[field] = st.text_area(f"{field.replace('_', ' ').title()}", value=text_value, height=200, key=f"form_{run_id}_{selected_id}_{field}")
            else:
                # 默认使用单行输入框
                form_inputs[field] = st.text_input(f"{field.replace('_', ' ').title()}", value=str(value or ''), key=f"form_{run_id}_{selected_id}_{field}")

        submitted = st.form_submit_button("💾 保存修改")
        if submitted:
            try:
                # 从表单回填数据到 Task 对象
                for field, new_value in form_inputs.items():
                    original_value = full_task_data.get(field)

                    if field == 'id': # id 是只读的，跳过
                        continue
                    
                    # 根据原始数据类型转换新值
                    if field in ['instructions', 'input_brief', 'constraints', 'acceptance_criteria']:
                        setattr(task_obj, field, text_to_list(new_value)) # type: ignore
                    elif field == 'results':
                        # 对于 'results' 字段，我们需要解析JSON并更新到 task_obj.results
                        try:
                            updated_remaining_results = json.loads(new_value)
                            task_obj.results.update(updated_remaining_results)
                        except json.JSONDecodeError:
                            st.error("“剩余结果 (Results JSON)” 字段中的JSON格式无效，请检查。")
                            return # 阻止保存
                    elif field in Task.model_fields:
                        # 确保将表单输入作为字符串处理
                        # 处理 Task 模型的直接字段
                        setattr(task_obj, field, new_value)
                    else:
                        # 处理存储在 results 中的字段
                        task_obj.results[field] = new_value

                task_db = get_task_db(run_id)
                task_db.add_task(task_obj)
                task_db.add_result(task_obj) # 确保 results 中的字段也被保存
                st.success(f"任务 {selected_id} 已成功保存！")
                st.rerun()
            except json.JSONDecodeError:
                st.error("某个JSON字段的格式无效，请检查后重新保存。")
            except Exception as e:
                st.error(f"保存失败: {e}")