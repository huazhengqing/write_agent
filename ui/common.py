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
from story.do_story import sync_meta_to_task_db
from story.do_task import do_write, do_design, do_search


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
    return [dict_to_task({**book_meta, **t}) for t in tasks_data if t] # type: ignore

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

# --- 项目操作函数 ---

def create_cyberpunk_test_data():
    """
    在UI中直接创建一套完整的赛博朋克小说测试数据。
    """
    st.toast("开始创建《赛博朋克：迷雾之城》测试数据...")
    logger.info("开始创建《赛博朋克：迷雾之城》测试数据...")

    book_info = {
        'category': "story", 'language': "cn", 'name': "赛博朋克：迷雾之城",
        'goal': "创作一部赛博朋克侦探小说，主角在反乌托邦的未来城市中调查一宗神秘的失踪案。",
        'instructions': "故事应充满霓虹灯、雨夜、高科技与社会底层挣扎的元素。主角需要有鲜明的个性和过去。",
        'input_brief': "城市名为“夜之城”，被巨型企业“荒坂公司”所控制。主角是一名被解雇的前企业特工。",
        'constraints': "避免魔法或超自然元素，所有科技都应有合理的解释。",
        'acceptance_criteria': "完成开篇三章，揭示案件的初步线索，并塑造主角的困境。",
        'length': "约2万字", 'day_wordcount_goal': 500
    }

    meta_db = get_meta_db()
    meta_db.add_book(book_info)
    
    all_books = meta_db.get_all_book_meta()
    cyberpunk_book = next((b for b in all_books if b['root_name'] == book_info['name']), None)
    if not cyberpunk_book:
        st.error("创建书籍元数据后未能找到，测试数据生成失败！")
        return

    run_id = cyberpunk_book['run_id']
    logger.info(f"获取到书籍的 run_id: {run_id}")
    
    meta_db.update_book_level_design(run_id, "全书设计：采用三幕式结构，第一幕引入主角和案件，第二幕深入调查并遭遇挫折，第三幕揭开真相并与反派对决。")
    meta_db.update_global_state_summary(run_id, "全局状态：主角“杰克”已被“荒坂公司”解雇，身无分文。他刚接手寻找失踪数据分析师“伊芙”的委托。")

    task_db = get_task_db(run_id)
    
    def create_task(task_id, parent_id, task_type, goal, hierarchical_pos, status="pending", results=None):
        return Task(
            id=task_id, parent_id=parent_id, task_type=task_type, goal=goal,
            hierarchical_position=hierarchical_pos, status=status, results=results or {},
            category=book_info['category'], language=book_info['language'],
            root_name=book_info['name'], run_id=run_id
        )

    tasks_to_add = [
        create_task("1", "", "write", book_info['goal'], "全书", status="pending"),
        create_task("1.1", "1", "design", "设计小说第一章的详细情节", "第一章", status="completed", results={"design": "第一章情节：杰克在破旧的公寓中被神秘客户联系，接下寻找伊芙的委托。他前往伊芙最后出现的酒吧进行调查。"}),
        create_task("1.2", "1", "write", "撰写第一章的全部内容", "第一章", status="running"),
        create_task("1.3", "1", "design", "设计第二章的核心悬念", "第二章", status="pending"),
        create_task("1.2.1", "1.2", "search", "搜索关于“未来城市酒吧”的描写和氛围资料", "第一章-场景1", status="completed", results={"search": "参考资料：银翼杀手、攻壳机动队中的酒吧场景，特点是全息广告、合成酒精、各类改造人顾客。"}),
        create_task("1.2.2", "1.2", "write", "撰写杰克进入酒吧并与酒保交谈的场景", "第一章-场景1", "pending"),
        create_task("1.2.3", "1.2", "write", "撰写杰克发现伊芙留下的加密数据棒的场景", "第一章-场景2", status="pending"),
    ]

    for task in tasks_to_add:
        task_db.add_task(task)
        if task.results:
            task_db.add_result(task)

    st.success("《赛博朋克：迷雾之城》测试数据创建成功！")
    logger.success("测试数据创建完成！")

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
    sync_meta_to_task_db(run_id)
    st.success(f"项目 {run_id} 已同步到任务库！")

# --- 通用UI渲染函数 ---

def _get_all_db_fields() -> List[str]:
    """从 TaskDB 定义中获取所有字段名，用于动态生成表单"""
    # 这些是 Task 模型的核心字段
    task_model_fields = list(Task.model_fields.keys())
    # 移除 results，因为它是一个容器
    task_model_fields.remove('results')
    
    # 这些是存储在 results 字典中，但在数据库里是独立列的字段
    result_fields_in_db = [
        "reasoning", 
        "expert", 
        "atom", 
        "atom_reasoning", 
        "atom_result",
        "plan", "plan_reasoning", 
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
        "context_task", 
        "context_search", 
        "kg_design", 
        "kg_write",
        "rag_design", 
        "rag_summary", 
        "x_litellm_cache_key"
    ]
    
    # 排除一些不应在UI中直接编辑的字段
    excluded_fields = ['run_id', 'root_name', 'category', 'language', 'sub_tasks']
    
    # 定义希望优先显示在表单顶部的核心字段
    primary_fields_order = [
        'goal',
        'hierarchical_position',
        'task_type',
        'status',
        'parent_id',
        'length',
        'instructions',
        'input_brief',
        'constraints',
        'acceptance_criteria',
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
    st.header("任务详情")
    run_id = task_obj.run_id
    selected_id = task_obj.id

    st.subheader(f"编辑任务: {task_obj.id} ({task_obj.hierarchical_position})")
    st.caption(f"项目: {task_obj.root_name}")

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
            
            if field == 'status':
                status_options = ["pending", "running", "completed", "failed", "cancelled", "paused"]
                form_inputs[field] = st.selectbox(f"状态 (Status)", options=status_options, index=status_options.index(value or "pending"), key=f"form_{run_id}_{selected_id}_{field}")
            elif isinstance(value, list):
                form_inputs[field] = st.text_area(f"{field.replace('_', ' ').title()}", value=list_to_text(value), height=100, key=f"form_{run_id}_{selected_id}_{field}")
            elif isinstance(value, dict):
                 form_inputs[field] = st.text_area(f"{field.replace('_', ' ').title()} (JSON)", value=json.dumps(value, indent=2, ensure_ascii=False), height=150, key=f"form_{run_id}_{selected_id}_{field}")
            elif field in ['design', 'write', 'summary', 'search', 'plan', 'hierarchy', 'atom', 'reasoning'] or 'reasoning' in field:
                # 为较长的文本字段提供更大的输入框
                form_inputs[field] = st.text_area(f"{field.replace('_', ' ').title()}", value=str(value or ''), height=200, key=f"form_{run_id}_{selected_id}_{field}")
            else:
                form_inputs[field] = st.text_input(f"{field.replace('_', ' ').title()}", value=str(value or ''), key=f"form_{run_id}_{selected_id}_{field}")

        submitted = st.form_submit_button("💾 保存修改")
        if submitted:
            try:
                # 从表单回填数据到 Task 对象
                for field, new_value in form_inputs.items():
                    original_value = full_task_data.get(field)
                    
                    # 根据原始数据类型转换新值
                    if isinstance(original_value, list):
                        setattr(task_obj, field, text_to_list(new_value))
                    elif isinstance(original_value, dict):
                        setattr(task_obj, field, json.loads(new_value))
                    elif field in Task.model_fields:
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