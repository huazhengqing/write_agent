import streamlit as st
from typing import List, Dict, Optional
import os
import sys

# --- 项目根目录设置 ---
# 确保项目根目录在sys.path中，以便正确导入模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入共享模块
from ui.common import (
    get_all_books,
    get_all_tasks_for_book,
    render_task_details_and_actions,
    Task,
    natural_sort_key
)

def render_task_tree(tasks: List[Task], run_id: str, book_name: str):
    """递归渲染任务树"""
    tasks_by_id = {task.id: task for task in tasks}
    children_by_parent_id: Dict[str, List[Task]] = {}
    root_tasks: List[Task] = []

    for task in tasks:
        if task.parent_id and task.parent_id in tasks_by_id:
            if task.parent_id not in children_by_parent_id:
                children_by_parent_id[task.parent_id] = []
            children_by_parent_id[task.parent_id].append(task)
        else:
            root_tasks.append(task)

    # 按任务ID自然排序
    for children in children_by_parent_id.values():
        children.sort(key=lambda t: natural_sort_key(t.id))
    root_tasks.sort(key=lambda t: natural_sort_key(t.id))

    st.subheader(f"项目: {book_name}")
    for task in root_tasks:
        _render_tree_node(task, children_by_parent_id, run_id, level=0)

def toggle_expand(node_id):
    """切换节点的展开/折叠状态"""
    st.session_state.expanded_nodes.symmetric_difference_update([node_id])

def select_task(composite_id):
    """选中任务的回调函数"""
    st.session_state.selected_composite_id = composite_id


def render_task_workspace_page():
    """渲染任务工作台页面"""
    st.header("📝 任务工作台")

    all_books = get_all_books()
    if not all_books:
        st.info("没有可用的项目。请先在'项目管理'页面创建项目。")
        return

    # 初始化 session_state 来存储展开的节点
    if "expanded_nodes" not in st.session_state:
        st.session_state.expanded_nodes = set()
    # 初始化 session_state 来存储选中的任务
    if "selected_composite_id" not in st.session_state:
        st.session_state.selected_composite_id = None


    # --- 布局定义 ---
    # 调整列宽比例，为右侧详情区域分配更多空间
    col_main, col_right = st.columns([1.5, 2.5])

    # --- 侧边栏项目选择 ---
    book_options = {book['run_id']: book['root_name'] for book in all_books}
    book_options_with_all = {"all": "所有项目", **book_options}

    # Streamlit多页面应用会自动在侧边栏创建导航，这里我们添加一个项目选择器
    st.sidebar.divider()
    selected_run_id_view = st.sidebar.selectbox(
        "选择要查看的项目",
        options=list(book_options_with_all.keys()),
        format_func=lambda x: book_options_with_all[x],
        key="selected_run_id_view"
    )

    # --- 数据准备 ---
    all_tasks_by_book: Dict[str, List[Task]] = {}
    if selected_run_id_view == "all":
        for run_id in book_options.keys():
            all_tasks_by_book[run_id] = get_all_tasks_for_book(run_id)
    else:
        all_tasks_by_book[selected_run_id_view] = get_all_tasks_for_book(selected_run_id_view)

    # --- 主区域：任务树 ---
    with col_main:
        # 注入自定义CSS以实现紧凑和左对齐的布局
        # 通过设置 gap: 0 !important; 来移除垂直元素间的默认间距，使行高更紧凑
        st.markdown("""
        <style>
            .st-emotion-cache-1r4qj8v { /* 主容器 */
                gap: 0rem !important;
            }
            /* 任务项容器 */
            .task-item-container {
                display: flex;
                align-items: center;
                padding: 0.15rem 0.2rem;
                border-radius: 0.25rem;
                transition: background-color 0.2s;
            }
            .task-item-container:hover {
                background-color: #f0f2f6;
            }
            .task-item-container.selected {
                background-color: #e0e7ff;
            }
            .task-item-container button {
                justify-content: flex-start;
                padding: 0.1rem 0.3rem;
                margin: 0;
                font-size: 0.9rem;
                border: none;
                background-color: transparent;
                width: 100%;
                text-align: left;
                line-height: 1.4;
            }
            .expand-icon {
                cursor: pointer;
                font-size: 0.8rem;
                width: 1.2rem;
                text-align: center;
                color: #555;
            }
            .expand-icon.placeholder {
                color: transparent; /* Make placeholder invisible but take space */
            }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("任务树状图")
        has_tasks = any(all_tasks_by_book.values())

        if not has_tasks:
            st.warning("所选项目还没有任何任务。请在'项目管理'页面同步项目。")
        else:
            # 为每个项目渲染一个任务树
            for run_id, tasks in all_tasks_by_book.items():
                if not tasks:
                    continue
                book_name = book_options.get(run_id, run_id[:8])
                render_task_tree(tasks, run_id, book_name)
                st.divider()

        if st.button("🔄 刷新任务树"):
            st.rerun()

    # --- 右侧区域：任务详情与全局信息 ---
    with col_right:
        # 将所有任务收集到一个map中，以便通过 composite_id 快速查找
        task_map: Dict[str, Task] = {
            f"{run_id}::{task.id}": task
            for run_id, tasks in all_tasks_by_book.items()
            for task in tasks
        }

        selected_composite_id = st.session_state.get('selected_composite_id')

        if not selected_composite_id or selected_composite_id not in task_map:
            st.info("在左侧任务树中点击一个任务以查看和编辑详情。")
            # 全局信息区域已根据要求移除
        else:
            task_obj = task_map[selected_composite_id]
            render_task_details_and_actions(task_obj)

def _render_tree_node(task: Task, children_map: Dict[str, List[Task]], run_id: str, level: int):
    """渲染树的单个节点"""
    composite_id = f"{run_id}::{task.id}"
    
    # 使用任务状态来决定图标
    status_icon_map = {"completed": "✅", "running": "⏳", "failed": "❌", "pending": "📄", "cancelled": "⏹️", "paused": "⏸️"}
    icon = status_icon_map.get(task.status, "📝")
    # 根据你的要求，加入字数信息
    length_str = f"({task.length}字)" if task.length else ""
    label_text = f"{icon} {task.id} [{task.task_type}] {length_str} - {task.goal or '未命名任务'}"
    
    has_children = task.id in children_map and children_map[task.id]
    is_expanded = task.id in st.session_state.expanded_nodes
    is_selected = st.session_state.selected_composite_id == composite_id

    # 使用列来布局：缩进、展开图标、任务按钮
    cols = st.columns([level * 0.05 + 0.01, 0.1, 2]) # 动态缩进, 图标, 按钮
    cols[0].write("") # 仅用于占位实现缩进

    with cols[1]:
        if has_children:
            expand_icon = "▼" if is_expanded else "▶"
            if st.button(expand_icon, key=f"expand_{composite_id}", use_container_width=True):
                toggle_expand(task.id)
                st.rerun()
        else:
            st.write('<div class="expand-icon placeholder"></div>', unsafe_allow_html=True)

    with cols[2]:
        container_class = "task-item-container selected" if is_selected else "task-item-container"
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        if st.button(label_text, key=f"btn_{composite_id}", use_container_width=True):
            select_task(composite_id)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if has_children and is_expanded:
        for child in children_map.get(task.id, []):
            _render_tree_node(child, children_map, run_id, level + 1)

st.set_page_config(layout="wide", page_title="任务工作台")
render_task_workspace_page()