import streamlit as st
import json
import os
import sys
import asyncio
from typing import List, Dict, Any, Callable, Coroutine
from streamlit_agraph import agraph, Node, Edge, Config
from loguru import logger
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.models import Task
from utils.sqlite_meta import get_meta_db, BookMetaDB
from utils.sqlite_task import get_task_db, dict_to_task
from story.do_story import add_sample_story_task_to_meta_db, sync_meta_to_task_db
from story.do_task import do_write, do_design, do_search


@st.cache_data(ttl=5)
def get_all_books():
    """从 BookMetaDB 获取所有书籍元数据"""
    meta_db = get_meta_db()
    return meta_db.get_all_book_meta()

@st.cache_data(ttl=5)
def get_all_tasks_for_book(run_id: str) -> List[Dict[str, Any]]:
    """根据 run_id 从对应的 TaskDB 获取所有任务"""
    if not run_id:
        return []
    task_db = get_task_db(run_id)
    return task_db.get_all_tasks()

# 任务执行器分发字典
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

def list_to_text(data: List[str]) -> str:
    return "\n".join(data)

def text_to_list(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]

def create_cyberpunk_test_data():
    """
    在UI中直接创建一套完整的赛博朋克小说测试数据。
    """
    st.toast("开始创建《赛博朋克：迷雾之城》测试数据...")
    logger.info("开始创建《赛博朋克：迷雾之城》测试数据...")

    # 1. 定义书籍元数据
    book_info = {
        'category': "story", 'language': "cn", 'name': "赛博朋克：迷雾之城",
        'goal': "创作一部赛博朋克侦探小说，主角在反乌托邦的未来城市中调查一宗神秘的失踪案。",
        'instructions': "故事应充满霓虹灯、雨夜、高科技与社会底层挣扎的元素。主角需要有鲜明的个性和过去。",
        'input_brief': "城市名为“夜之城”，被巨型企业“荒坂公司”所控制。主角是一名被解雇的前企业特工。",
        'constraints': "避免魔法或超自然元素，所有科技都应有合理的解释。",
        'acceptance_criteria': "完成开篇三章，揭示案件的初步线索，并塑造主角的困境。",
        'length': "约2万字", 'day_wordcount_goal': 500
    }

    # 2. 添加到 BookMetaDB
    meta_db = get_meta_db()
    meta_db.add_book(book_info)
    
    # 查找刚创建的书籍以获取 run_id
    all_books = meta_db.get_all_book_meta()
    cyberpunk_book = next((b for b in all_books if b['root_name'] == book_info['name']), None)
    if not cyberpunk_book:
        st.error("创建书籍元数据后未能找到，测试数据生成失败！")
        return

    run_id = cyberpunk_book['run_id']
    logger.info(f"获取到书籍的 run_id: {run_id}")
    
    # 更新全局信息
    meta_db.update_book_level_design(run_id, "全书设计：采用三幕式结构，第一幕引入主角和案件，第二幕深入调查并遭遇挫折，第三幕揭开真相并与反派对决。")
    meta_db.update_global_state_summary(run_id, "全局状态：主角“杰克”已被“荒坂公司”解雇，身无分文。他刚接手寻找失踪数据分析师“伊芙”的委托。")

    # 3. 在 TaskDB 中创建树状任务
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

    # 1. 删除元数据
    meta_db = get_meta_db()
    meta_db.delete_book_meta(run_id)
    logger.info(f"已从 BookMetaDB 删除元数据: {run_id}")

    # 2. 删除项目文件夹
    from utils.file import data_dir
    project_path = data_dir / run_id
    if project_path.exists() and project_path.is_dir():
        shutil.rmtree(project_path)
        logger.info(f"已删除项目文件夹: {project_path}")
    
    st.success(f"项目 {root_name} 已被彻底删除。")

st.set_page_config(layout="wide")
st.title("📚 AI 写作智能体监控面板")

# --- 布局定义 ---
col_left, col_main, col_right = st.columns([1, 2.5, 1.5])

# --- 1. 左侧边栏：项目选择区 ---
with col_left:
    st.header("项目列表")

    if st.button("➕ 新建示例小说", use_container_width=True):
        add_sample_story_task_to_meta_db()
        st.success("已添加示例小说《龙与魔法之歌》！")
        st.rerun()

    if st.button("🤖 创建赛博朋克测试数据", use_container_width=True):
        create_cyberpunk_test_data()
        st.rerun()

    if st.button("🔄 同步元数据到任务库", use_container_width=True):
        sync_meta_to_task_db()
        st.success("同步完成！")
        st.rerun()

    st.divider()

    all_books = get_all_books()
    if not all_books:
        st.info("当前没有项目。请新建或同步。")
    else:
        # 为每个项目创建一个条目，包含选择和删除按钮
        for book in all_books:
            book_run_id = book['run_id']
            book_name = book['root_name']
            
            row = st.columns([4, 1])
            with row[0]:
                if st.button(book_name, key=f"select_{book_run_id}", use_container_width=True):
                    st.session_state.selected_run_id = book_run_id
                    st.session_state.selected_task_id = None # 切换项目时清空任务选择
                    st.rerun()
            with row[1]:
                if st.button("🗑️", key=f"delete_{book_run_id}", help=f"删除项目: {book_name}"):
                    delete_project(book_run_id, book_name)
                    st.session_state.selected_run_id = None # 清空选择
                    st.rerun()

# --- 2. 中间主区域 & 3. 右侧边栏 ---
run_id = st.session_state.get('selected_run_id')
if run_id:
    tasks = get_all_tasks_for_book(run_id)
    task_map = {task['id']: task for task in tasks}

    with col_main:
        # --- 2.1 上部：任务流程图 ---
        st.header("任务流程图")
        if not tasks:
            st.warning("此项目还没有任何任务。请先同步元数据。")
        else:
            nodes = []
            edges = []
            status_colors = {
                "completed": "#90EE90",  # LightGreen
                "running": "#FFA500",    # Orange
                "pending": "#D3D3D3",    # LightGray
                "failed": "#FF474C",      # Red
                "cancelled": "#808080",   # Gray
                "paused": "#ADD8E6",      # LightBlue
            }
            for task_id, task in task_map.items():
                nodes.append(Node(id=task_id,
                                  label=f"ID: {task_id}\n{task.get('hierarchical_position', '')}",
                                  shape='box',
                                  color=status_colors.get(task.get('status', 'pending'), '#D3D3D3')))
                if task.get('parent_id') and task.get('parent_id') in task_map:
                    edges.append(Edge(source=task['parent_id'], target=task_id))

            config = Config(width='100%',
                            height=400,
                            directed=True,
                            physics=False,
                            hierarchical=True,
                            node={'font': {'size': 12}})
            
            clicked_node_id = agraph(nodes=nodes, edges=edges, config=config)

            if clicked_node_id:
                st.session_state.selected_task_id = clicked_node_id
            
            if st.button("刷新任务树"):
                st.rerun()

        # --- 2.2 下部：任务详情与编辑区 ---
        st.header("任务详情")
        selected_id = st.session_state.get('selected_task_id')

        if not selected_id:
            st.info("请在上面的任务图中点击一个节点以查看和编辑详情。")
        elif selected_id not in task_map:
            st.error(f"任务 {selected_id} 未找到，请刷新。")
            st.session_state.selected_task_id = None
        else:
            task_data = task_map[selected_id]
            task_obj = dict_to_task(task_data)

            # 使用表单来收集所有修改
            with st.form(key=f"form_{selected_id}"):
                st.subheader(f"编辑任务: {task_obj.id} ({task_obj.hierarchical_position})")

                # 核心信息
                status_options = ["pending", "running", "completed", "failed", "cancelled", "paused"]
                goal = st.text_input("目标 (Goal)", value=task_obj.goal)
                status = st.selectbox("状态 (Status)", options=status_options, index=status_options.index(task_obj.status))
                instructions = st.text_area("指令 (Instructions)", value=list_to_text(task_obj.instructions), height=150)
                input_brief = st.text_area("输入指引 (Input Brief)", value=list_to_text(task_obj.input_brief), height=100)
                
                # 结果展示
                with st.expander("查看/编辑产出结果 (Results)", expanded=False):
                    results_text = st.text_area("JSON 格式的结果", value=json.dumps(task_obj.results, indent=2, ensure_ascii=False), height=300)

                # 提交按钮
                submitted = st.form_submit_button("💾 保存修改")
                if submitted:
                    try:
                        # 更新 Task 对象
                        task_obj.goal = goal
                        task_obj.status = status # type: ignore
                        task_obj.instructions = text_to_list(instructions)
                        task_obj.input_brief = text_to_list(input_brief)
                        task_obj.results = json.loads(results_text)

                        # 写回数据库
                        task_db = get_task_db(run_id)
                        task_db.add_task(task_obj) # add_task 具有更新功能
                        st.success(f"任务 {task_obj.id} 已成功保存！")
                        st.rerun()
                    except json.JSONDecodeError:
                        st.error("结果(Results)中的JSON格式无效，请检查后重新保存。")
                    except Exception as e:
                        st.error(f"保存失败: {e}")

            # --- 操作按钮（表单外）---
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button(f"▶️ 执行此任务 ({task_obj.task_type})", key=f"run_{selected_id}", use_container_width=True):
                    asyncio.run(do_task(task_obj))
            with action_cols[1]:
                if st.button(f"🗑️ 删除此任务及子任务", key=f"delete_task_{selected_id}", use_container_width=True, type="secondary"):
                    get_task_db(run_id).delete_task_and_subtasks(selected_id)
                    st.session_state.selected_task_id = None # 清空选择
                    st.rerun()

    with col_right:
        # --- 3. 右侧边栏：全局信息区 ---
        st.header("全局信息")
        meta_db = get_meta_db()
        book_meta = meta_db.get_book_meta(run_id)
        if book_meta:
            st.subheader("全书设计")
            st.text_area(
                "Book Level Design", 
                value=book_meta.get("book_level_design", "暂无"), 
                height=200, 
                disabled=True
            )
            st.subheader("全局状态摘要")
            st.text_area(
                "Global State Summary", 
                value=book_meta.get("global_state_summary", "暂无"), 
                height=200, 
                disabled=True
            )
        else:
            st.info("未找到本书的全局信息。")

else:
    with col_main:
        st.info("👈 请在左侧的项目列表中选择一个项目开始。")