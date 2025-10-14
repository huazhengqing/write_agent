import streamlit as st
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
    sync_book_to_task_db,
    delete_project,
    get_meta_db,
    generate_idea
)

def render_project_management_page():
    """渲染项目管理页面"""
    st.header("📚 项目管理")

    # 初始化 session_state 用于控制编辑对话框
    if "editing_run_id" not in st.session_state:
        st.session_state.editing_run_id = None
    # 初始化 session_state 用于存储AI生成的项目点子
    if "generated_idea" not in st.session_state:
        st.session_state.generated_idea = {}

    with st.expander("➕ 创建新项目", expanded=False):
        with st.form("new_project_form"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("填写新项目信息")
            with col2:
                if st.form_submit_button("🤖 自动生成点子", use_container_width=True):
                    with st.spinner("正在向AI请求创意..."):
                        idea = generate_idea()
                        if idea:
                            st.session_state.generated_idea = idea.model_dump()
                            st.rerun()
                        else:
                            st.error("无法生成项目点子，请稍后重试。")

            # 使用 session_state 中的值作为默认值
            name = st.text_input("书名/项目名", st.session_state.generated_idea.get("name", "例如：赛博朋克：迷雾之城"))
            goal = st.text_area("核心目标", st.session_state.generated_idea.get("goal", "创作一部赛博朋克侦探小说..."))
            category = st.selectbox("类别", ["story", "report", "book"], index=0)
            language = st.selectbox("语言", ["cn", "en"], index=0)
            instructions = st.text_area("具体指令", st.session_state.generated_idea.get("instructions", "故事应充满霓虹灯、雨夜..."))
            input_brief = st.text_area("输入指引", st.session_state.generated_idea.get("input_brief", "城市名为“夜之城”..."))
            constraints = st.text_area("限制和禁忌", st.session_state.generated_idea.get("constraints", "避免魔法或超自然元素..."))
            acceptance_criteria = st.text_area("验收标准", st.session_state.generated_idea.get("acceptance_criteria", "完成开篇三章..."))
            length = st.text_input("预估总字数", "约100万字")

            submitted = st.form_submit_button("创建项目")
            if submitted:
                if not name or not goal:
                    st.error("书名和核心目标不能为空！")
                else:
                    book_info = {
                        'category': category, 'language': language, 'name': name,
                        'goal': goal, 'instructions': instructions, 'input_brief': input_brief,
                        'constraints': constraints, 'acceptance_criteria': acceptance_criteria,
                        'length': length
                    }
                    meta_db = get_meta_db()
                    meta_db.add_book(book_info)
                    st.success(f"项目《{name}》已成功创建！")
                    st.session_state.generated_idea = {} # 清空已生成的点子
                    st.rerun()

    st.divider()

    st.subheader("项目列表")
    all_books = get_all_books()
    if not all_books:
        st.info("当前没有项目。请在上方创建新项目或生成测试数据。")
    else:
        for book in all_books:
            # --- 编辑项目的对话框逻辑 ---
            if st.session_state.editing_run_id == book["run_id"]:
                with st.dialog("编辑项目信息"):
                    with st.form(f"edit_form_{book['run_id']}"):
                        st.subheader(f"正在编辑: {book['root_name']}")
                        
                        # 注意：项目名称和目标在创建后通常不建议修改，因为 run_id 与之关联
                        # 如果要修改，需要有更复杂的逻辑来处理文件目录重命名等问题，此处设为只读
                        name = st.text_input("书名/项目名 (只读)", value=book.get('root_name', ''), disabled=True)
                        goal = st.text_area("核心目标 (只读)", value=book.get('goal', ''), disabled=True)

                        category = st.selectbox("类别", ["story", "report", "book"], index=["story", "report", "book"].index(book.get('category', 'story')))
                        language = st.selectbox("语言", ["cn", "en"], index=["cn", "en"].index(book.get('language', 'cn')))
                        instructions = st.text_area("具体指令", value=book.get('instructions', ''))
                        input_brief = st.text_area("输入指引", value=book.get('input_brief', ''))
                        constraints = st.text_area("限制和禁忌", value=book.get('constraints', ''))
                        acceptance_criteria = st.text_area("验收标准", value=book.get('acceptance_criteria', ''))
                        length = st.text_input("预估总字数", value=book.get('length', ''))
                        day_wordcount_goal = st.number_input("每日字数目标", value=book.get('day_wordcount_goal', 0))

                        if st.form_submit_button("💾 保存修改"):
                            updated_book_info = {
                                'run_id': book['run_id'], # 传入 run_id 以便执行更新
                                'name': name, # 即使是 disabled，也需要传回
                                'goal': goal, # 即使是 disabled，也需要传回
                                'category': category, 'language': language, 
                                'instructions': instructions, 'input_brief': input_brief,
                                'constraints': constraints, 'acceptance_criteria': acceptance_criteria,
                                'length': length, 'day_wordcount_goal': day_wordcount_goal
                            }
                            meta_db = get_meta_db()
                            meta_db.add_book(updated_book_info)
                            st.success(f"项目《{name}》已成功更新！")
                            st.session_state.editing_run_id = None # 关闭对话框
                            st.rerun()


            # --- 项目列表展示 ---
            with st.expander(f"**{book['root_name']}** (ID: {book['run_id']})"):
                # 使用列来布局
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.text(f"类别: {book.get('category', 'N/A')} | 语言: {book.get('language', 'N/A')} | 预估总字数: {book.get('length', 'N/A')} | 每日字数目标: {book.get('day_wordcount_goal', 'N/A')}")
                    st.divider()
                    st.markdown(f"**标题 (Title):** {book.get('title') or '暂无'}")
                    st.markdown(f"**简介 (Synopsis):**\n> {book.get('synopsis') or '暂无'}")
                    st.markdown(f"**风格 (Style):**\n> {book.get('style') or '暂无'}")
                    st.divider()
                    st.markdown(f"**核心目标 (Goal):**\n> {book.get('goal', 'N/A')}")
                    with st.expander("查看任务定义 (Instructions, Constraints, etc.)"):
                        st.markdown(f"**具体指令 (Instructions):**\n> {book.get('instructions', 'N/A')}")
                        st.markdown(f"**输入指引 (Input Brief):**\n> {book.get('input_brief', 'N/A')}")
                        st.markdown(f"**限制和禁忌 (Constraints):**\n> {book.get('constraints', 'N/A')}")
                        st.markdown(f"**验收标准 (Acceptance Criteria):**\n> {book.get('acceptance_criteria', 'N/A')}")
                    st.divider()
                    st.markdown(f"**全书设计 (Book Level Design):**\n> {book.get('book_level_design') or '暂无'}")
                    st.markdown(f"**全局状态 (Global State):**\n> {book.get('global_state_summary') or '暂无'}")

                with col2:
                    st.button("🔄 同步到任务库", key=f"sync_{book['run_id']}", on_click=sync_book_to_task_db, args=(book['run_id'],), use_container_width=True)
                    if st.button("📝 编辑项目", key=f"edit_{book['run_id']}", use_container_width=True):
                        # 点击编辑按钮时，设置 session_state 并重新运行以打开对话框
                        st.session_state.editing_run_id = book['run_id']
                        st.rerun()

                    st.button("🗑️ 删除此项目", key=f"delete_{book['run_id']}", on_click=delete_project, args=(book['run_id'], book['root_name']), use_container_width=True, type="secondary")

st.set_page_config(layout="wide", page_title="项目管理")
render_project_management_page()