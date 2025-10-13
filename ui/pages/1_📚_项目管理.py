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
    create_cyberpunk_test_data,
    sync_book_to_task_db,
    delete_project,
    get_meta_db
)

def render_project_management_page():
    """渲染项目管理页面"""
    st.header("📚 项目管理")

    with st.expander("➕ 创建新项目", expanded=False):
        with st.form("new_project_form"):
            st.subheader("填写新项目信息")
            name = st.text_input("书名/项目名", "例如：赛博朋克：迷雾之城")
            goal = st.text_area("核心目标", "例如：创作一部赛博朋克侦探小说...")
            category = st.selectbox("类别", ["story", "report", "book"], index=0)
            language = st.selectbox("语言", ["cn", "en"], index=0)
            instructions = st.text_area("具体指令", "例如：故事应充满霓虹灯、雨夜...")
            input_brief = st.text_area("输入指引", "例如：城市名为“夜之城”...")
            constraints = st.text_area("限制和禁忌", "例如：避免魔法或超自然元素...")
            acceptance_criteria = st.text_area("验收标准", "例如：完成开篇三章...")
            length = st.text_input("预估总字数", "约2万字")
            day_wordcount_goal = st.number_input("每日字数目标", 500)

            submitted = st.form_submit_button("创建项目")
            if submitted:
                if not name or not goal:
                    st.error("书名和核心目标不能为空！")
                else:
                    book_info = {
                        'category': category, 'language': language, 'name': name,
                        'goal': goal, 'instructions': instructions, 'input_brief': input_brief,
                        'constraints': constraints, 'acceptance_criteria': acceptance_criteria,
                        'length': length, 'day_wordcount_goal': day_wordcount_goal
                    }
                    meta_db = get_meta_db()
                    meta_db.add_book(book_info)
                    st.success(f"项目《{name}》已成功创建！")
                    st.rerun()

    st.divider()

    st.subheader("项目列表")
    if st.button("🤖 创建赛博朋克测试数据", use_container_width=True):
        create_cyberpunk_test_data()
        st.rerun()

    all_books = get_all_books()
    if not all_books:
        st.info("当前没有项目。请在上方创建新项目或生成测试数据。")
    else:
        for book in all_books:
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
                    st.button("🗑️ 删除此项目", key=f"delete_{book['run_id']}", on_click=delete_project, args=(book['run_id'], book['root_name']), use_container_width=True, type="secondary")

st.set_page_config(layout="wide", page_title="项目管理")
render_project_management_page()