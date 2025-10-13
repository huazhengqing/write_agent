import streamlit as st
import os
import sys

# --- 项目根目录设置 ---
# 确保在 Home.py 中也进行路径设置，因为Streamlit可能从任何页面启动
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    st.set_page_config(layout="wide", page_title="AI 写作智能体")
    st.title("欢迎使用 AI 写作智能体 🚀")

    st.markdown("""
    这是一个用于管理和监控 AI 驱动的写作项目的仪表盘。

    **请使用左侧的侧边栏导航到不同的功能页面：**

    - **📚 项目管理**: 在这里创建、查看和管理你的写作项目。
    - **📝 任务工作台**: 在这里可视化任务流程、执行任务并查看细节。

    祝你创作愉快！
    """)

if __name__ == "__main__":
    main()