import streamlit as st
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.common import (
    get_all_books,
    sync_book_to_task_db,
    delete_project,
    get_meta_db,
    generate_idea,
    st_autoresize_text_area  # 导入新的自动高度组件
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
                if st.form_submit_button("🤖 生成创意", use_container_width=True):
                    with st.spinner("正在向AI请求创意..."):
                        idea = generate_idea()
                        if idea:
                            st.session_state.generated_idea = idea.model_dump()
                            st.rerun()
                        else:
                            st.error("无法生成创意，请稍后重试。")

            # 使用 session_state 中的值作为默认值
            name = st.text_input("书名/项目名", st.session_state.generated_idea.get("name", "我靠美食修炼，竟成了国运之神"))
            goal = st_autoresize_text_area("核心目标", st.session_state.generated_idea.get("goal", "在一个美食能觉醒‘食灵’、影响国运的现代都市，一个背负家族衰落秘密的青年厨师，意外激活了能‘解析万物’的神秘系统。他从一道家传菜开始，通过烹饪极致美味，唤醒沉睡的食灵，积累美食气运，兑换超凡能力。他不仅要重振家族荣耀，还要在各国美食世家、神秘组织和国家力量的博弈中，揭开导致美食文明断代的历史真相，最终以美食之道，守护国运，登顶世界之巅。"), key="new_goal")
            category = st.selectbox("类别", ["story", "report", "book"], index=0)
            language = st.selectbox("语言", ["cn", "en"], index=0)
            instructions = st_autoresize_text_area("具体指令", st.session_state.generated_idea.get("instructions", """
### 1. 产品定位与目标读者
- **目标平台**: 番茄小说、起点中文网。兼顾免费阅读的快节奏和付费阅读的深度设定。
- **核心读者**: 18-35岁男性读者，对系统流、都市异能、美食文、国运元素有阅读偏好。他们追求新奇设定、强烈的爽点反馈和有深度的情节布局。

### 2. 故事风格与基调
- **整体风格**: 现代都市幻想，节奏明快，爽点密集。
- **基调**: 以轻松、热血的成长为主线，中期引入悬疑和权谋元素增加故事张力。主角性格杀伐果断，智商在线，行为逻辑符合“精致利己主义者”的成长过程。

### 3. 核心爽点设计
- **主爽点 (系统成长)**:
  - **解析万物**: 系统能解析食材、菜谱、对手技能，甚至古代遗迹，提供最优解，带来信息差碾压的快感。
  - **气运兑换**: 烹饪美食获得“美食气运”，兑换稀有食材、传说厨具、强大食灵、甚至临时国运加持。
- **辅爽点 (情节驱动)**:
  - **美食对决**: 不仅是厨艺比拼，更是“食灵”的战斗和背后势力的博弈，场面宏大。
  - **打脸逆袭**: 主角从一个落魄厨师，一步步打脸看不起他的竞争对手、美食世家和海外挑战者。
  - **探索解谜**: 揭开家族衰败、美食文明断代的历史谜团，获得古代传承和宝藏。
  - **国运之争**: 以美食影响国运，参与国际级的美食竞赛，为国争光，获得国民崇拜和现实世界的影响力。
"""), key="new_instructions")
            input_brief = st_autoresize_text_area("输入指引", st.session_state.generated_idea.get("input_brief", """
### 1. 主角设定 (Character)
- **姓名**: 楚曜
- **背景**: 曾经是名震一方的美食世家“楚家”的嫡系传人，但家族在一夜之间离奇衰败，父母失踪，他只能在城市角落开一家小餐馆勉强度日，并守护着家族最后的秘密。
- **成长弧光**:
  - **初始缺陷 (Lie)**: “家族的衰败是我的原罪，我只能苟且偷生，无法重现辉煌。”
  - **外在欲望 (Want)**: 赚钱，找到父母，重振家族。
  - **内在需求 (Need)**: 找到自我身份认同，不再背负家族的沉重枷锁，而是作为“楚曜”自己，开创一条前无古人的美食之道，并承担起守护国运的责任。
- **金手指 (System)**:
  - **名称**: “道”系统 (内部代号，初期表现为解析能力)
  - **核心机制**:
    1. **解析**: 可解析一切与“食”相关事物。解析食材，可知其最佳烹饪方式；解析菜谱，可优化流程；解析对手，可知其厨艺弱点。
    2. **演化**: 随着主角烹饪的菜品等级和蕴含气运的提升，系统会解锁新功能，如“气运熔炉”（兑换）、“食灵空间”（培养）、“时空秘境”（获取特殊食材）。
  - **限制**: 解析和演化需要消耗精神力和“美食气运”，过度使用会导致虚弱。高级物品的兑换有前置条件和成功率。

### 2. 世界观核心设定 (World Building)
- **核心概念**: 美食是连接物质世界与精神能量的桥梁。极致的美味可以诞生拥有自主意识的能量体——**食灵**。
- **独特法则**:
  1. **食灵共生**: 厨师可以通过血脉或特殊仪式与“食灵”签订契约，获得超凡厨艺和战斗力。强大的食灵甚至可以影响一方水土的气候。
  2. **美食气运**: 食物不仅提供能量，还蕴含“气运”。普通食物滋养个人，而蕴含历史文化、国民情感的“国菜”则与国运相连。烹饪或品尝国菜，可以增强或削弱国运。
  3. **文明断代**: 历史上曾存在一个辉煌的美食文明，厨师能移山填海，食灵可媲美神明。但一场未知的灾难导致了文明断代，大部分强大的食灵和菜谱失传，主角的家族秘密与此相关。
- **背景谜团**:
  - 楚家为何一夜衰败？父母失踪是否与守护的秘密有关？
  - 历史上的美食文明断代是天灾还是人祸？
  - 海外神秘组织为何也在寻找失落的食灵和菜谱？

### 3. 黄金三章核心情节构思 (Opening Chapters)
**第一章：【深夜餐馆的神秘客人】**
- **钩子**: 深夜，一家即将倒闭的小餐馆，主角楚曜为一位神秘的白发老人做了一道家传的“黯然销魂饭”。
- **冲突**: 老人吃后竟泪流满面，引发天地异象，楚曜脑中系统意外激活，【解析】功能初次上线，显示“黯然销魂饭”蕴含“龙脉气运”0.01%。此时，城中美食协会的执法队闯入，指控他非法烹饪“禁忌菜谱”。
- **爽点**: 系统激活的震撼；家传菜谱的牛逼背景；开局即面临顶级势力的冲突，悬念拉满。

**第二章：【解析！蛋炒饭的十八种变化】**
- **钩子**: 执法队队长不信邪，要求楚曜现场做一道最简单的蛋炒饭来证明清白。
- **冲突**: 楚曜利用系统的解析能力，瞬间洞悉了对方的味觉偏好、身体状况，并解析出普通食材的最佳搭配。他现场做出了一碗看似平凡却让队长食欲大开、暗伤痊愈的“黄金开口笑”蛋炒饭。
- **爽点**: 利用系统信息差的降维打击；于平凡中见神奇的装逼快感；化解危机并反将一军，让执法队长欠下人情。

**第三章：【食灵‘锅巴’，参见主人！】**
- **钩子**: 蛋炒饭的极致美味，竟让锅底剩下的一块锅巴诞生了微弱的灵智——这是食灵诞生的前兆！
- **冲突**: 神秘老人揭示身份，是守护国运的“龙厨”之一，他一直在寻找能唤醒食灵的人。他警告楚曜，他的能力已经引起了海外“饕餮议会”的注意，并赠予他一块能够隐藏气息的“龙鳞”。
- **爽点**: 获得第一个（潜在的）食灵，开启养成线；世界观宏大背景揭开一角；主角正式踏入超凡美食世界，获得新手导师和保命道具。
"""), key="new_input_brief")
            constraints = st_autoresize_text_area("限制和禁忌", st.session_state.generated_idea.get("constraints", """
### 1. 竞品分析与差异化
- **差异化**:
  - **世界观**: 引入“食灵”和“美食文明断代史”，相比传统美食文，增加了养成和探索元素，格局更宏大。
  - **金手指**: “解析”能力比简单的“兑换”更具操作感和智慧感，为主角的“智斗”情节提供支撑。
  - **核心冲突**: 从个人恩怨、家族复兴上升到国运之争和文明传承，目标更远大。

### 2. 市场风险与规避策略
- **风险1 (设定复杂)**: 世界观和系统设定较多，可能劝退部分只想看快节奏爽文的读者。
  - **规避**: 不在早期堆砌设定。通过“神秘老人”的口和具体情节，逐步、自然地揭示世界观。系统功能逐级解锁，保持新鲜感。
- **风险2 (金手指平衡)**: “解析”能力过强可能导致主角无敌，失去成长感。
  - **规避**: 设定精神力消耗和冷却时间。高级目标的解析需要前置条件（如特定道具、自身厨艺等级）。强调解析只是提供“最优解”，执行仍需主角自身努力和技巧。
- **风险3 (读者毒点)**:
  - **圣母/降智**: 主角楚曜设定为有家族仇恨背景的现实主义者，行事以自身利益和复仇为优先，不圣母，不无脑。
  - **感情线**: 感情线为辅，可以有红颜知己，但绝不拖沓，不影响主线节奏。女性角色应独立、有魅力，是伙伴而非附庸。
"""), key="new_constraints")
            acceptance_criteria = st_autoresize_text_area("验收标准", st.session_state.generated_idea.get("acceptance_criteria", """
1. **开篇留存**: 三章内必须完成主角背景交代、金手指激活、核心悬念（家族之谜、国运之争）的铺设，并至少完成一次“打脸-装逼”的完整爽点循环。读者评论区应出现“养肥”、“追了”等正面反馈。
2. **角色塑造**: 主角楚曜的“背负仇恨”和“渴望崛起”的形象必须在开篇通过行动（而非旁白）鲜明地建立起来。
3. **爽点验证**: “解析”能力的核心爽点必须在第二章得到清晰展示，并证明其在解决冲突中的有效性和强大之处。
4. **世界观呈现**: “食灵”或“美食气运”的核心概念必须在第三章通过具体情节（而非设定轰炸）向读者揭示，并引发读者对后续世界的好奇。
5. **钩子强度**: 第三章结尾必须留下一个强有力的钩子，例如“饕餮议会”的第一个敌人已经出现，或者主角接到了一个不可能完成的新手任务。
"""), key="new_acceptance_criteria")
            length = st.text_input("预估总字数", st.session_state.generated_idea.get("length", "100万字"))

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
        for i, book in enumerate(all_books):
            with st.expander(f"**{book.get('name', book.get('root_name', ''))}** (ID: {book['run_id']})", expanded=True):
                # 将操作按钮移动到表单外部的顶部
                b1, b2, b3 = st.columns([1,1,2])
                with b1:
                    st.button("🔄 同步到任务库", key=f"sync_{book['run_id']}", on_click=sync_book_to_task_db, args=(book['run_id'],), use_container_width=True)
                with b2:
                    st.button("🗑️ 删除此项目", key=f"delete_{book['run_id']}", on_click=delete_project, args=(book['run_id'], book.get('name', book.get('root_name', ''))), use_container_width=True, type="secondary")

                with st.form(key=f"edit_form_{book['run_id']}"):
                    st.subheader(f"编辑项目: {book.get('name', book.get('root_name', ''))}")
                    
                    name = st.text_input("书名/项目名", value=book.get('name', book.get('root_name', '')), key=f"name_{book['run_id']}")
                    goal = st_autoresize_text_area("核心目标", value=book.get('goal', ''), key=f"goal_{book['run_id']}")

                    c1, c2, c3, c4 = st.columns(4)
                    category = c1.selectbox("类别", ["story", "report", "book"], index=["story", "report", "book"].index(book.get('category', 'story')), key=f"cat_{book['run_id']}")
                    language = c2.selectbox("语言", ["cn", "en"], index=["cn", "en"].index(book.get('language', 'cn')), key=f"lang_{book['run_id']}")
                    length = c3.text_input("预估总字数", value=book.get('length', ''), key=f"len_{book['run_id']}")
                    day_wordcount_goal = c4.number_input("每日字数目标", value=book.get('day_wordcount_goal', 0), key=f"day_wc_{book['run_id']}")

                    st.divider()
                    st.markdown("#### 核心创作指令")
                    instructions = st_autoresize_text_area("具体指令 (Instructions)", value=book.get('instructions', ''), key=f"instr_{book['run_id']}")
                    input_brief = st_autoresize_text_area("输入指引 (Input Brief)", value=book.get('input_brief', ''), key=f"brief_{book['run_id']}")
                    constraints = st_autoresize_text_area("限制和禁忌 (Constraints)", value=book.get('constraints', ''), key=f"constr_{book['run_id']}")
                    acceptance_criteria = st_autoresize_text_area("验收标准 (Acceptance Criteria)", value=book.get('acceptance_criteria', ''), key=f"accept_{book['run_id']}")
                    
                    st.divider()
                    st.markdown("#### AI生成内容")
                    title = st.text_input("标题 (Title)", value=book.get('title', ''), key=f"title_{book['run_id']}") # 标题通常是单行，保持 st.text_input
                    synopsis = st_autoresize_text_area("简介 (Synopsis)", value=book.get('synopsis', ''), key=f"synopsis_{book['run_id']}") # 已是自动高度
                    style = st_autoresize_text_area("风格 (Style)", value=book.get('style', ''), key=f"style_{book['run_id']}") # 已是自动高度
                    book_level_design = st_autoresize_text_area("全书设计 (Book Level Design)", value=book.get('book_level_design', ''), key=f"design_{book['run_id']}") # 已是自动高度
                    global_state_summary = st_autoresize_text_area("全局状态 (Global State)", value=book.get('global_state_summary', ''), key=f"state_{book['run_id']}") # 已是自动高度

                    st.divider()

                    # 操作按钮
                    submitted = st.form_submit_button("💾 保存修改", use_container_width=True, type="primary")
                    
                    if submitted:
                        updated_book_info = {
                            'name': name,
                            'goal': goal,
                            'category': category, 'language': language, 
                            'instructions': instructions, 'input_brief': input_brief,
                            'constraints': constraints, 'acceptance_criteria': acceptance_criteria,
                            'length': length, 'day_wordcount_goal': day_wordcount_goal,
                            'title': title, 'synopsis': synopsis, 'style': style,
                            'book_level_design': book_level_design, 'global_state_summary': global_state_summary
                        }
                        meta_db = get_meta_db()
                        meta_db.update_book(book['run_id'], updated_book_info)
                        st.success(f"项目《{name}》已成功更新！")
                        st.rerun()


st.set_page_config(layout="wide", page_title="项目管理")
render_project_management_page()