import hashlib
from loguru import logger

from utils.file import sanitize_filename
from utils.models import Task
from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import get_task_db


def add_sample_story_task_to_meta_db():
    task_info = {
        'category': "story",
        'language': "cn",
        'name': "龙与魔法之歌",
        'goal': "创作一部史诗奇幻小说，讲述一位年轻的法师如何与一头古老的龙结盟，共同对抗威胁世界的黑暗势力。",
        'instructions': "小说应包含丰富的世界观设定。引人入胜的角色发展。以及激动人心的魔法战斗场面。",
        'input_brief': "世界背景是一个名为“艾瑞多”的大陆，魔法分为元素、光明和黑暗三大体系。龙族是古老的守护者，但已久不现世。",
        'constraints': "避免使用过于现代的词汇。主角不能一开始就无敌，需要有成长过程。",
        'acceptance_criteria': "完成至少10万字的小说初稿。主线故事完整，角色弧光清晰。",
        'length': "10万字",
        'day_wordcount_goal': 1000
    }
    book_meta_db = get_meta_db()
    book_meta_db.add_book(task_info)
    


def sync_meta_to_task_db(run_id_to_sync: str = None):
    """
    检查 BookMetaDB 中的所有书籍元数据，并为每本书在对应的 TaskDB 中创建或更新根任务。
    如果提供了 run_id_to_sync, 则只同步指定的书籍。
    """
    book_meta_db = get_meta_db()
    
    if run_id_to_sync:
        logger.info(f"开始为单个项目 {run_id_to_sync} 同步元数据到 TaskDB...")
        book_meta = book_meta_db.get_book_meta(run_id_to_sync)
        all_books_meta = [book_meta] if book_meta else []
    else:
        logger.info("开始同步所有项目的 BookMetaDB 到各自的 TaskDB...")
        all_books_meta = book_meta_db.get_all_book_meta()

    if not all_books_meta:
        logger.warning("BookMetaDB 中没有找到任何书籍元数据，无需同步。")
        return
    
    for book_meta in all_books_meta:
        run_id = book_meta.get("run_id")
        if not run_id:
            logger.error(f"发现一条缺少 run_id 的元数据，已跳过: {book_meta}")
            continue

        logger.info(f"正在处理书籍: {book_meta.get('root_name')} (run_id: {run_id})")

        # 1. 为每本书获取对应的 TaskDB 实例
        task_db = get_task_db(run_id)

        # 2. 从 book_meta 构建根任务 Task 对象
        # 注意：这里的字段需要和 Task 模型以及 t_book_meta 表的字段对应
        root_task = Task(
            id="1",
            parent_id="",
            task_type="write",
            status="pending",
            hierarchical_position="全书",
            goal=book_meta.get("goal", ""),
            instructions=book_meta.get("instructions", "").split("\n"),
            input_brief=book_meta.get("input_brief", "").split("\n"),
            constraints=book_meta.get("constraints", "").split("\n"),
            acceptance_criteria=book_meta.get("acceptance_criteria", "").split("\n"),
            length=book_meta.get("length", "根据任务要求确定"),
            category=book_meta.get("category"),
            language=book_meta.get("language"),
            root_name=book_meta.get("root_name"),
            run_id=run_id,
            day_wordcount_goal=book_meta.get("day_wordcount_goal", 0)
        )

        # 3. 将根任务添加到对应的 TaskDB 中
        task_db.add_task(root_task)
        logger.success(f"已在 TaskDB (run_id: {run_id}) 中成功创建/更新根任务 (id: 1)。")

    logger.info(f"同步完成，共处理了 {len(all_books_meta)} 本书。")