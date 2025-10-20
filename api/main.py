import os
import sys
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, AnyHttpUrl
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from dotenv import load_dotenv
load_dotenv()


from utils.sqlite_meta import get_meta_db
from utils.sqlite_task import get_task_db, dict_to_task, TaskDB
from utils.models import Task
from utils.file import data_dir
from story.task import create_root_task, do_task, delete_task, stop_task
from story.idea import generate_idea, IdeaOutput


app = FastAPI(
    title="AI 写作智能体 API",
    description="一个用于管理写作项目(书籍)和任务的 API。",
    version="1.0.0",
)

# --- CORS 中间件配置 ---
# 允许所有来源, 所有方法, 所有头, 这在开发阶段非常方便。
# 在生产环境中, 您应该将其限制为您的前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或者指定前端地址, 如 ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- Pydantic 模型定义 ---


class BookMeta(BaseModel):
    """书籍元数据的完整模型, 用于响应"""
    run_id: str
    name: str
    goal: Optional[str] = None
    category: Optional[str] = "story"
    language: Optional[str] = "cn"
    instructions: Optional[str] = None
    input_brief: Optional[str] = None
    constraints: Optional[str] = None
    acceptance_criteria: Optional[str] = None
    length: Optional[str] = None
    day_wordcount_goal: Optional[int] = 0
    title: Optional[str] = None
    synopsis: Optional[str] = None
    style: Optional[str] = None
    book_level_design: Optional[str] = None
    global_state_summary: Optional[str] = None
    status: Optional[str] = "idle"
    word_count_today: Optional[int] = 0


class BookCreate(BaseModel):
    """创建书籍时请求体使用的模型。""" 
    name: str = Field(..., min_length=1, description="书名/项目名, 不能为空")
    goal: str = Field(..., min_length=1, description="核心目标, 不能为空")
    category: Optional[str] = "story"
    language: Optional[str] = "cn"
    instructions: Optional[str] = ""
    input_brief: Optional[str] = ""
    constraints: Optional[str] = ""
    acceptance_criteria: Optional[str] = ""
    length: Optional[str] = ""
    day_wordcount_goal: Optional[int] = 20000


class TaskUpdate(BaseModel):
    """用于更新任务的模型, 包含所有可更新的字段"""
    parent_id: Optional[str] = None
    hierarchical_position: Optional[str] = None
    task_type: Optional[str] = None
    status: Optional[str] = None
    goal: Optional[str] = None
    length: Optional[str] = None
    instructions: Optional[str] = None
    input_brief: Optional[str] = None
    constraints: Optional[str] = None
    acceptance_criteria: Optional[str] = None
    expert: Optional[str] = None
    atom: Optional[str] = None
    atom_reasoning: Optional[str] = None
    plan: Optional[str] = None
    plan_reasoning: Optional[str] = None
    plan_completed: Optional[int] = None
    design: Optional[str] = None
    design_reasoning: Optional[str] = None
    search: Optional[str] = None
    search_reasoning: Optional[str] = None
    hierarchy: Optional[str] = None
    hierarchy_reasoning: Optional[str] = None
    hierarchy_completed: Optional[int] = None
    write: Optional[str] = None
    write_reasoning: Optional[str] = None
    summary: Optional[str] = None
    summary_reasoning: Optional[str] = None
    book_level_design: Optional[str] = None
    global_state: Optional[str] = None
    write_review: Optional[str] = None
    write_review_reasoning: Optional[str] = None
    translation: Optional[str] = None
    translation_reasoning: Optional[str] = None
    results: Optional[Dict[str, Any]]=None


class TaskRunResponse(BaseModel):
    """任务开始执行时的响应模型"""
    message: str
    run_id: str
    task_id: str
    status_url: AnyHttpUrl


@app.get("/api/books", response_model=List[BookMeta], tags=["Books"])
def get_all_books_api():
    """获取所有书籍的元数据列表。"""
    meta_db = get_meta_db()
    books_meta = meta_db.get_all_book_meta()
    # 为每本书籍补充今日字数
    for book in books_meta:
        task_db = get_task_db(book['run_id'])
        book['word_count_today'] = task_db.get_word_count_last_24h()
    return books_meta


@app.post("/api/books", response_model=Dict, status_code=201, tags=["Books"])
def create_book_api(book_data: BookCreate):
    """创建一个新的书籍项目。"""
    meta_db = get_meta_db()
    run_id = meta_db.add_book(book_data.model_dump())
    new_book_meta = meta_db.get_book_meta(run_id)
    if not new_book_meta:
        raise HTTPException(status_code=500, detail="创建书籍后无法立即找到, 请检查数据库。")   
    return new_book_meta


@app.post("/api/books/generate-idea", response_model=IdeaOutput, tags=["Books"])
async def generate_idea_api():
    """
    使用 AI 生成一个新的书籍创意, 包含名称、目标、指令等元信息。
    """
    try:
        idea = await generate_idea()
        if not idea:
            raise HTTPException(status_code=500, detail="AI 未能生成有效的创意。")
        return idea
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成创意时发生内部错误: {e}")


@app.get("/api/books/{run_id}", response_model=BookMeta, tags=["Books"])
def get_book_api(run_id: str):
    """根据 run_id 获取单本书籍的详细元数据。"""
    meta_db = get_meta_db()
    book = meta_db.get_book_meta(run_id)
    if not book:
        raise HTTPException(status_code=404, detail=f"未找到 run_id 为 '{run_id}' 的书籍。")
    task_db = get_task_db(run_id)
    book['word_count_today'] = task_db.get_word_count_last_24h()
    return book


@app.post("/api/books/{run_id}/sync", status_code=200, tags=["Books"])
def sync_book_to_task_db_api(run_id: str):
    """将书籍项目同步到任务库, 创建根任务。"""
    meta_db = get_meta_db()
    if not meta_db.get_book_meta(run_id):
        raise HTTPException(status_code=404, detail=f"未找到 run_id 为 '{run_id}' 的书籍。")
    create_root_task(run_id)
    return {"message": f"项目 {run_id} 已成功同步到任务库！"}


@app.post("/api/books/{run_id}/stop", status_code=status.HTTP_200_OK, tags=["Books"])
def stop_book_task_api(run_id: str):
    """停止正在运行的书籍项目。"""
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(run_id)
    if not book_meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未找到 run_id 为 '{run_id}' 的书籍。")
    if book_meta.get("status") != "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"项目 '{run_id}' 当前状态为 '{book_meta.get('status')}', 无法停止。只有 'running' 状态的项目才能被停止。"
        )
    meta_db.update_status(run_id, "stopping")
    return {"message": f"项目 '{run_id}' 已被标记为暂停。任务将在下一个检查点停止。"}


@app.get("/api/books/{run_id}/word-count-today", response_model=Dict[str, int], tags=["Books"])
def get_word_count_today_api(run_id: str):
    """获取指定项目在过去24小时内完成的写作字数。"""
    task_db = get_task_db(run_id)
    count = task_db.get_word_count_last_24h()
    return {"word_count_last_24h": count}


@app.put("/api/books/{run_id}", response_model=BookMeta, tags=["Books"])
def update_book_api(run_id: str, book_update: BookMeta):
    """更新指定 run_id 的书籍信息。"""
    meta_db = get_meta_db()
    if not meta_db.get_book_meta(run_id):
        raise HTTPException(status_code=404, detail=f"未找到 run_id 为 '{run_id}' 的书籍。")
    # Pydantic 的 model_dump 可以方便地将模型转为字典
    update_data = book_update.model_dump(exclude={'word_count_today'})
    meta_db.add_book(update_data)
    updated_book = meta_db.get_book_meta(run_id)
    if not updated_book:
         raise HTTPException(status_code=500, detail="更新书籍后无法立即找到, 请检查数据库。")
    task_db = get_task_db(run_id)
    updated_book['word_count_today'] = task_db.get_word_count_last_24h()
    return updated_book


@app.delete("/api/books/{run_id}", status_code=204, tags=["Books"])
def delete_book_api(run_id: str):
    """删除指定 run_id 的书籍及其所有相关文件。"""
    meta_db = get_meta_db()
    if not meta_db.get_book_meta(run_id):
        raise HTTPException(status_code=404, detail=f"未找到 run_id 为 '{run_id}' 的书籍。")
    # 删除元数据
    meta_db.delete_book_meta(run_id)
    return None


@app.get("/api/books/{run_id}/tasks", response_model=List[Task], tags=["Tasks"])
def get_tasks_for_book_api(run_id: str):
    """获取指定书籍的所有任务列表。"""
    meta_db = get_meta_db()
    if not meta_db.get_book_meta(run_id):
        raise HTTPException(status_code=404, detail=f"未找到 run_id 为 '{run_id}' 的书籍。")
    task_db = get_task_db(run_id)
    tasks_data = task_db.get_all_tasks()
    return [dict_to_task({**t, "run_id": run_id}) for t in tasks_data if t]


@app.get("/api/tasks/{task_id}", response_model=Task, tags=["Tasks"])
def get_task_api(task_id: str, run_id:str):
    """根据 task_id 和 run_id 获取单个任务的详细信息。"""
    task_db: TaskDB = get_task_db(run_id)
    task_data = task_db.get_task_by_id(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail=f"在项目 '{run_id}' 中未找到 id 为 '{task_id}' 的任务。")
    return dict_to_task({**task_data, "run_id": run_id})


@app.put("/api/tasks/{task_id}", response_model=Task, tags=["Tasks"])
def update_task_api(task_id: str, run_id: str, task_update: TaskUpdate):
    """更新指定 task_id 的任务信息。"""
    task_db = get_task_db(run_id)
    existing_task = task_db.get_task_by_id(task_id)

    if not existing_task:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 '{task_id}' 的任务。")

    task_data = dict_to_task({**existing_task, "run_id": run_id})

    for field, value in task_update.model_dump(exclude_unset=True).items():
        if field == 'results' and isinstance(value, dict):
            # 如果 results 字段本身被更新, 则合并它
            if task_data.results:
                task_data.results.update(value)
            else:
                task_data.results = value
        elif field in Task.model_fields:
            setattr(task_data, field, value)
        else:
            # 对于不在 Task 模型中但在 TaskUpdate 中的字段 (现在都是结果字段), 也放入 results
            if task_data.results is not None:
                task_data.results[field] = value

    task_db.add_task(task_data)
    task_db.add_result(task_data)
    updated_task = task_db.get_task_by_id(task_id)
    if not updated_task:
        raise HTTPException(status_code=500, detail="更新任务后无法立即找到, 请检查数据库。")

    return dict_to_task({**updated_task, "run_id": run_id})


@app.delete("/api/tasks/{task_id}", status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
def delete_task_api(task_id: str, run_id: str, background_tasks: BackgroundTasks):
    """在后台递归删除指定 task_id 的任务及其所有子任务。"""
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(run_id)
    if book_meta and book_meta.get("status") != "idle":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"项目 '{run_id}' 当前状态为 '{book_meta.get('status')}', 无法执行删除操作。请等待其变为空闲(idle)状态。"
        )
    task_db = get_task_db(run_id)
    task_data = task_db.get_task_by_id(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 '{task_id}' 的任务。")
    task = dict_to_task({**task_data, "run_id": run_id})
    if not task:
        raise HTTPException(status_code=500, detail=f"无法将任务数据转换为任务对象 for task_id '{task_id}'。")
    background_tasks.add_task(delete_task, task)
    return {"message": f"任务 '{task_id}' 已被接受并将在后台删除。"}



@app.post("/api/tasks/{task_id}/run", response_model=TaskRunResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
async def run_task_api(task_id: str, run_id: str, background_tasks: BackgroundTasks):
    """
    在后台开始执行一个指定的任务。
    此接口会立即返回, 任务将在后台运行。
    """
    meta_db = get_meta_db()
    book_meta = meta_db.get_book_meta(run_id)
    if book_meta and book_meta.get("status") in ["running", "deleting"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"项目 '{run_id}' 当前状态为 '{book_meta.get('status')}', 无法执行新任务。"
        )
    task_db = get_task_db(run_id)
    task_data = task_db.get_task_by_id(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail=f"在项目 '{run_id}' 中未找到 id 为 '{task_id}' 的任务。")
    task = dict_to_task({**task_data, "run_id": run_id})
    background_tasks.add_task(do_task, task)
    return TaskRunResponse(
        message=f"任务 '{task_id}' 已开始在后台执行。",
        run_id=run_id,
        task_id=task_id,
        status_url=f"/api/tasks/{task_id}?run_id={run_id}"
    )


# --- 如何运行 ---
# 1. 确保你已经安装了 fastapi 和 uvicorn:
#    pip install fastapi "uvicorn[standard]"
#
# 2. 在命令行中, 切换到 `ai_book` 目录, 然后运行:
#    uvicorn write_agent.api.main:app --reload
#
# 3. API 将在 http://127.0.0.1:8000 启动。
#
# 4. 访问 http://127.0.0.1:8000/docs 查看自动生成的交互式 API 文档。
#


if __name__ == "__main__":
    import uvicorn
    # 这使得你可以通过 `python -m write_agent.api.main` 直接运行
    uvicorn.run(app, host="127.0.0.1", port=8000)