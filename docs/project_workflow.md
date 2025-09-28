# 项目工作流程详解

write_agent 是一个基于AI驱动的自动化写作智能代理系统，采用模块化设计，通过多个智能体协同工作来完成复杂的创作任务。其核心工作流程基于任务分解与递归执行机制，结合记忆服务实现上下文感知和结果持久化。

## 1. 整体架构与核心组件

系统采用分层架构，主要包括以下几个核心组件：

- **入口层 (`main.py`)**: 负责解析命令行参数和JSON输入，为每个任务创建独立的运行上下文。
- **工作流层 (`story/story_write.py`)**: 定义了主流程 `flow_story_write`，协调原子性判断、任务规划、执行和结果聚合。
- **代理层 (`agents/`)**: 实现具体的业务逻辑，包括原子性判断 (`atom.py`)、任务规划 (`plan.py`)、内容设计 (`design.py`)、信息搜索 (`search.py`)、内容写作 (`write.py`) 等。
- **模型层 (`utils/models.py`)**: 定义了核心数据结构 `Task`，作为各组件间传递的核心数据载体。
- **记忆层 (`story/story_rag.py`, `utils/sqlite_task.py`)**: 提供持久化存储和上下文管理，利用SQLite数据库和LlamaIndex RAG框架提升效率。
- **提示词层 (`prompts/`)**: 存放控制各智能体行为的系统提示词模板。

## 2. 主要工作流程

### 2.1 主流程启动

1. 用户通过 `main.py` 提交任务配置（JSON格式）。
2. `main.py` 解析参数并加载任务配置。
3. 为每个根任务调用 `flow_story_write` 函数启动工作流。

```
graph TD
    A[用户提交任务配置] --> B[main.py解析参数]
    B --> C[加载任务配置]
    C --> D[调用flow_story_write]
    D --> E[开始任务处理循环]
```

### 2.2 任务处理循环

`flow_story_write` 是系统的核心执行流程，采用递归方式处理任务树。其基本逻辑如下：

1. **字数目标检查**: 首先检查是否达到每日字数目标(`day_wordcount_goal`)，如果达到则暂停任务。
2. **原子性判断**: 调用 `task_atom` 智能体判断当前任务是否为原子任务。
   - 如果是原子任务，则进入原子任务执行流程。
   - 如果是复杂任务，则进入任务分解流程。

```
graph TD
    A[flow_story_write开始] --> B{检查字数目标}
    B -->|已达到| C[暂停任务]
    B -->|未达到| D[调用task_atom]
    D --> E{任务是原子的?}
    E -->|是| F[原子任务执行流程]
    E -->|否| G[复杂任务分解流程]
    F --> H[任务完成]
    G --> H
```

### 2.3 原子任务执行流程

当任务被判定为原子任务时，系统根据任务类型执行相应操作：

- **design任务**:
  1. 调用 `task_route` 确定设计类别。
  2. 调用 `task_design` 执行具体的设计任务。
  3. 调用 `task_design_reflection` 对设计结果进行反思和优化。
  4. 将结果持久化到记忆库。

```
graph TD
    A[design任务] --> B[调用task_route]
    B --> C[调用task_design]
    C --> D[调用task_design_reflection]
    D --> E[持久化结果]
```

- **search任务**:
  1. 调用 `task_search` 执行搜索任务。
  2. 将结果持久化到记忆库。

```
graph TD
    A[search任务] --> B[调用task_search]
    B --> C[持久化结果]
```

- **write任务**:
  1. 调用 `task_write_before_reflection` 进行写作前的反思。
  2. 调用 `task_write` 执行写作任务。
  3. 调用 `task_write_reflection` 对初稿进行反思和优化。
  4. 调用 `task_summary` 生成正文摘要。
  5. 根据任务层级判断是否需要调用 `task_review_write` 进行正文审查。
  6. 将结果持久化到记忆库。

```
graph TD
    A[write任务] --> B[调用task_write_before_reflection]
    B --> C[调用task_write]
    C --> D[调用task_write_reflection]
    D --> E[调用task_summary]
    E --> F{需要审查?}
    F -->|是| G[调用task_review_write]
    F -->|否| H[持久化结果]
    G --> H
```

### 2.4 复杂任务分解流程

当任务被判定为复杂任务时，系统会对其进行分解：

1. **前置检查**: 对于write任务，检查是否存在前置的设计任务。
2. **任务规划**:
   - 如果是write任务且缺少前置设计，则调用 `task_plan_write_to_design` 生成设计任务。
   - 否则，调用 `task_review_design` 审查设计方案，然后调用 `task_hierarchy` 和 `task_hierarchy_reflection` 划分结构，最后调用 `task_plan_write_to_write` 分解写作任务。
   - 对于design/search任务，直接调用 `task_plan` 进行分解。
3. **规划反思**: 调用 `task_plan_reflection` 对规划结果进行反思和优化。
4. **子任务处理**: 对每个子任务递归调用 `flow_story_write`。
5. **结果聚合**:
   - 对于design任务，调用 `task_aggregate_design` 聚合结果。
   - 对于search任务，调用 `task_aggregate_search` 聚合结果。
   - 对于write任务，调用 `task_aggregate_summary` 聚合摘要，并根据需要调用 `task_review_write` 进行正文审查。
   - 将聚合结果持久化到记忆库。

```
graph TD
    A[复杂任务] --> B{write任务且缺少设计?}
    B -->|是| C[调用task_plan_write_to_design]
    B -->|否| D[调用task_review_design]
    D --> E[调用task_hierarchy]
    E --> F[调用task_hierarchy_reflection]
    F --> G[调用task_plan_write_to_write]
    G --> H[调用task_plan]
    C --> I[调用task_plan_reflection]
    H --> I
    I --> J[递归处理子任务]
    J --> K[结果聚合]
    K --> L[持久化聚合结果]
```

## 3. 任务状态管理与持久化

系统通过以下机制管理任务状态并实现持久化：

- **Task模型**: `Task` 对象作为状态载体，在各个处理节点间传递，其 `results` 字段存储任务执行结果。
- **状态持久化**: 通过 `task_save_data` 函数将任务数据路由到 `StoryRAG` 系统进行统一的持久化处理，存储到SQLite数据库和RAG索引中。
- **缓存机制**: 利用 `diskcache` 实现多级缓存，提升上下文获取效率。

## 4. 智能体协作机制

系统中的智能体遵循统一的执行契约：

1. **输入验证**: 验证任务ID和目标是否为空。
2. **上下文构建**: 调用 `StoryRAG` 获取任务依赖、历史设计、最新内容等上下文信息。
3. **提示词注入**: 加载相应提示词模板，并将上下文信息注入提示词。
4. **LLM调用**: 封装LLM调用参数并发送请求。
5. **输出解析**: 解析LLM响应并封装到 `Task` 对象中。
6. **结果返回**: 返回更新后的 `Task` 对象。

## 5. 工作流特点

- **递归执行**: 通过递归调用 `flow_story_write` 实现任务树的深度处理。
- **并行处理**: 利用 `asyncio` 和 Prefect 框架实现高效的并发执行。
- **上下文感知**: 通过RAG系统获取丰富的上下文信息，确保生成内容的一致性和连贯性。
- **结果持久化**: 所有任务执行结果都会被持久化存储，支持任务中断恢复和结果追溯。
- **灵活扩展**: 模块化设计使得系统易于扩展新的智能体和功能。

通过以上机制，write_agent系统能够高效、可靠地完成从简单任务到复杂创作项目的自动化处理。