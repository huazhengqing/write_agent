import os
import json
import asyncio
import litellm
from loguru import logger
from typing import List, TypedDict, Annotated, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_community.utilities import SearxSearchWrapper
from langgraph.graph import StateGraph, END
from ..util.models import Task
from ..util.llm import get_llm_params
from ..memory import memory
from ..prompts.story.search_cn import SYSTEM_PROMPT, USER_PROMPT
from ..prompts.story.search_aggregate_cn import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_AGGREGATE,
    USER_PROMPT as USER_PROMPT_AGGREGATE,
)


"""


分析、审查当前文件的代码, 找出bug并改正,  指出可以优化的地方。


根据以上分析, 改进建议,  请直接修改 文件, 并提供diff。


内容筛选   内容摘要

整体架构 (Overall Architecture)
这个搜索模块的架构是一个典型的 ReAct (Reason+Act) 框架。它不是简单地将用户问题直接扔给搜索引擎, 而是通过一个智能代理（Agent）来模拟人类研究问题时的思考和行动过程。

其主要组成部分包括：

思考与规划模块 (Think & Plan): Agent的核心, 由一个大语言模型（LLM）驱动。它负责理解当前任务目标, 分析已有信息, 规划下一步的搜索动作。
行动执行模块 (Action): 负责执行具体的搜索操作。它可以对接不同的搜索引擎后端, 如 SearXNG 或 SerpApi。
信息处理模块 (Information Processing):
网页抓取 (Web Fetching): 并发下载搜索结果中的网页内容。
内容筛选 (Selector): 使用一个专门的LLM（selector_model）从众多网页中筛选出最相关的部分。
内容摘要 (Summarizer): 使用另一个LLM（summarizer_model）对筛选出的网页内容进行总结, 提取关键信息。
结果整合模块 (Result Merging): 在多轮搜索结束后, 使用LLM将所有收集到的信息整合成一个连贯、全面的最终答案。
这种分层、分模块的设计使得整个搜索过程更加精细、高效且结果质量更高。

核心流程 (Core Workflow)
当一个检索任务被触发时, execute配置下的流程会被激活：

启动ReAct循环: Agent进入一个迭代循环（最多执行 max_turn 轮, 这里是4轮）。
思考与生成查询 (Turn 1):
Agent使用 SearchAgentENPrompt 提示词, 让主LLM（global_use_model）进行思考（planning_and_think）。
LLM根据思考, 生成一个或多个具体的搜索查询词（current_turn_search_querys）。
执行搜索与抓取:
系统使用配置的搜索后端（searcher_type, 如 SerpApi）并行执行这些查询（search_max_thread）。
获取每个查询的前 topk (20) 个结果。
系统会并行下载这些搜索结果链接指向的网页内容（webpage_helper_max_threads）。
筛选与摘要:
筛选 (Select): 使用 selector_model (gpt-4o-mini) 并行地（selector_max_workers）从下载的 pk_quota (20) 个页面中, 挑选出最相关的 select_quota (12) 个页面。
摘要 (Summarize): 使用 summarizer_model (gpt-4o-mini) 并行地（summarizier_max_workers）为这12个筛选出的页面生成内容摘要。
观察与再次思考 (Turn 2, 3, ...):
所有页面的摘要信息被整合为“观察结果（observation）”。
Agent将这个观察结果、原始问题和之前的思考历史一起, 再次输入给主LLM。
LLM分析新的信息, 判断是否还需要补充信息（missing_info）, 并规划下一轮的搜索查询。
循环结束: 这个过程会重复, 直到达到 max_turn 上限, 或者LLM认为信息已经足够, 不再生成新的搜索查询。
最终整合:
循环结束后, 如果 llm_merge 为 True, 系统会将所有轮次中收集到的摘要信息汇总。
调用主LLM, 使用 MergeSearchResultVFinal 提示词（在search_merge配置中定义）, 对所有信息进行最终的综合、提炼和整理, 生成最终的答案（result）。
关键技术与组件
ReAct 框架: 这是整个Agent的灵魂。通过“思考 -> 行动 -> 观察”的闭环, 让LLM能够动态规划、执行和反思, 解决了复杂问题需要多步推理和信息搜集的需求。
多模型协作 (Multi-LLM Collaboration): 系统没有依赖单一的LLM, 而是为不同子任务（主逻辑、筛选、摘要）配置了最适合的模型（这里都用了gpt-4o-mini, 但可以配置不同模型）, 实现了“专业的人做专业的事”, 提升了效率和质量。
并行处理 (Parallel Processing): 在多个环节（执行搜索、下载网页、筛选、摘要）都采用了并行处理, 通过 *_max_threads 和 *_max_workers 参数控制并发数。这极大地缩短了I/O密集型和计算密集型任务的等待时间, 是保证Agent响应速度的关键。
可插拔搜索引擎后端: 通过 searcher_type 和 backend_engine 参数, 可以灵活切换搜索引擎。
SearXNG: 一个开源的、可自托管的元搜索引擎, 注重隐私和可定制性。
SerpApi: 一个商业化的API服务, 可以稳定地调用Google、Bing等主流搜索引擎, 避免了自己处理反爬虫问题。
配置驱动 (Configuration-Driven): 整个Agent的行为逻辑（如prompt版本、模型、并发数、循环次数等）都通过这个Python字典来定义, 而不是硬编码在代码中。这使得系统非常灵活, 便于调试、迭代和针对不同场景进行优化。
配置参数详解
react_agent: True: 明确启用ReAct模式的搜索Agent。
prompt_version: "SearchAgentENPrompt": 指定ReAct循环中主LLM使用的提示词模板。
searcher_type: 根据 engine_backend 变量动态选择 SearXNG 或 SerpApi。
llm_args: 主LLM的参数, 如模型名称。
react_parse_arg_dict: 定义如何从主LLM返回的XML格式响应中解析出思考、查询词等结构化数据。
temperature: 0.2: Agent思考时LLM的温度参数, 较低的值意味着更确定、更保守的思考过程。
max_turn: 4: ReAct循环的最大轮次。
llm_merge: True: 开启最终的LLM整合步骤。
webpage_helper_max_threads: 10: 下载网页时的最大并发线程数。
search_max_thread: 4: 同时执行搜索查询的最大并发数。
backend_engine: SerpApi 使用的后端, 如 google 或 bing。
cc: "US": 指定搜索区域为美国。
topk: 20: 从搜索引擎获取排名前20的结果。
pk_quota: 20: 初步处理的网页数量上限。
select_quota: 12: 从初步处理的网页中最终筛选出的数量。
selector_model / summarizer_model: 分别为筛选和摘要任务指定的LLM模型。
总而言之, 这段配置定义了一个非常强大和完善的自动化研究员（AI Researcher）。它通过模拟人类的思考-行动循环, 结合多模型协作和大规模并行处理, 能够高效、深入地研究一个问题, 并最终给出一个高质量的综合性报告。


"""


###############################################################################


class SearchQueries(BaseModel):
    queries: List[str] = Field(
        ...,
        description="根据原始任务生成的一个或多个精确、简洁的搜索查询关键词列表。"
    )


async def search(task: Task) -> Task:
    reasoning_data = {}
    context_dict = await memory.get_context(task)
    search_queries = await _generate_search_queries(context_dict, reasoning_data)
    all_search_results_str = await _execute_searches(search_queries, reasoning_data)
    final_report = await _synthesize_results(task.description, all_search_results_str)
    updated_task = task.model_copy(deep=True)
    updated_task.results = {
        "result": final_report,
        "reasoning": reasoning_data,
    }
    return updated_task


async def _generate_search_queries(context_dict: dict, reasoning_data: dict) -> List[str]:
    json_instructions = f"""
请严格按照以下 JSON 格式返回你的回答, 不要包含任何其他解释或注释。
JSON Schema:
```json
{SearchQueries.model_json_schema()}
```
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + json_instructions},
        {"role": "user", "content": USER_PROMPT.format(**context_dict)}
    ]
    llm_params = get_llm_params(messages, temperature=0.1)
    llm_params['response_format'] = {"type": "json_object", "schema": SearchQueries.model_json_schema()}

    content = ""
    response = await litellm.acompletion(**llm_params)
    content = response.choices[0].message.content
    reasoning_data["1_keyword_generation_llm_output"] = content

    if not content:
        logger.error("关键词生成 LLM 返回了空内容。")
        raise ValueError("关键词生成 LLM 返回了空内容。")

    search_queries_model = SearchQueries.model_validate_json(content)
    queries = search_queries_model.queries
    
    if not queries:
        logger.warning("LLM 生成了空的查询列表, 将跳过搜索步骤。")
        return [] # 返回空列表, 让后续步骤决定如何处理
        
    reasoning_data["2_generated_search_queries"] = queries
    return queries



async def _execute_searches(queries: List[str], reasoning_data: dict) -> str:
    if not queries:
        logger.info("没有提供搜索关键词, 跳过网络搜索。")
        return "没有进行网络搜索, 因为没有提供有效的搜索关键词。"

    search_tool = SearxSearchWrapper(searx_host=os.environ.get("SearXNG", "http://127.0.0.1:8080/search"))
    
    try:
        search_tasks = [search_tool.arun(query) for query in queries]
        search_results_list = await asyncio.gather(*search_tasks)
        
        all_search_results_str = "\n\n".join(
            f"查询 '{query}' 的结果:\n{result}" for query, result in zip(queries, search_results_list)
        )
        reasoning_data["3_collected_search_results"] = all_search_results_str
        return all_search_results_str
    except Exception as e:
        logger.error(f"执行网络搜索时发生错误: {e}")
        # 即使搜索失败, 也返回一个信息, 而不是让整个流程崩溃
        return f"执行网络搜索时发生错误: {e}"


async def _synthesize_results(task_description: str, search_results: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGGREGATE},
        {"role": "user", "content": USER_PROMPT_AGGREGATE.format(task_description=task_description, search_results=search_results)}
    ]
    llm_params = get_llm_params(messages, temperature=0.1)
    response = await litellm.acompletion(**llm_params)
    content = response.choices[0].message.content
    if not content:
        logger.error("综合报告 LLM 返回了空内容。")
        raise ValueError("综合报告 LLM 返回了空内容。")
    return content


