0,0 @@
import os
import sys
import pytest
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import Task
from utils.llm import get_llm_messages, llm_completion, get_llm_params
from story.prompts.rag_query import (
    Inquiry,
    system_prompt_design,
    user_prompt_design,
    system_prompt_design_for_write,
    system_prompt_write,
    user_prompt_write,
    system_prompt_search,
    user_prompt_search,
)


@pytest.mark.asyncio
async def test_rag_query_design_prompt():
    logger.info("--- 测试 RAG Query (Design) ---")
    task = Task(
        id="1.2",
        parent_id="1",
        task_type="design",
        hierarchical_position="全书",
        goal="设计主角'龙傲天'的背景故事、核心动机与成长弧光。",
        category="story",
        language="cn",
        root_name="赛博真仙",
        run_id="test_run_rag_query_design",
    )
    context = {
        "task": task.model_dump_json(indent=2),
        "task_list": "1. 全书 write -> 1.1 design 世界观 -> 1.2 design 主角",
        "design_dependent": "",
        "search_dependent": "",
        "latest_text": "",
    }
    messages = get_llm_messages(
        system_prompt=system_prompt_design,
        user_prompt=user_prompt_design,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.0,
        messages=messages
    )
    response = await llm_completion(llm_params, response_model=Inquiry)
    result = response.validated_data
    logger.info(f"LLM 生成的探询问题 (Design):\n{result.model_dump_json(indent=2, ensure_ascii=False)}")


@pytest.mark.asyncio
async def test_rag_query_design_for_write_prompt():
    task = Task(
        id="1.1.1",
        parent_id="1.1",
        task_type="write",
        hierarchical_position="第一章",
        goal="续写龙傲天与赵日天在黑松林遭遇埋伏后的情节。",
        length="1500",
        category="story",
        language="cn",
        root_name="赛博真仙",
        run_id="test_run_rag_query_design_for_write",
    )
    context = {
        "task": task.model_dump_json(indent=2),
        "task_list": "1. 全书 write -> 1.1 第一卷 write -> 1.1.1 第一章 write",
        "design_dependent": "# 章节设计: 黑松林之变后续\n- 核心事件: 龙傲天检查敌人留下的令牌, 发现了一个神秘符号, 决心查明真相。",
        "search_dependent": "",
        "latest_text": "黑衣人头领见状, 眼中闪过一丝惊异, 随即果断下令: "撤！"",
    }
    messages = get_llm_messages(
        system_prompt=system_prompt_design_for_write,
        user_prompt=user_prompt_design,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.0,
        messages=messages
    )
    response = await llm_completion(llm_params, response_model=Inquiry)
    result = response.validated_data
    logger.info(f"LLM 生成的探询问题 (Design for Write):\n{result.model_dump_json(indent=2, ensure_ascii=False)}")


@pytest.mark.asyncio
async def test_rag_query_write_prompt():
    task = Task(
        id="1.1.1",
        parent_id="1.1",
        task_type="write",
        hierarchical_position="第一章",
        goal="续写龙傲天与赵日天在黑松林遭遇埋伏后的情节。",
        length="1500",
        category="story",
        language="cn",
        root_name="赛博真仙",
        run_id="test_run_rag_query_write",
    )
    context = {
        "task": task.model_dump_json(indent=2),
        "task_list": "1. 全书 write -> 1.1 第一卷 write -> 1.1.1 第一章 write",
        "design_dependent": "# 章节设计: 黑松林之变后续\n- 核心事件: 龙傲天检查敌人留下的令牌, 发现了一个神秘符号, 决心查明真相。",
        "search_dependent": "",
        "latest_text": "黑衣人头领见状, 眼中闪过一丝惊异, 随即果断下令: "撤！"",
    }
    messages = get_llm_messages(
        system_prompt=system_prompt_write,
        user_prompt=user_prompt_write,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.0,
        messages=messages
    )
    response = await llm_completion(llm_params, response_model=Inquiry)
    result = response.validated_data
    logger.info(f"LLM 生成的探询问题 (Write):\n{result.model_dump_json(indent=2, ensure_ascii=False)}")


@pytest.mark.asyncio
async def test_rag_query_search_prompt():
    task = Task(
        id="1.3",
        parent_id="1",
        task_type="search",
        hierarchical_position="全书",
        goal="研究中世纪海战的战术与船只类型。",
        category="story",
        language="cn",
        root_name="航海霸业",
        run_id="test_run_rag_query_search",
    )
    context = {
        "task": task.model_dump_json(indent=2),
        "task_list": "1. 全书 write -> 1.3 search 研究海战",
        "design_dependent": "# 故事背景\n- 时代设定在15世纪的大航海时代前期。",
        "search_dependent": "",
        "latest_text": "",
    }
    messages = get_llm_messages(
        system_prompt=system_prompt_search,
        user_prompt=user_prompt_search,
        context_dict_user=context
    )
    llm_params = get_llm_params(
        llm_group="reasoning",
        temperature=0.0,
        messages=messages
    )
    response = await llm_completion(llm_params, response_model=Inquiry)
    result = response.validated_data
    logger.info(f"LLM 生成的探询问题 (Search):\n{result.model_dump_json(indent=2, ensure_ascii=False)}")

