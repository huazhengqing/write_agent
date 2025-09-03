#coding: utf8
import os
import json
import mem0
import litellm 
import hashlib
import asyncio
from loguru import logger
from diskcache import Cache
from mem0 import AsyncMemory
from typing import Dict, Any
from datetime import datetime
from util.models import Task
from util.llm import get_llm_params
from prompts.story.mem_cn import custom_fact_extraction_prompt, custom_update_memory_prompt, SYSTEM_PROMPT_design, USER_PROMPT_design, SYSTEM_PROMPT_text, USER_PROMPT_text, SYSTEM_PROMPT_search, USER_PROMPT_search




"""


分析、审查当前文件的代码，找出bug并改正， 指出可以优化的地方。


根据以上分析，改进建议， 请直接修改 文件，并提供diff。



 缓存 要改进
 缓存要分开，分为 正文   设计    2部分。
 正文的追加和提取，这是单独的

 设计的增加和提取，这也是单独的

缓存 不能是无限的，超过一定的大小，就不能一直变大了

"""


###############################################################################


class MemoryService:
    def __init__(self):
        logger.info("正在初始化 MemoryService...")
        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": int(os.getenv("embedding_dims"))
                }
            },
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": 'openrouter/deepseek/deepseek-chat-v3-0324:free',
                    "temperature": 0.0,
                    "max_tokens": 8000,
                    "caching": True,
                    "max_completion_tokens": 8000,
                    "timeout": 300,
                    "num_retries": 30,
                    "respect_retry_after": True,
                    "fallbacks": [
                        'openai/deepseek-ai/DeepSeek-V3',
                        'openrouter/deepseek/deepseek-r1-0528-qwen3-8b', 
                    ]
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "openai_base_url": os.getenv("embedder_BASE_URL"),
                    "api_key": os.getenv("embedder_API_KEY"),
                    "model": os.getenv("embedder_model"),
                    "embedding_dims": int(os.getenv("embedding_dims"))
                }
            },
            "graph_store": {
                "provider": "memgraph", 
                "config": {
                    "url": os.getenv("memgraph_url"),
                    "username": os.getenv("memgraph_username"),
                    "password": os.getenv("memgraph_password")
                }
            },
            "custom_fact_extraction_prompt": custom_fact_extraction_prompt, 
            "custom_update_memory_prompt": custom_update_memory_prompt
        }
        self.mem0 = AsyncMemory.from_config(config_dict=mem0_config)

        cache_dir = os.path.join("output", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.text_file_cache = Cache(os.path.join(cache_dir, "text_file"), size_limit=int(128 * (1024**2)))
        self.context_cache = Cache(os.path.join(cache_dir, "context"), size_limit=int(128 * (1024**2)))

        logger.info("MemoryService 初始化完成。")
 
###############################################################################


    async def add(self, task: Task, task_type: str):
        logger.info(f"开始向记忆中添加任务结果，任务ID: {task.id}, 任务类型: {task_type}")
        task_id = task.get("id")
        if not task_id:
            logger.error(f"任务信息中未找到任务ID: {task}")
            raise ValueError(f"任务信息中未找到任务ID {task_id} \n 任务信息: {task}")
        
        dependency = task.get("dependency", [])
        dependency_str = json.dumps(dependency, ensure_ascii=False) if dependency else "[]"

        category = ""
        content = ""
        task_result = task.results.get("result")
        if task_type == "task_atom":
            category = "task"
            if task.results.get("goal_update"):
                content = task.model_dump_json(indent=2, exclude_none=True)
        elif task_type == "task_plan":
            category = "task"
            content = task_result
        elif task_type == "task_execute":
            if task.task_type == "write":
                category = "text"
                content = task_result
                if content:
                    logger.info(f"任务 {task_id} 是 'write' 类型，将内容追加到文本文件。")
                    await asyncio.to_thread(self.text_file_append, self.get_text_file_path(task), content)
            elif task.task_type == "design":
                category = "design"
                content = task_result
            elif task.task_type == "search":
                category = "search"
                content = task_result
            else:
                logger.error(f"在 'task_execute' 中遇到不支持的任务类型: {task.task_type}")
                raise ValueError(f"不支持的任务类型: {task.task_type}")
        elif task_type == "task_aggregate":
            if task.task_type == "design":
                category = "design"
                content = task_result
            elif task.task_type == "search":
                category = "search"
                content = task_result
            else:
                logger.error(f"在 'task_aggregate' 中遇到不支持的任务类型: {task.task_type}")
                raise ValueError(f"不支持的任务类型: {task.task_type}")

        if not content:
            logger.warning(f"任务 {task_id} 的内容为空，不添加到记忆中。")
            return

        logger.debug(f"为任务 {task_id} 准备记忆元数据，类别: {category}")
        parent_task_id = ".".join(task_id.split(".")[:-1]) if task_id and "." in task_id else ""
        mem_metadata = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "hierarchy_level": len(task_id.split(".")),
            "parent_task_id": parent_task_id,
            "dependency": dependency_str,
        }
        await self.mem0.add(
            content,
            user_id=f"{task.run_id}",
            metadata=mem_metadata
        )
        logger.info(f"成功将任务 {task_id} 的内容添加到 mem0。")
        self.context_cache.evict(tag=f"{task.run_id}")
        logger.info(f"已为 run_id: {task.run_id} 清理上下文缓存。")


###############################################################################


    def text_file_append(self, file_path: str, content: str):
        logger.info(f"准备向文件追加内容: {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{content}")
            f.flush()
            os.fsync(f.fileno())
        logger.success(f"内容成功追加到文件: {file_path}")
        self.text_file_cache.evict(tag=file_path)
        logger.info(f"已为文件 {file_path} 清理文本文件缓存。")

    def text_file_read(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            logger.warning(f"尝试读取但文件不存在: {file_path}")
            return ""
        
        logger.info(f"正在读取文件: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_text_file_path(self, task: Task) -> str:
        return os.path.join("output", task.category, f"{task.root_name}.txt")
    
    async def get_text_latest(self, task: Task, length: int = 3000) -> str:
        file_path = self.get_text_file_path(task)
        key = f"get_text_latest:{file_path}:{length}"
        cached_result = self.text_file_cache.get(key)
        if cached_result is not None:
            logger.debug(f"从缓存中获取最新的文本内容，任务ID: {task.id}")
            return cached_result

        logger.info(f"缓存未命中，从文件读取最新的文本内容，任务ID: {task.id}")
        full_content = await asyncio.to_thread(self.text_file_read, file_path)
        if len(full_content) <= length:
            result = full_content
            logger.debug(f"全文内容长度小于 {length}，返回全部内容。")
        else:
            start_pos = len(full_content) - length
            first_newline_after_start = full_content.find('\n', start_pos)
            if first_newline_after_start != -1:
                result = full_content[first_newline_after_start + 1:]
            else:
                last_newline_before_start = full_content.rfind('\n', 0, start_pos)
                if last_newline_before_start != -1:
                    result = full_content[last_newline_before_start + 1:]
                else:
                    result = full_content[-length:]
            logger.debug(f"内容已截取，返回最后约 {length} 字符。")
        
        self.text_file_cache.set(key, result, tag=file_path)
        logger.info(f"已将最新的文本内容缓存，任务ID: {task.id}")
        return result


###############################################################################


    async def get_context(self, task: Task) -> Dict[str, Any]:
        """
        返回字段详解:
            - "task" (str): 当前任务对象的完整JSON表示。
            - "dependent_design" (str): 与当前任务同层级的其他“设计”任务的成果汇总。
            - "dependent_search" (str): 与当前任务同层级的其他“搜索”任务的结果汇总。
            - "text_latest" (str): 最终产出文本（如文章）的最新一部分内容。
            - "subtask_design" (str | None): (可选) 所有子任务产出的“设计”成果的汇总。
            - "subtask_search" (str | None): (可选) 所有子任务产出的“搜索”结果的汇总。
            - "upper_level_design" (str | None): (可选) 通过智能检索从所有“上级”任务中找到的相关“设计”信息。
            - "upper_level_search" (str | None): (可选) 通过智能检索从所有“上级”任务中找到的相关“搜索”结果。
            - "text_summary" (str | None): (可选) 对已生成的全部文本进行智能检索，找到与当前任务最相关的部分。
            - "task_list" (str | None): (可选) 从根任务到当前任务的完整任务链条，展示其在整个计划中的位置。
        """
        logger.info(f"开始为任务 {task.id} 获取上下文。")
        if not task.id:
            logger.error(f"任务信息中未找到任务ID: {task}")
            raise ValueError(f"任务信息中未找到任务ID {task.id} \n 任务信息: {task}")

        cache_key = f"context:{task.run_id}:{task.id}"
        cached_context = self.context_cache.get(cache_key)
        if cached_context:
            logger.debug(f"从缓存中获取上下文，任务ID: {task.id}")
            return cached_context

        ret = {
            "task": task.model_dump_json(indent=2, exclude_none=True)
        }

        # 并发获取初始依赖信息
        logger.info(f"并发获取任务 {task.id} 的初始依赖信息（设计、搜索、最新文本）。")
        dependent_design_task = self.get_dependent(task, "design")
        dependent_search_task = self.get_dependent(task, "search")
        text_latest_task = self.get_text_latest(task)
        dependent_design, dependent_search, text_latest = await asyncio.gather(
            dependent_design_task, dependent_search_task, text_latest_task
        )
        ret["dependent_design"] = dependent_design
        ret["dependent_search"] = dependent_search
        ret["text_latest"] = text_latest
        logger.debug(f"任务 {task.id} 的初始依赖信息获取完成。")

        if task.sub_tasks:
            logger.info(f"任务 {task.id} 存在子任务，开始获取子任务结果。")
            if task.task_type == "design":
                ret["subtask_design"] = await self.get_subtask_results(task, "design")
            if task.task_type == "search":
                ret["subtask_search"] = await self.get_subtask_results(task, "search")
            logger.debug(f"子任务结果获取完成。")

        # 为同层级的依赖信息添加标题，以便LLM更好地区分和理解
        dependent_parts = []
        if dependent_design:
            dependent_parts.append(f"### 同层级的设计成果:\n{dependent_design}")
        if dependent_search:
            dependent_parts.append(f"### 同层级的搜索结果:\n{dependent_search}")
        dependent = "\n\n".join(dependent_parts)

        # 并发执行所有后续的上下文获取任务
        logger.info(f"开始为任务 {task.id} 并发执行后续的上下文获取任务。")
        context_tasks = {}
        current_level = len(task.id.split("."))
        if current_level >= 3:
            context_tasks["upper_level_design"] = self._get_and_search_context(task, "design", dependent, text_latest)
            context_tasks["upper_level_search"] = self._get_and_search_context(task, "search", dependent, text_latest)
        if text_latest and len(text_latest) > 500:
            context_tasks["text_summary"] = self._get_and_search_context(task, "text", dependent, text_latest)
        if current_level >= 2:
            context_tasks["task_list"] = self.get_task_list(task)
        if context_tasks:
            coroutines = list(context_tasks.values())
            results = await asyncio.gather(*coroutines)
            for i, key in enumerate(context_tasks.keys()):
                ret[key] = results[i]
        logger.debug(f"任务 {task.id} 的所有上下文获取任务完成。")

        self.context_cache.set(cache_key, ret, tag=f"{task.run_id}")
        logger.success(f"已为任务 {task.id} 生成并缓存上下文。")
        return ret

    async def get_dependent(self, task: Task, category: str) -> str:
        logger.info(f"获取任务 {task.id} 的同层级依赖，类别: {category}")
        current_level = len(task.id.split("."))
        filters = {
            "category": category,
            "hierarchy_level": current_level
        }
        logger.debug(f"使用过滤器查询 mem0: {filters}")
        all_memories = await self.mem0.get_all(
            user_id=f"{task.run_id}", 
            filters=filters
            )
        if not all_memories:
            logger.info(f"未找到任务 {task.id} 的同层级依赖（类别: {category}）。")
            result = ""
        else:
            logger.info(f"找到 {len(all_memories)} 条同层级依赖（类别: {category}），正在合并内容。")
            content_list = [item.get("memory", "") for item in all_memories if item.get("memory")]
            result = "\n\n".join(content_list)
        
        return result
    
    async def get_subtask_results(self, task: Task, category: str) -> str:
        logger.info(f"获取任务 {task.id} 的子任务结果，类别: {category}")
        parent_level = len(task.id.split("."))
        filters = {
            "category": category,
            "hierarchy_level": {"gt": parent_level} 
        }
        results = await self.mem0.get_all(
            user_id=f"{task.run_id}",
            limit=300,
            filters=filters
        )
        if not results:
            logger.info(f"未找到任务 {task.id} 的子任务结果（类别: {category}）。")
            result = ""
        else:
            logger.debug(f"从 mem0 粗略获取了 {len(results)} 条可能的子任务结果，正在进行精确过滤。")
            # 在代码中再次精确过滤，确保它们确实是当前任务的子任务（ID前缀匹配）
            subtask_prefix = f"{task.id}."
            content_list = [item.get("memory", "") for item in results if item.get("memory") and item.get("metadata", {}).get("task_id", "").startswith(subtask_prefix)]
            logger.info(f"精确过滤后，找到 {len(content_list)} 条子任务结果（类别: {category}），正在合并内容。")
            result = "\n\n".join(content_list)

        return result

    async def get_query(self, task: Task, category: str, dependent_results:str, text_latest: str) -> list:
        logger.info(f"为任务 {task.id} 生成上下文检索查询，类别: {category}")
        PROMPTS = {
            "design": (SYSTEM_PROMPT_design, USER_PROMPT_design),
            "text": (SYSTEM_PROMPT_text, USER_PROMPT_text),
            "search": (SYSTEM_PROMPT_search, USER_PROMPT_search),
        }
        if category not in PROMPTS:
            logger.error(f"不支持的查询生成类别: {category}")
            raise ValueError(f"不支持的查询生成类别: {category}")

        prompt_input = {
            "task": task.model_dump_json(indent=2, exclude_none=True),
            "dependent_results": dependent_results, 
            "text_latest": text_latest
        }
        logger.debug(f"用于生成查询的LLM输入: {prompt_input}")
        system_prompt, user_prompt_template = PROMPTS[category]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format(**prompt_input)}
        ]
        llm_params = get_llm_params(messages, temperature=0.1)
        llm_params['response_format'] = {"type": "json_object"}
        response = await litellm.acompletion(**llm_params)
        logger.debug(f"LLM 返回查询生成结果: {response.choices[0].message.content}")
        response_content = response.choices[0].message.content
        queries_dict = json.loads(response_content)
        keywords = queries_dict.get("keywords", [])
        result = keywords if isinstance(keywords, list) else []
        logger.info(f"为任务 {task.id} 成功生成 {len(result)} 个查询关键词。")
        return result

    async def _get_and_search_context(self, task: Task, category: str, dependent_results: str, text_latest: str) -> str:
        logger.info(f"开始为任务 {task.id} 获取并搜索上下文，类别: {category}")
        keywords = await self.get_query(task, category, dependent_results, text_latest)
        if not keywords:
            logger.warning(f"未为任务 {task.id}（类别: {category}）生成任何查询关键词，返回空上下文。")
            result = ""
        else:
            logger.info(f"使用关键词 {keywords} 为任务 {task.id} 搜索上层结果。")
            result = await self.search_upper_level_results(task, category, keywords)
        return result


    async def search_upper_level_results(self, task: Task, category: str, keywords: list) -> str:
        logger.info(f"开始为任务 {task.id} 搜索上层结果，类别: {category}, 关键词: {keywords}")
        filters = {
            "category": category
        }
        if category in ["design", "search"]:
            current_level = len(task.id.split("."))
            filters["hierarchy_level"] = {"gte": 1, "lt": current_level}
        
        logger.debug(f"搜索上层结果的过滤器: {filters}")
        unique_queries = list(dict.fromkeys([q.strip() for q in keywords if q and q.strip()]))
        if not unique_queries:
            logger.warning(f"没有有效的唯一查询关键词，搜索中止。")
            result = ""
        else:
            query_str = " OR ".join([f"({q})" for q in unique_queries])
            logger.info(f"执行 mem0 搜索，查询: '{query_str}'")
            results = await self.mem0.search(
                query=query_str,
                user_id=f"{task.run_id}",
                limit=300,
                filters=filters
            )
            if not results:
                logger.info(f"对于查询 '{query_str}'，mem0 未返回任何结果。")
                result = ""
            else:
                logger.info(f"mem0 返回 {len(results)} 条搜索结果，正在合并内容。")
                content_list = [item.get("memory", "") for item in results if item.get("memory")]
                result = "\n\n".join(content_list)
        
        return result


    """
    获取从根任务到当前任务的整个任务链信息。
    任务ID格式为 "父任务ID.子任务序号"，根任务ID为 "1"。
    例如，对于任务ID "1.3.5"，此函数将按顺序获取 "1"、"1.3" 和 "1.3.5" 的完整任务信息。
    """
    async def get_task_list(self, task: Task) -> str:
        logger.info(f"正在为任务 {task.id} 获取任务链信息。")
        id_parts = task.id.split('.')
        hierarchical_ids = [".".join(id_parts[:i+1]) for i in range(len(id_parts))]
        if not hierarchical_ids:
            return ""
        filters = {
            "category": "task",
            "task_id": {"in": hierarchical_ids}
        }
        logger.debug(f"使用过滤器查询 mem0 以获取任务链: {filters}")
        results = await self.mem0.get_all(
            user_id=f"{task.run_id}",
            limit=100, 
            filters=filters
        )
        if not results:
            logger.warning(f"未找到任务 {task.id} 的任何历史任务信息。")
            result = ""
        else:
            memory_map = {item.get("metadata", {}).get("task_id"): item.get("memory") for item in results}

            task_summaries = []
            for task_id in hierarchical_ids:
                memory_content = memory_map.get(task_id)
                if not memory_content:
                    continue
                task_summaries.append(memory_content)
            result = "\n\n".join(task_summaries)
            logger.info(f"成功构建了任务 {task.id} 的任务链信息。")
        
        return result


###############################################################################


memory = MemoryService()
