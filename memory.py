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


class MemoryService:
    def __init__(self):
        logger.info("正在初始化 MemoryService...")
        embedding_dims_str = os.getenv("embedding_dims")
        if not embedding_dims_str or not embedding_dims_str.isdigit():
            raise ValueError("环境变量 'embedding_dims' 未设置或不是一个有效的整数。")
        embedding_dims = int(embedding_dims_str)
        self.mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": embedding_dims
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
                    "embedding_dims": embedding_dims
                }
            },
            "graph_store": {
                "provider": "memgraph", 
                "config": {
                    "url": os.getenv("memgraph_url"),
                    "username": os.getenv("memgraph_username"),
                    "password": os.getenv("memgraph_password")
                }
            }
        }
        self.mem0_instances: Dict[str, AsyncMemory] = {}

        cache_dir = os.path.join("output", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.text_file_cache = Cache(os.path.join(cache_dir, "text_file"), size_limit=int(128 * (1024**2)))
        self.context_cache = Cache(os.path.join(cache_dir, "context"), size_limit=int(128 * (1024**2)))

        logger.info("MemoryService 初始化完成。")
 
    def get_mem0(self, category: str) -> AsyncMemory:
        if category in self.mem0_instances:
            return self.mem0_instances[category]

        config = self.mem0_config.copy()
        if category == "story":
            from prompts.story.mem_cn import custom_fact_extraction_prompt, custom_update_memory_prompt
            config["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt
            config["custom_update_memory_prompt"] = custom_update_memory_prompt
        elif category == "book":
            pass
        elif category == "report":
            pass
        else:
            logger.error(f"不支持的任务类型: {category}")
            raise ValueError(f"不支持的任务类型: {category}")

        instance = AsyncMemory.from_config(config_dict=config)
        self.mem0_instances[category] = instance
        return instance


###############################################################################


    async def add(self, task: Task, task_type: str):
        if not task.id or task.id == "0":
            logger.error(f"任务信息中未找到任务ID: {task}")
            raise ValueError(f"任务信息中未找到任务ID {task.id} \n 任务信息: {task}")

        logger.info(f"开始向记忆中添加任务结果, 任务ID: {task.id}, 任务类型: {task_type}")
        
        task_result = task.results.get("result")
        if not task_result:
            logger.error(f"task_result为空 {task} {task_type}")
            raise ValueError(f"task_result为空 {task} {task_type}")
        
        category = ""
        content = ""
        if task_type == "task_atom":
            if task.results.get("goal_update"):
                category = "task"
                content = task.model_dump_json(indent=2, exclude_none=True)
        elif task_type == "task_plan":
            category = "task"
            content = task_result
        elif task_type == "task_execute":
            if task.task_type == "write":
                category = "text"
                content = task_result
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
            logger.info(f"任务 {task.id} 的内容为空, 不添加到记忆中。")
            return

        logger.debug(f"为任务 {task.id} 准备记忆元数据, 类别: {category}")
        mem_metadata = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "task_id": task.id,
            "hierarchy_level": len(task.id.split(".")),
            "parent_id": task.parent_id or "",
            "dependency": json.dumps(task.dependency, ensure_ascii=False),
        }
        await self.get_mem0(task.category).add(
            content,
            user_id=f"{task.run_id}",
            metadata=mem_metadata
        )
        logger.info(f"成功将任务 {task.id} 的内容添加到 mem0。")

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
        return os.path.join("output", task.category, f"{task.run_id}.txt")
    
    async def get_text_latest(self, task: Task, length: int = 3000) -> str:
        file_path = self.get_text_file_path(task)
        key = f"get_text_latest:{file_path}:{length}"
        cached_result = self.text_file_cache.get(key)
        if cached_result is not None:
            logger.debug(f"从缓存中获取最新的文本内容, 任务ID: {task.id}")
            return cached_result

        logger.info(f"缓存未命中, 从文件读取最新的文本内容, 任务ID: {task.id}")
        full_content = await asyncio.to_thread(self.text_file_read, file_path)
        if len(full_content) <= length:
            result = full_content
            logger.debug(f"全文内容长度小于 {length}, 返回全部内容。")
        else:
            # 为了避免截断一个完整的段落或句子, 我们尝试从一个换行符后开始截取
            start_pos = len(full_content) - length
            # 1. 尝试在截取点之后找到第一个换行符, 从该换行符后开始截取
            first_newline_after_start = full_content.find('\n', start_pos)
            if first_newline_after_start != -1:
                result = full_content[first_newline_after_start + 1:]
            else:
                # 2. 如果后面没有换行符（说明我们在最后一段）, 则尝试在截取点之前找到最后一个换行符
                last_newline_before_start = full_content.rfind('\n', 0, start_pos)
                if last_newline_before_start != -1:
                    result = full_content[last_newline_before_start + 1:]
                else:
                    # 3. 如果全文都没有换行符, 或者只有一段很长的内容, 则直接硬截取
                    result = full_content[-length:]
            logger.debug(f"内容已截取, 返回最后约 {length} 字符。")
        
        self.text_file_cache.set(key, result, tag=file_path)
        logger.info(f"已将最新的文本内容缓存, 任务ID: {task.id}")
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
            - "text_summary" (str | None): (可选) 对已生成的全部文本进行智能检索, 找到与当前任务最相关的部分。
            - "task_list" (str | None): (可选) 从根任务到当前任务的完整任务链条, 展示其在整个计划中的位置。
        """
        logger.info(f"开始为任务 {task.id} 获取上下文。")
        if not task.id:
            logger.error(f"任务信息中未找到任务ID: {task}")
            raise ValueError(f"任务信息中未找到任务ID {task.id} \n 任务信息: {task}")

        cache_key = f"context:{task.run_id}:{task.id}"
        cached_context = self.context_cache.get(cache_key)
        if cached_context:
            logger.debug(f"从缓存中获取上下文, 任务ID: {task.id}")
            return cached_context

        ret = {
            "task": task.model_dump_json(indent=2, exclude_none=True), 
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            logger.info(f"任务 {task.id} 存在子任务, 开始获取子任务结果。")
            if task.task_type == "design":
                ret["subtask_design"] = await self.get_subtask_results(task, "design")
            if task.task_type == "search":
                ret["subtask_search"] = await self.get_subtask_results(task, "search")
            logger.debug(f"子任务结果获取完成。")

        # 为同层级的依赖信息添加标题, 以便LLM更好地区分和理解
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
        logger.info(f"获取任务 {task.id} 的同层级依赖, 类别: {category}")
        if not task.parent_id:
            logger.info(f"任务 {task.id} 没有父任务, 无同层级依赖。")
            return ""
        filters = {
            "category": category,
            "parent_id": task.parent_id
        }
        all_memories = await self.get_mem0(task.category).get_all(
            user_id=f"{task.run_id}", 
            filters=filters
            )
        if not all_memories:
            logger.info(f"未找到任务 {task.id} 的同层级依赖（类别: {category}）。")
            result = ""
        else:
            logger.info(f"找到 {len(all_memories)} 条同层级依赖（类别: {category}）, 正在合并内容。")
            content_list = [item.get("memory", "") for item in all_memories if item.get("memory")]
            result = "\n\n".join(content_list)
        
        return result
    
    async def get_by_parent_id(self, run_id: str, parent_id: str, data_category: str, main_category: str) -> str:
        if not parent_id:
            logger.debug("get_by_parent_id 调用时 parent_id 为空, 返回空字符串。")
            return ""
        filters = {
            "category": data_category,
            "parent_id": parent_id
        }
        all_memories = await self.get_mem0(main_category).get_all(
            user_id=run_id, 
            filters=filters
            )
        if not all_memories:
            return ""
        content_list = [item.get("memory", "") for item in all_memories if item.get("memory")]
        return "\n\n".join(content_list)
    
    async def get_subtask_results(self, task: Task, category: str) -> str:
        logger.info(f"正在为任务 {task.id} 获取 '{category}' 类的子任务结果。")
        result = await self.get_by_parent_id(task.run_id, task.id, category, task.category)
        if not result:
            logger.info(f"未找到任务 {task.id} 的 '{category}' 类子任务结果。")
        else:
            logger.info(f"成功获取任务 {task.id} 的 '{category}' 类子任务结果, 内容长度: {len(result)}")
        return result

    async def get_query(self, task: Task, category: str, dependent_results:str, text_latest: str) -> list:
        logger.info(f"为任务 {task.id} 生成上下文检索查询, 类别: {category}")

        if task.category == "story":    
            from prompts.story.mem_cn import SYSTEM_PROMPT_design, USER_PROMPT_design, SYSTEM_PROMPT_text, USER_PROMPT_text, SYSTEM_PROMPT_search, USER_PROMPT_search
            PROMPTS = {
                "design": (SYSTEM_PROMPT_design, USER_PROMPT_design),
                "text": (SYSTEM_PROMPT_text, USER_PROMPT_text),
                "search": (SYSTEM_PROMPT_search, USER_PROMPT_search),
            }
        elif task.category == "book":
            raise ValueError(f"不支持的任务类型: {task.category}")
        elif task.category == "report":
            raise ValueError(f"不支持的任务类型: {task.category}")
        else:
            raise ValueError(f"不支持的任务类型: {task.category}")
        
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
        logger.info(f"开始为任务 {task.id} 获取并搜索上下文, 类别: {category}")
        keywords = await self.get_query(task, category, dependent_results, text_latest)
        if not keywords:
            logger.warning(f"未为任务 {task.id}（类别: {category}）生成任何查询关键词, 返回空上下文。")
            result = ""
        else:
            logger.info(f"使用关键词 {keywords} 为任务 {task.id} 搜索上层结果。")
            result = await self.search_upper_level_results(task, category, keywords)
        return result


    async def search_upper_level_results(self, task: Task, category: str, keywords: list) -> str:
        logger.info(f"开始为任务 {task.id} 搜索上层结果, 类别: {category}, 关键词: {keywords}")
        filters = {
            "category": category
        }
        if category in ["design", "search"]:
            current_level = len(task.id.split("."))
            filters["hierarchy_level"] = {"gte": 1, "lt": current_level}
        
        logger.debug(f"搜索上层结果的过滤器: {filters}")
        unique_queries = list(dict.fromkeys([q.strip() for q in keywords if q and q.strip()]))
        if not unique_queries:
            logger.error(f"没有有效的唯一查询关键词, 搜索中止。")
            return ""

        # 并行执行所有关键词的搜索
        search_tasks = [
            self.get_mem0(task.category).search(
                query=q,
                user_id=f"{task.run_id}",
                limit=300,
                filters=filters
            ) for q in unique_queries
        ]
        logger.info(f"为 {len(unique_queries)} 个关键词并行执行 mem0 搜索...")
        list_of_results = await asyncio.gather(*search_tasks)

        # 合并并基于ID去重结果
        all_results = [item for sublist in list_of_results for item in sublist]
        if not all_results:
            logger.error(f"对于关键词 {unique_queries}, mem0 未返回任何结果。")
            return ""

        logger.info(f"mem0 共返回 {len(all_results)} 条原始搜索结果, 正在去重和合并内容。")
        seen_ids = set()
        unique_content_list = []
        for item in all_results:
            if item.get("id") not in seen_ids and item.get("memory"):
                unique_content_list.append(item["memory"])
                seen_ids.add(item["id"])

        logger.info(f"合并后得到 {len(unique_content_list)} 条唯一结果。")
        return "\n\n".join(unique_content_list)


    """
    获取任务上下文, 包括两部分:
    1. 父任务链: 从根任务到当前任务的父任务的完整任务信息。
    2. 当前层级任务: 与当前任务同层级的所有任务(兄弟任务)的信息。

    任务ID格式为 "父任务ID.子任务序号", 根任务ID为 "1"。
    例如, 对于任务ID "1.3.5", 此函数将:
    - 获取父任务链 "1" 和 "1.3" 的信息。
    - 获取所有父ID为 "1.3" 的任务信息。
    """
    async def get_task_list(self, task: Task) -> str:
        logger.info(f"正在为任务 {task.id} 获取任务链和同级任务信息。")
        id_parts = task.id.split('.')

        # 1. 获取父任务链 (不包括当前任务)
        parent_chain_ids = [".".join(id_parts[:i+1]) for i in range(len(id_parts) - 1)]
        
        parent_chain_summaries = []
        if parent_chain_ids:
            filters = {
                "category": "task", 
                "task_id": {"in": parent_chain_ids}
            }
            logger.debug(f"使用过滤器查询 mem0 以获取父任务链: {filters}")
            results = await self.get_mem0(task.category).get_all(
                user_id=f"{task.run_id}",
                limit=100, 
                filters=filters
            )
            if results:
                # 按顺序构建父任务链
                memory_map = {item.get("metadata", {}).get("task_id"): item.get("memory") for item in results}
                for task_id in parent_chain_ids:
                    memory_content = memory_map.get(task_id)
                    if memory_content:
                        parent_chain_summaries.append(memory_content)
        
        # 2. 获取当前层级的所有任务
        current_level_tasks_str = await self.get_by_parent_id(task.run_id, task.parent_id, "task", task.category)

        # 3. 组合结果
        task_list_parts = []
        if parent_chain_summaries:
            parent_chain_str = "\n\n".join(parent_chain_summaries)
            task_list_parts.append(f"### 父任务链:\n{parent_chain_str}")
            logger.info(f"成功构建了任务 {task.id} 的父任务链信息。")

        if current_level_tasks_str:
            task_list_parts.append(f"### 当前层级所有任务:\n{current_level_tasks_str}")
            logger.info(f"成功获取了任务 {task.id} 的同级任务信息。")

        result = "\n\n".join(task_list_parts)

        if not result:
            logger.warning(f"未找到任务 {task.id} 的任何历史或同级任务信息。")
        else:
            logger.info(f"成功为任务 {task.id} 构建了任务上下文。")
        
        return result


###############################################################################


memory = MemoryService()


###############################################################################


async def get_llm_messages(task: Task, SYSTEM_PROMPT: str, USER_PROMPT: str, context_dict: Dict[str, Any] = None) -> list[dict]:
    if not task.id or not task.goal:
        raise ValueError("任务ID和目标不能为空。")

    context = await memory.get_context(task)
    if context_dict:
        context.update(context_dict)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(**context)}
    ]
    return messages
