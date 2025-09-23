import os
import json
import hashlib
import asyncio
import re
from datetime import datetime
import sys
import threading
from loguru import logger
from diskcache import Cache
from typing import Dict, Any, List, Literal, Optional, Callable
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.sqlite_task import get_task_db
from utils.file import get_text_file_path, text_file_append, text_file_read
from utils.models import Task, get_sibling_ids_up_to_current
from utils.loader import load_prompts
from utils.kg import get_kg_query_engine, kg_add
from utils.hybrid_query import hybrid_query_batch
from utils.vector import index_query_batch, vector_add, get_vector_query_engine, index_query
from utils.file import cache_dir
from utils.llm import (
    llm_temperatures,
    get_llm_messages,
    get_llm_params,
    llm_completion
)
from story.base import get_story_kg_store, get_story_vector_store


class StoryRAG:

    def __init__(self):
        cache_story_dir = cache_dir / "story"
        os.makedirs(cache_story_dir, exist_ok=True)
        self.caches: Dict[str, Cache] = {
            'dependent_design': Cache(os.path.join(cache_story_dir, "dependent_design"), size_limit=int(32 * (1024**2))),
            'dependent_search': Cache(os.path.join(cache_story_dir, "dependent_search"), size_limit=int(32 * (1024**2))),
            'text_latest': Cache(os.path.join(cache_story_dir, "text_latest"), size_limit=int(32 * (1024**2))),
            'text_length': Cache(os.path.join(cache_story_dir, "text_length"), size_limit=int(1 * (1024**2))),
            'upper_design': Cache(os.path.join(cache_story_dir, "upper_design"), size_limit=int(128 * (1024**2))),
            'upper_search': Cache(os.path.join(cache_story_dir, "upper_search"), size_limit=int(128 * (1024**2))),
            'text_summary': Cache(os.path.join(cache_story_dir, "text_summary"), size_limit=int(128 * (1024**2))),
            'task_list': Cache(os.path.join(cache_story_dir, "task_list"), size_limit=int(32 * (1024**2))),
        }

    def save_data(self, task: Task, task_type: str):
        task_db = get_task_db(run_id=task.run_id)
        if task_type == "task_atom":
            if task.id == "1" or task.results.get("update_goal"):
                self.caches['task_list'].evict(tag=task.run_id)
                task_db.add_task(task)
            task_db.add_result(task)
        elif task_type == "task_plan":
            if task.results.get("plan"):
                task_db.add_result(task)
        elif task_type == "task_plan_reflection":
            if task.results.get("plan_reflection"):
                self.caches['task_list'].evict(tag=task.run_id)
                self.caches['upper_design'].evict(tag=task.run_id)
                self.caches['upper_search'].evict(tag=task.run_id)
                task_db.add_result(task)
                task_db.add_sub_tasks(task)
        elif task_type == "review_design":
            if task.results.get("review_design"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_design(task, task.results.get("review_design"))
        elif task_type == "review_write":
            if task.results.get("review_write"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_design(task, task.results.get("review_write"))
        elif task_type in [
            "task_design",
            "task_design_market",
            "task_design_title",
            "task_design_style",
            "task_design_review",
            "task_design_character",
            "task_design_system",
            "task_design_concept",
            "task_design_worldview",
            "task_design_plot",
            "task_design_general",
        ]:
            if task.results.get("design"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_design(task, task.results.get("design"))
        elif task_type == "task_design_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_design(task, task.results.get("design_reflection"))
        elif task_type == "task_hierarchy":
            if task.results.get("design"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
        elif task_type == "task_hierarchy_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_design(task, task.results.get("design_reflection"))
        elif task_type == "task_search":
            if task.results.get("search"):
                self.caches['dependent_search'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_search(task, task.results.get("search"))
        elif task_type == "task_write_before_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_design(task, task.results.get("design_reflection"))
        elif task_type == "task_write":
            if task.results.get("write"):
                task_db.add_result(task)
        elif task_type == "task_write_reflection":
            write_reflection = task.results.get("write_reflection")
            if write_reflection:
                self.caches['text_latest'].evict(tag=task.run_id)
                self.caches['text_length'].evict(tag=task.run_id)
                task_db.add_result(task)
                header_parts = [
                    task.id,
                    task.hierarchical_position,
                    task.goal,
                    task.length,
                ]
                header = " ".join(filter(None, header_parts))
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                content = f"## 任务\n{header}\n{timestamp}\n\n{write_reflection}"
                text_file_append(get_text_file_path(task), content)
                self.save_write(task, write_reflection)
        elif task_type == "task_summary":
            if task.results.get("summary"):
                self.caches['text_summary'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_summary(task, task.results.get("summary"))
        elif task_type == "task_aggregate_design":
            if task.results.get("design"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_design(task, task.results.get("design"))
        elif task_type == "task_aggregate_search":
            if task.results.get("search"):
                self.caches['dependent_search'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_search(task, task.results.get("search"))
        elif task_type == "task_aggregate_summary":
            if task.results.get("summary"):
                self.caches['text_summary'].evict(tag=task.run_id)
                task_db.add_result(task)
                self.save_summary(task, task.results.get("summary"))
        else:
            raise ValueError("不支持的任务类型")


    def save_design(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 design 内容 (向量索引与知识图谱)...")
        header_parts = [
            task.id,
            task.hierarchical_position,
            task.goal,
        ]
        header = " ".join(filter(None, header_parts))
        content = f"# 任务\n{header}\n\n{content}"
        vector_store = get_story_vector_store(task.run_id, "design")
        kg_store = get_story_kg_store(task.run_id, "design")
        doc_metadata = {
            "task_id": task.id,
            "hierarchical_position": task.hierarchical_position,
            "status": "active",                         # 状态, 用于标记/取消文档
            "created_at": datetime.now().isoformat()
        }
        vector_add(
            vector_store=vector_store,
            content=content,
            metadata=doc_metadata,
            content_format="md",
            doc_id=task.id
        )
        logger.info(f"[{task.id}] design 内容向量化完成, 开始构建知识图谱...")
        kg_add(
            kg_store=kg_store,
            content=content,
            metadata=doc_metadata,
            doc_id=task.id,
            content_format="md",
            chars_per_triplet=50, # 设计文档信息密度较高, 每100个字符提取一个三元组
            kg_extraction_prompt=load_prompts(task.category, "kg", "kg_extraction_prompt_design")[0]
        )
        logger.info(f"[{task.id}] design 内容存储完成。")


    def save_search(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 search 内容 (向量索引)...")
        header_parts = [task.id, task.hierarchical_position, task.goal]
        header = " ".join(filter(None, header_parts))
        full_content = f"# 任务\n{header}\n\n{content}"
        vector_store = get_story_vector_store(task.run_id, "search")
        doc_metadata = {
            "task_id": task.id,
            "created_at": datetime.now().isoformat()
        }
        vector_add(
            vector_store=vector_store,
            content=full_content,
            metadata=doc_metadata,
            content_format="md",
            doc_id=task.id
        )
        logger.info(f"[{task.id}] search 内容存储完成。")


    def save_write(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 write 内容 (构建知识图谱)...")
        kg_store = get_story_kg_store(task.run_id, "write")
        doc_metadata = {
            "task_id": task.id,
            "hierarchical_position": task.hierarchical_position,
            "created_at": datetime.now().isoformat()
        }
        kg_add(
            kg_store=kg_store,
            content=content,
            metadata=doc_metadata,
            doc_id=task.id,
            content_format="txt",
            chars_per_triplet=100, # 正文叙述信息密度较低, 每200个字符提取一个三元组
            kg_extraction_prompt=load_prompts(task.category, "kg", "kg_extraction_prompt_write")[0]
        )
        logger.info(f"[{task.id}] write 内容存储完成。")
    

    def save_summary(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 summary 内容 (向量索引)...")
        header_parts = [
            task.id,
            task.hierarchical_position,
            task.goal,
            task.length
        ]
        header = " ".join(filter(None, header_parts))
        full_content = f"# 任务\n{header}\n\n{content}"
        vector_store = get_story_vector_store(task.run_id, "summary")
        doc_metadata = {
            "task_id": task.id,
            "hierarchical_position": task.hierarchical_position,
            "created_at": datetime.now().isoformat()
        }
        vector_add(
            vector_store=vector_store,
            content=full_content,
            metadata=doc_metadata,
            content_format="md",
            doc_id=task.id
        )
        logger.info(f"[{task.id}] summary 内容存储完成。")


###############################################################################


    def get_context_base(self, task: Task) -> Dict[str, Any]:
        ret = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'parent_id', 'task_type', 'hierarchical_position', 'goal', 'length', 'dependency', 'instructions', 'input_brief', 'constraints', 'acceptance_criteria'}
            ),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if not task.parent_id:
            return ret
        task_db = get_task_db(run_id=task.run_id)
        dependent_design = self.get_dependent_design(task_db, task)
        dependent_search = self.get_dependent_search(task_db, task)
        text_latest = self.get_text_latest(task)
        task_list = ""
        if len(task.id.split(".")) >= 2:
            task_list = self.get_task_list(task_db, task)
        ret.update({
            "dependent_design": dependent_design,
            "dependent_search": dependent_search,
            "text_latest": text_latest,
            "task_list": task_list,
        })
        return ret
    

    async def get_context(self, task: Task) -> Dict[str, Any]:
        ret = self.get_context_base(task)
        if not task.parent_id:
            return ret
    
        dependent_design = ret.get("dependent_design", "")
        dependent_search = ret.get("dependent_search", "")
        text_latest = ret.get("text_latest", "")
        task_list = ret.get("task_list", "")

        tasks_to_run = {}
        current_level = len(task.id.split("."))
    
        if current_level >= 3:
            tasks_to_run["upper_design"] = self.get_upper_design(task, dependent_design, dependent_search, text_latest, task_list)
            tasks_to_run["upper_search"] = self.get_upper_search(task, dependent_design, dependent_search, text_latest, task_list)
    
        if len(text_latest) > 500:
            tasks_to_run["text_summary"] = self.get_text_summary(task, dependent_design, dependent_search, text_latest, task_list)
    
        if tasks_to_run:
            results = await asyncio.gather(*tasks_to_run.values())
            rag_results = dict(zip(tasks_to_run.keys(), results))
            ret.update(rag_results)
    
        return ret


    def get_dependent_design(self, task_db: Any, task: Task) -> str:
        cache_key = f"dependent_design:{task.run_id}:{task.id}"
        cached_result = self.caches['dependent_design'].get(cache_key)
        if cached_result is not None:
            return cached_result
        result = task_db.get_dependent_design(task)
        self.caches['dependent_design'].set(cache_key, result, tag=task.run_id)
        return result


    def get_dependent_search(self, task_db: Any, task: Task) -> str:
        cache_key = f"dependent_search:{task.run_id}:{task.id}"
        cached_result = self.caches['dependent_search'].get(cache_key)
        if cached_result is not None:
            return cached_result
        result = task_db.get_dependent_search(task)
        self.caches['dependent_search'].set(cache_key, result, tag=task.run_id)
        return result


    def get_task_list(self, task_db: Any, task: Task) -> str:
        cache_key = f"task_list:{task.run_id}:{task.parent_id}"
        cached_result = self.caches['task_list'].get(cache_key)
        if cached_result is not None:
            return cached_result
        result = task_db.get_task_list(task)
        self.caches['task_list'].set(cache_key, result, tag=task.run_id)
        return result


    def get_text_latest(self, task: Task, length: int = 500) -> str:
        key = f"get_text_latest:{task.run_id}:{length}"
        cached_result = self.caches['text_latest'].get(key)
        if cached_result is not None:
            return cached_result
        task_db = get_task_db(run_id=task.run_id)
        full_content = task_db.get_latest_write_reflection(length)
        if len(full_content) <= length:
            result = full_content
        else:
            # 为了避免截断一个完整的段落或句子, 我们尝试从一个换行符后开始截取
            start_pos = len(full_content) - length
            # 1. 尝试在截取点之后找到第一个换行符, 从该换行符后开始截取
            first_newline_after_start = full_content.find('\n', start_pos)
            if first_newline_after_start != -1:
                result = full_content[first_newline_after_start + 1:]
            else:
                # 2. 如果后面没有换行符(说明我们在最后一段), 则尝试在截取点之前找到最后一个换行符
                last_newline_before_start = full_content.rfind('\n', 0, start_pos)
                if last_newline_before_start != -1:
                    result = full_content[last_newline_before_start + 1:]
                else:
                    # 3. 如果全文都没有换行符, 或者只有一段很长的内容, 则直接硬截取
                    result = full_content[-length:]
        self.caches['text_latest'].set(key, result, tag=task.run_id)
        return result


    def get_text_length(self, task: Task) -> int:
        file_path = get_text_file_path(task)
        key = f"get_text_length:{file_path}"
        cached_result = self.caches['text_length'].get(key)
        if cached_result is not None:
            return cached_result
        full_content = text_file_read(file_path)
        length = len(full_content)
        self.caches['text_length'].set(key, length, tag=task.run_id)
        return length


    def get_aggregate_design(self, task: Task) -> Dict[str, Any]:
        ret = self.get_context_base(task)
        if task.sub_tasks:
            task_db = get_task_db(run_id=task.run_id)
            ret["subtask_design"] = task_db.get_subtask_design(task.id)
        return ret


    def get_aggregate_search(self, task: Task) -> Dict[str, Any]:
        ret = self.get_context_base(task)
        if task.sub_tasks:
            task_db = get_task_db(run_id=task.run_id)
            ret["subtask_search"] = task_db.get_subtask_search(task.id)
        return ret


    def get_aggregate_summary(self, task: Task) -> Dict[str, Any]:
        ret = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'hierarchical_position', 'goal', 'length', 'dependency', 'instructions', 'input_brief', 'constraints', 'acceptance_criteria'}
            ),
        }
        if task.sub_tasks:
            task_db = get_task_db(run_id=task.run_id)
            ret["subtask_summary"] = task_db.get_subtask_summary(task.id)
        return ret


    async def get_inquiry(
        self,
        task: Task,
        dependent_design: str,
        dependent_search: str,
        text_latest: str,
        task_list: str,
        inquiry_type: Literal['search', 'design', 'write']
    ) -> Dict[str, Any]:
        if inquiry_type == 'search':
            system_prompt, user_prompt, Inquiry = load_prompts(task.category, "rag_query", "system_prompt_search", "user_prompt_search", "Inquiry")
        elif inquiry_type == 'design':
            system_prompt, user_prompt, system_prompt_design_for_write, Inquiry = load_prompts(task.category, "rag_query", "system_prompt_design", "user_prompt_design", "system_prompt_design_for_write", "Inquiry")
            if task.task_type == 'write' and task.results.get("atom_result") == "atom":
                system_prompt = system_prompt_design_for_write
        elif inquiry_type == 'write':
            system_prompt, user_prompt, Inquiry = load_prompts(task.category, "rag_query", "system_prompt_write", "user_prompt_write", "Inquiry")
        else:
            raise ValueError(f"不支持的探询类型: {inquiry_type}")
        context_dict_user = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'parent_id', 'task_type', 'hierarchical_position', 'goal', 'length', 'dependency', 'instructions', 'input_brief', 'constraints', 'acceptance_criteria'}
            ),
            "dependent_design": dependent_design,
            "dependent_search": dependent_search,
            "text_latest": text_latest,
            "task_list": task_list
        }
        messages = get_llm_messages(system_prompt, user_prompt, None, context_dict_user)
        llm_params = get_llm_params(messages=messages, temperature=llm_temperatures["reasoning"])
        message = await llm_completion(llm_params, response_model=Inquiry)
        return message.validated_data.model_dump()
    
    
    async def get_upper_search(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取上层搜索(upper_search)上下文...")
        cache_key = f"get_upper_search:{task.run_id}:{task.id}"
        cached_result = self.caches['upper_search'].get(cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}] 命中上层搜索(upper_search)缓存。")
            return cached_result
        
        inquiry = await self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'search')
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过上层搜索。")
            return ""
        
        vector_store = get_story_vector_store(task.run_id, "search")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        filters = MetadataFilters(filters=active_filters) if active_filters else None
        
        query_engine = get_vector_query_engine(
            vector_store=vector_store,
            filters=filters,
            similarity_top_k=150,
            top_n=50,
        )

        results = await index_query_batch(query_engine, all_questions)
        result = "\n\n---\n\n".join(results)
        if not results:
            logger.warning(f"[{task.id}] 上层搜索(upper_search)未能生成答案或找到相关文档。")
        else:
            logger.success(f"[{task.id}] 上层搜索(upper_search)成功完成。")

        self.caches['upper_search'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层搜索(upper_search)上下文的流程已结束。")
        return result


    async def get_upper_design(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取上层设计(upper_design)上下文...")

        cache_key = f"get_upper_design:{task.run_id}:{task.id}"
        cached_result = self.caches['upper_design'].get(cache_key)
        if cached_result is not None:
            return cached_result
        
        inquiry = await self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'design')
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过上层设计检索。")
            return ""

        vector_store = get_story_vector_store(task.run_id, "design")
        kg_store = get_story_kg_store(task.run_id, "design")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        vector_filters = MetadataFilters(filters=active_filters) if active_filters else None

        # 创建向量查询引擎
        vector_query_engine = get_vector_query_engine(
            vector_store=vector_store,
            filters=vector_filters,
            similarity_top_k=150,
            top_n=30,
        )

        # 创建知识图谱查询引擎
        kg_query_engine = get_kg_query_engine(
            kg_store=kg_store,
            kg_similarity_top_k=600,
            top_n=100,
        )

        results = await hybrid_query_batch(
            vector_query_engine=vector_query_engine,
            kg_query_engine=kg_query_engine,
            questions=all_questions,
        )
        result = "\n\n---\n\n".join(results)

        self.caches['upper_design'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层设计(upper_design)上下文完成。")
        return result
 

    async def get_text_summary(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取历史情节概要(text_summary)上下文...")
        cache_key = f"get_text_summary:{task.run_id}:{task.id}"
        cached_result = self.caches['text_summary'].get(cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}] 命中历史情节概要(text_summary)缓存。")
            return cached_result

        inquiry = await self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'write')
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过历史情节概要检索。")
            return ""

        summary_vector_store = get_story_vector_store(task.run_id, "summary")
        kg_store = get_story_kg_store(task.run_id, "write")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        vector_filters = MetadataFilters(filters=active_filters) if active_filters else None

        # 创建摘要的向量查询引擎
        vector_query_engine = get_vector_query_engine(
            vector_store=summary_vector_store,
            filters=vector_filters,
            similarity_top_k=300,
            top_n=50,
        )

        # 创建正文的知识图谱查询引擎
        kg_query_engine = get_kg_query_engine(
            kg_store=kg_store,
            kg_similarity_top_k=600,
            top_n=100,
        )

        results = await hybrid_query_batch(
            vector_query_engine=vector_query_engine,
            kg_query_engine=kg_query_engine,
            questions=all_questions
        )
        result = "\n\n---\n\n".join(results)

        self.caches['text_summary'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取历史情节概要(text_summary)上下文完成。")
        return result


###############################################################################

_rag_instance = None
def get_story_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = StoryRAG()
    return _rag_instance
