import os
import json
import hashlib
import re
from datetime import datetime
import sys
import threading
from loguru import logger
from diskcache import Cache
from typing import Dict, Any, List, Literal, Optional, Callable
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.sqlite import get_task_db
from utils.file import get_text_file_path, text_file_append, text_file_read
from utils.models import Task, get_sibling_ids_up_to_current
from utils.prompt_loader import load_prompts
from utils.kg import get_kg_store, kg_add, hybrid_query
from utils.vector import vector_add, vector_query, get_vector_store
from utils.file import cache_dir, data_dir
from utils.llm import (
    LLM_TEMPERATURES,
    get_llm_messages,
    get_llm_params,
    llm_completion
)


class story_rag:

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

        self.vector_stores: Dict[str, ChromaVectorStore] = {}
        self.graph_stores: Dict[str, KuzuGraphStore] = {}
        self._storage_lock = threading.Lock()


    def _get_vector_store(self, run_id: str, content_type: str) -> ChromaVectorStore:
        context_key = f"{run_id}_{content_type}"
        if context_key in self.vector_stores:
            return self.vector_stores[context_key]

        with self._storage_lock:
            if context_key in self.vector_stores:
                return self.vector_stores[context_key]

            chroma_path = os.path.join(data_dir, run_id, content_type)
            collection_name = f"{run_id}_{content_type}"
            vector_store = get_vector_store(db_path=chroma_path, collection_name=collection_name)
            
            self.vector_stores[context_key] = vector_store
            return vector_store


    def _get_graph_store(self, run_id: str, content_type: str) -> KuzuGraphStore:
        context_key = f"{run_id}_{content_type}"
        if context_key in self.graph_stores:
            return self.graph_stores[context_key]

        with self._storage_lock:
            if context_key in self.graph_stores:
                return self.graph_stores[context_key]

            kuzu_db_path = os.path.join(data_dir, run_id, content_type)
            graph_store = get_kg_store(db_path=kuzu_db_path)
            
            self.graph_stores[context_key] = graph_store
            return graph_store


    def save_data(self, task: Task, task_type: str):
        task_db = get_task_db(run_id=task.run_id)
        if task_type == "task_atom":
            if task.id == "1" or task.results.get("goal_update"):
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
        vector_store = self._get_vector_store(task.run_id, "design")
        graph_store = self._get_graph_store(task.run_id, "design")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            graph_store=graph_store
        )
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
            content_format="markdown",
            doc_id=task.id
        )
        logger.info(f"[{task.id}] design 内容向量化完成, 开始构建知识图谱...")
        kg_add(
            storage_context=storage_context,
            content=content,
            metadata=doc_metadata,
            doc_id=task.id,
            kg_extraction_prompt=load_prompts(task.category, "graph_cn", "kg_extraction_prompt_design")[0],
            content_format="markdown",
            max_triplets_per_chunk=15,
        )
        logger.info(f"[{task.id}] design 内容存储完成。")


    def save_search(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 search 内容 (向量索引)...")
        header_parts = [task.id, task.hierarchical_position, task.goal]
        header = " ".join(filter(None, header_parts))
        full_content = f"# 任务\n{header}\n\n{content}"
        vector_store = self._get_vector_store(task.run_id, "search")
        doc_metadata = {
            "task_id": task.id,
            "created_at": datetime.now().isoformat()
        }
        vector_add(
            vector_store=vector_store,
            content=full_content,
            metadata=doc_metadata,
            content_format="markdown",
            doc_id=task.id
        )
        logger.info(f"[{task.id}] search 内容存储完成。")


    def save_write(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 write 内容 (构建知识图谱)...")
        graph_store = self._get_graph_store(task.run_id, "write")
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        doc_metadata = {
            "task_id": task.id,
            "hierarchical_position": task.hierarchical_position,
            "created_at": datetime.now().isoformat()
        }
        kg_add(
            storage_context=storage_context,
            content=content,
            metadata=doc_metadata,
            doc_id=task.id,
            kg_extraction_prompt=load_prompts(task.category, "graph_cn", "kg_extraction_prompt_write")[0],
            content_format="text",
            max_triplets_per_chunk=15,
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
        vector_store = self._get_vector_store(task.run_id, "summary")
        doc_metadata = {
            "task_id": task.id,
            "hierarchical_position": task.hierarchical_position,
            "created_at": datetime.now().isoformat()
        }
        vector_add(
            vector_store=vector_store,
            content=full_content,
            metadata=doc_metadata,
            content_format="markdown",
            doc_id=task.id
        )
        logger.info(f"[{task.id}] summary 内容存储完成。")


###############################################################################


    def get_context_base(self, task: Task) -> Dict[str, Any]:
        ret = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'parent_id', 'task_type', 'hierarchical_position', 'goal', 'length', 'dependency'}
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
    

    def get_context(self, task: Task) -> Dict[str, Any]:
        ret = self.get_context_base(task)
        if not task.parent_id:
            return ret
        dependent_design = ret.get("dependent_design", "")
        dependent_search = ret.get("dependent_search", "")
        text_latest = ret.get("text_latest", "")
        task_list = ret.get("task_list", "")
        rag_results = {}
        current_level = len(task.id.split("."))
        if current_level >= 3:
            rag_results["upper_design"] = self.get_upper_design(task, dependent_design, dependent_search, text_latest, task_list)
            rag_results["upper_search"] = self.get_upper_search(task, dependent_design, dependent_search, text_latest, task_list)
        if len(text_latest) > 500:
            rag_results["text_summary"] = self.get_text_summary(task, dependent_design, dependent_search, text_latest, task_list)
        if rag_results:
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


    def get_text_latest(self, task: Task, length: int = 3000) -> str:
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
                include={'id', 'hierarchical_position', 'goal', 'length'}
            ),
        }
        if task.sub_tasks:
            task_db = get_task_db(run_id=task.run_id)
            ret["subtask_summary"] = task_db.get_subtask_summary(task.id)
        return ret


    def get_inquiry(
        self,
        task: Task,
        dependent_design: str,
        dependent_search: str,
        text_latest: str,
        task_list: str,
        inquiry_type: Literal['search', 'design', 'write']
    ) -> Dict[str, Any]:
        if inquiry_type == 'search':
            system_prompt, user_prompt, Inquiry = load_prompts(task.category, "query_cn", "system_prompt_search", "user_prompt_search", "Inquiry")
        elif inquiry_type == 'design':
            system_prompt, user_prompt, system_prompt_design_for_write, Inquiry = load_prompts(task.category, "query_cn", "system_prompt_design", "user_prompt_design", "system_prompt_design_for_write", "Inquiry")
            if task.task_type == 'write' and task.results.get("atom_result") == "atom":
                system_prompt = system_prompt_design_for_write
        elif inquiry_type == 'write':
            system_prompt, user_prompt, Inquiry = load_prompts(task.category, "query_cn", "system_prompt_write", "user_prompt_write", "Inquiry")
        else:
            raise ValueError(f"不支持的探询类型: {inquiry_type}")
        context_dict_user = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'task_type', 'hierarchical_position', 'goal', 'length'}
            ),
            "dependent_design": dependent_design,
            "dependent_search": dependent_search,
            "text_latest": text_latest,
            "task_list": task_list
        }
        messages = get_llm_messages(system_prompt, user_prompt, None, context_dict_user)
        llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
        message = llm_completion(llm_params, response_model=Inquiry)
        return message.validated_data.model_dump()
    
    
    def get_upper_search(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取上层搜索(upper_search)上下文...")
        cache_key = f"get_upper_search:{task.run_id}:{task.id}"
        cached_result = self.caches['upper_search'].get(cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}] 命中上层搜索(upper_search)缓存。")
            return cached_result
        
        inquiry = self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'search')
        if not inquiry or not inquiry.get("questions"):
            logger.warning(f"[{task.id}] 生成的探询问题为空或无效, 跳过上层搜索。")
            return ""
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 探询列表中没有问题, 无法执行上层搜索查询。")
            return ""
        single_question = "\n".join(all_questions)
        logger.info(f"[{task.id}] 整合后的上层搜索查询问题:\n{single_question}")
        
        vector_store = self._get_vector_store(task.run_id, "search")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        
        filters = MetadataFilters(filters=active_filters) if active_filters else None
        logger.info(f"[{task.id}] 上层搜索使用的过滤器: {filters.to_dict() if filters else 'None'}")

        answer, _ = vector_query(
            vector_store=vector_store,
            query_text=single_question,
            filters=filters,
            rerank_top_n=4,
        )

        result = answer or ""
        if not result:
            logger.warning(f"[{task.id}] 上层搜索(upper_search)未能生成答案或找到相关文档。")
        else:
            logger.success(f"[{task.id}] 上层搜索(upper_search)成功完成，并生成了综合答案。")

        self.caches['upper_search'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层搜索(upper_search)上下文的流程已结束。")
        return result


    def get_upper_design(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取上层设计(upper_design)上下文...")

        cache_key = f"get_upper_design:{task.run_id}:{task.id}"
        cached_result = self.caches['upper_design'].get(cache_key)
        if cached_result is not None:
            return cached_result
        
        inquiry = self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'design')
        if not inquiry or not inquiry.get("questions"):
            logger.warning(f"[{task.id}] 生成的探询问题为空或无效, 跳过上层设计检索。")
            return ""
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 探询列表中没有问题, 无法执行上层设计查询。")
            return ""

        vector_store = self._get_vector_store(task.run_id, "design")
        graph_store = self._get_graph_store(task.run_id, "design")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        vector_filters = MetadataFilters(filters=active_filters) if active_filters else None

        kg_gen_query_prompt = load_prompts(task.category, "rag_cn", "kg_gen_query_prompt")[0]
        kg_query_gen_prompt = PromptTemplate(template=kg_gen_query_prompt)

        final_instruction = "角色: 首席故事架构师。\n任务: 整合上层设计, 提炼统一、无冲突的宏观设定和指导原则。"
        rules_text = """
# 整合规则
1.  时序优先: 结果按时间倒序。遇直接矛盾, 采纳最新版本。
2.  矛盾 vs. 细化:
    - 矛盾: 无法共存的描述 (A是孤儿 vs A父母健在)。
    - 细化: 补充细节, 不推翻核心 (A会用剑 -> A擅长流风剑法)。细化应融合, 不是矛盾。
3.  输出要求: 融合非冲突信息, 报告被忽略的冲突旧信息, 禁止罗列, 聚焦问题。

# 输出结构
- 统一设定: [以要点形式, 清晰列出整合后的最终设定]
- 设计演变与冲突: (可选) [简要说明关键设定的演变过程, 或指出已解决的重大设计矛盾]
"""
        synthesis_query_text = self.build_agent_query(all_questions, final_instruction, rules_text)
        retrieval_query_text = "\n".join(all_questions)

        synthesis_system_prompt, synthesis_user_prompt = load_prompts(task.category, "rag_cn", "synthesis_system_prompt", "synthesis_user_prompt")

        result = hybrid_query(
            vector_store=vector_store,
            graph_store=graph_store,
            retrieval_query_text=retrieval_query_text,
            synthesis_query_text=synthesis_query_text,
            synthesis_system_prompt=synthesis_system_prompt,
            synthesis_user_prompt=synthesis_user_prompt,
            kg_nl2graphquery_prompt=kg_query_gen_prompt,
            vector_filters=vector_filters,
            vector_similarity_top_k=150,
            vector_rerank_top_n=50,
            kg_similarity_top_k=300,
            kg_rerank_top_n=100,
            vector_sort_by='time',
            kg_sort_by='time',
        )

        self.caches['upper_design'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层设计(upper_design)上下文完成。")
        return result
 

    def get_text_summary(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取历史情节概要(text_summary)上下文...")
        cache_key = f"get_text_summary:{task.run_id}:{task.id}"
        cached_result = self.caches['text_summary'].get(cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}] 命中历史情节概要(text_summary)缓存。")
            return cached_result

        inquiry = self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'write')
        if not inquiry or not inquiry.get("questions"):
            logger.warning(f"[{task.id}] 生成的探询问题为空或无效, 跳过历史情节概要检索。")
            return ""

        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 探询列表中没有问题, 无法执行历史情节概要查询。")
            return ""

        summary_vector_store = self._get_vector_store(task.run_id, "summary")
        write_graph_store = self._get_graph_store(task.run_id, "write")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        vector_filters = MetadataFilters(filters=active_filters) if active_filters else None
        logger.info(f"[{task.id}] 历史情节概要检索使用的过滤器: {vector_filters.to_dict() if vector_filters else 'None'}")

        kg_gen_query_prompt = load_prompts(task.category, "rag_cn", "kg_gen_query_prompt")[0]
        kg_query_gen_prompt = PromptTemplate(template=kg_gen_query_prompt)

        final_instruction = "角色: 剧情连续性编辑。\n任务: 整合情节摘要(向量)与正文细节(图谱), 生成一份服务于续写的上下文报告。\n重点: 角色关系、关键伏笔、情节呼应。"
        rules_text = """
# 整合规则
1.  冲突解决: 向量(摘要)与图谱(细节)冲突时, 以图谱为准。
2.  排序依据:
    - 向量: 章节顺序 (时间线)。
    - 图谱: 章节顺序 (时间线)。
3.  输出要求: 整合为连贯叙述, 禁止罗列。

# 输出结构
- 核心上下文: [一段连贯的叙述, 总结最重要的背景信息]
- 关键要点: (可选) [以列表形式补充说明关键的角色状态、伏笔或设定]
"""
        
        synthesis_query_text = self.build_agent_query(all_questions, final_instruction, rules_text)
        retrieval_query_text = "\n".join(all_questions)
        synthesis_system_prompt, synthesis_user_prompt = load_prompts(task.category, "rag_cn", "synthesis_system_prompt", "synthesis_user_prompt")
        
        result = hybrid_query(
            vector_store=summary_vector_store,
            graph_store=write_graph_store,
            retrieval_query_text=retrieval_query_text,
            synthesis_query_text=synthesis_query_text,
            synthesis_system_prompt=synthesis_system_prompt,
            synthesis_user_prompt=synthesis_user_prompt,
            kg_nl2graphquery_prompt=kg_query_gen_prompt,
            vector_filters=vector_filters,
            vector_similarity_top_k=300,
            vector_rerank_top_n=100,
            kg_similarity_top_k=600,
            kg_rerank_top_n=200,
            vector_sort_by='narrative',
            kg_sort_by='narrative',
        )

        self.caches['text_summary'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取历史情节概要(text_summary)上下文完成。")
        return result


    def build_agent_query(self, questions: List[str], final_instruction: str, rules_text: str) -> str:
        main_inquiry = "请综合分析并回答以下问题。"
        query_text = f"# 核心探询目标\n{main_inquiry}\n\n# 具体信息需求 (按优先级降序排列)\n"
        for question in questions:
            query_text += f"- {question}\n"
        instruction_block = f"\n# 任务指令与规则\n"
        instruction_block += f"## 最终目标\n{final_instruction}\n"
        instruction_block += f"\n## 执行规则\n"
        adapted_rules_text = re.sub(r'^\s*#\s+', '### ', rules_text.lstrip(), count=1)
        instruction_block += adapted_rules_text
        query_text += instruction_block
        logger.debug(f"最终构建的 Agent 查询文本:\n{query_text}")
        return query_text


###############################################################################

_rag_instance = None
def get_story_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = story_rag()
    return _rag_instance






