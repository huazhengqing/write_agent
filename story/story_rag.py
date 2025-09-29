import os
from datetime import datetime
from functools import lru_cache
from loguru import logger
from typing import Dict, Any, Literal
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from utils.sqlite_task import get_task_db
from utils.file import get_text_file_path, text_file_append
from utils.models import Task, get_sibling_ids_up_to_current
from utils.loader import load_prompts
from rag.kg import get_kg_query_engine, kg_add
from rag.hybrid_query import hybrid_query_batch, hybrid_query
from rag.vector_add import vector_add
from rag.vector_query import get_vector_query_engine, index_query_batch
from utils.llm import llm_temperatures, get_llm_messages, get_llm_params, llm_completion
from story.base import get_story_kg_store, get_story_vector_store


class StoryRAG:
    def __init__(self):
        from utils.file import cache_dir
        cache_story_dir = cache_dir / "story"
        os.makedirs(cache_story_dir, exist_ok=True)
        from diskcache import Cache
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

        self._save_handlers = {
            "task_refine": self._save_refine,
            "task_atom": self._save_atom,
            "task_plan": self._save_plan_or_hierarchy,
            "task_hierarchy": self._save_plan_or_hierarchy,
            "task_design": self._save_design_content,
            "task_aggregate_design": self._save_design_content,
            "task_search": self._save_search_content,
            "task_aggregate_search": self._save_search_content,
            "task_write": self._save_write_content,
            "task_write_review": self._save_design_content,
            "task_summary": self._save_summary_content,
            "task_aggregate_summary": self._save_summary_content,
            "task_translation": self._save_translation,
        }

    def _save_refine(self, task: Task, task_db: Any):
        if task.id == "1" or task.results.get("refine"):
            self.caches['task_list'].evict(tag=task.run_id)
            task_db.add_task(task)

    def _save_atom(self, task: Task, task_db: Any):
        task_db.add_result(task)

    def _save_plan_or_hierarchy(self, task: Task, task_db: Any):
        task_db.add_result(task)
        if task.sub_tasks:
            self.caches['task_list'].evict(tag=task.run_id)
            self.caches['upper_design'].evict(tag=task.run_id)
            self.caches['upper_search'].evict(tag=task.run_id)
            task_db.add_sub_tasks(task)

    def _save_design_content(self, task: Task, task_db: Any):
        content_key = "write_review" if task.task_type == "task_write_review" else "design"
        content = task.results.get(content_key)
        if content:
            self.caches['dependent_design'].evict(tag=task.run_id)
            task_db.add_result(task)
            self.save_design(task, content)

    def _save_search_content(self, task: Task, task_db: Any):
        if task.results.get("search"):
            self.caches['dependent_search'].evict(tag=task.run_id)
            task_db.add_result(task)
            self.save_search(task, task.results.get("search"))

    def _save_write_content(self, task: Task, task_db: Any):
        final_content = task.results.get("write")
        if final_content:
            self.caches['text_latest'].evict(tag=task.run_id)
            self.caches['text_length'].evict(tag=task.run_id)
            task_db.add_result(task)
            header_parts = [task.id, task.hierarchical_position, task.goal, task.length]
            header = " ".join(filter(None, header_parts))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = f"## 任务\n{header}\n{timestamp}\n\n{final_content}"
            text_file_append(get_text_file_path(task), content)
            self.save_write(task, final_content)

    def _save_summary_content(self, task: Task, task_db: Any):
        if task.results.get("summary"):
            self.caches['text_summary'].evict(tag=task.run_id)
            task_db.add_result(task)
            self.save_summary(task, task.results.get("summary"))

    def _save_translation(self, task: Task, task_db: Any):
        final_content = task.results.get("translation")
        if final_content:
            task_db.add_result(task)
            from utils.file import get_translation_file_path
            header_parts = [task.id, task.hierarchical_position, task.goal, task.length]
            header = " ".join(filter(None, header_parts))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = f"## 翻译任务\n{header}\n{timestamp}\n\n{final_content}"
            text_file_append(get_translation_file_path(task), content)

    def save_data(self, task: Task, task_type: str):
        task_db = get_task_db(run_id=task.run_id)
        handler = self._save_handlers.get(task_type)
        if handler:
            handler(task, task_db)
        else:
            raise ValueError(f"不支持的保存任务类型: {task_type}")

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
            kg_extraction_prompt=load_prompts(f"prompts.{task.category}.rag.kg", "kg_extraction_prompt_write")[0]
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
            import asyncio
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
        full_content = task_db.get_latest_write(length)
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
        from utils.file import text_file_read
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
            system_prompt, user_prompt, Inquiry = load_prompts(f"prompts.{task.category}.rag_query", "system_prompt_search", "user_prompt_search", "Inquiry")
        elif inquiry_type == 'design':
            system_prompt, user_prompt, system_prompt_design_for_write, Inquiry = load_prompts(f"prompts.{task.category}.rag_query", "system_prompt_design", "user_prompt_design", "system_prompt_design_for_write", "Inquiry")
            if task.task_type == 'write' and task.results.get("atom_result") == "atom":
                system_prompt = system_prompt_design_for_write
        elif inquiry_type == 'write':
            system_prompt, user_prompt, Inquiry = load_prompts(f"prompts.{task.category}.rag_query", "system_prompt_write", "user_prompt_write", "Inquiry")
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
        cache_key = f"get_upper_search:{task.run_id}:{task.id}"
        cached_result = self.caches['upper_search'].get(cache_key)
        if cached_result is not None:
            return cached_result
        
        inquiry = await self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'search')
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过上层搜索。")
            return ""
        
        logger.info(f"[{task.id}] 开始获取上层搜索(upper_search)上下文...")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        filters = MetadataFilters(filters=active_filters) if active_filters else None
        
        query_engine = get_vector_query_engine(
            vector_store=get_story_vector_store(task.run_id, "search"),
            filters=filters,
            similarity_top_k=150,
            top_n=50,
        )

        results = await index_query_batch(query_engine, all_questions)
        result = "\n\n---\n\n".join(results)
        if not results:
            logger.warning(f"[{task.id}] 上层搜索(upper_search)未能生成答案或找到相关文档。")

        self.caches['upper_search'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层搜索(upper_search)上下文的流程已结束。")
        return result


    async def get_upper_design(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        cache_key = f"get_upper_design:{task.run_id}:{task.id}"
        cached_result = self.caches['upper_design'].get(cache_key)
        if cached_result is not None:
            return cached_result
        
        inquiry = await self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'design')
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过上层设计检索。")
            return ""

        logger.info(f"[{task.id}] 开始获取上层设计(upper_design)上下文...")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        vector_filters = MetadataFilters(filters=active_filters) if active_filters else None

        from story.base import hybrid_query_design
        result = await hybrid_query_design(task.run_id, all_questions, vector_filters)

        self.caches['upper_design'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层设计(upper_design)上下文完成。")
        return result
 

    async def get_text_summary(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        cache_key = f"get_text_summary:{task.run_id}:{task.id}"
        cached_result = self.caches['text_summary'].get(cache_key)
        if cached_result is not None:
            return cached_result

        inquiry = await self.get_inquiry(task, dependent_design, dependent_search, text_latest, task_list, 'write')
        all_questions = inquiry.get("questions", [])
        if not all_questions:
            logger.warning(f"[{task.id}] 生成的探询问题为空, 跳过历史情节概要检索。")
            return ""

        logger.info(f"[{task.id}] 开始获取历史情节概要(text_summary)上下文...")

        active_filters = []
        preceding_sibling_ids = get_sibling_ids_up_to_current(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        vector_filters = MetadataFilters(filters=active_filters) if active_filters else None

        from story.base import hybrid_query_write
        result = await hybrid_query_write(task.run_id, all_questions, vector_filters)

        self.caches['text_summary'].set(cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取历史情节概要(text_summary)上下文完成。")
        return result


###############################################################################



@lru_cache(maxsize=None)
def get_story_rag():
    return StoryRAG()
