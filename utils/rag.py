import os
import json
import hashlib
import asyncio
import re
from datetime import datetime
from loguru import logger
from diskcache import Cache
from typing import Dict, Any, List, Literal, Optional, Callable
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.graph_stores.memgraph import MemgraphGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (
    StorageContext, 
    Document, 
    VectorStoreIndex, 
    KnowledgeGraphIndex
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine, ResponseMode
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.tools import FunctionTool
from llama_index.core.base.response.schema import Response
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from utils.db import get_db, get_text_file_path, text_file_append, text_file_read
from utils.models import Task, get_preceding_sibling_ids
from utils.prompt_loader import load_prompts
from utils.llm import (
    LLM_TEMPERATURES,
    get_embedding_params,
    get_llm_messages,
    get_llm_params,
    llm_acompletion
)


class RAG:
    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.embed_params = get_embedding_params()
        self.embed_model_name = self.embed_params.pop('model')
        self.embed_model = LiteLLMEmbedding(
            model_name=self.embed_model_name,
            **self.embed_params,
        )
        self.embed_model.get_text_embedding("test")

        self.qdrant_client = QdrantClient(
            host=os.getenv("qdrant_host", "localhost"),
            port=int(os.getenv("qdrant_port", "6333")),
        )

        self.qdrant_aclient = AsyncQdrantClient(
            host=os.getenv("qdrant_host", "localhost"),
            port=int(os.getenv("qdrant_port", "6333")),
        )

        self.graph_store = MemgraphGraphStore(
            url=os.getenv("memgraph_url", "bolt://localhost:7687"),
            username=os.getenv("memgraph_username", "memgraph"),
            password=os.getenv("memgraph_password", "memgraph"),
        )

        self.llm_extract: LiteLLM = LiteLLM(**get_llm_params(llm='fast', temperature=LLM_TEMPERATURES["summarization"]))
        self.llm_reasoning: LiteLLM = LiteLLM(**get_llm_params(llm='reasoning', temperature=LLM_TEMPERATURES["reasoning"]))
        self.llm_synthesis: LiteLLM = LiteLLM(**get_llm_params(llm='reasoning', temperature=LLM_TEMPERATURES["synthesis"]))

        cache_base_dir = os.path.join(project_root, ".cache", "rag")
        os.makedirs(cache_base_dir, exist_ok=True)
        self.caches: Dict[str, Cache] = {
            'dependent_design': Cache(os.path.join(cache_base_dir, "dependent_design"), size_limit=int(32 * (1024**2))),
            'dependent_search': Cache(os.path.join(cache_base_dir, "dependent_search"), size_limit=int(32 * (1024**2))),
            'text_latest': Cache(os.path.join(cache_base_dir, "text_latest"), size_limit=int(32 * (1024**2))),
            'text_length': Cache(os.path.join(cache_base_dir, "text_length"), size_limit=int(1 * (1024**2))),
            'upper_design': Cache(os.path.join(cache_base_dir, "upper_design"), size_limit=int(128 * (1024**2))),
            'upper_search': Cache(os.path.join(cache_base_dir, "upper_search"), size_limit=int(128 * (1024**2))),
            'text_summary': Cache(os.path.join(cache_base_dir, "text_summary"), size_limit=int(128 * (1024**2))),
            'task_list': Cache(os.path.join(cache_base_dir, "task_list"), size_limit=int(32 * (1024**2))),
        }

    def _get_storage_context(self, run_id: str) -> StorageContext:
        sanitized_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', self.embed_model_name)
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            aclient=self.qdrant_aclient,
            collection_name=f"write_vectors_{run_id}_{sanitized_model_name}"
        )
        return StorageContext.from_defaults(
            vector_store=vector_store,
            graph_store=self.graph_store
        )

    async def add(self, task: Task, task_type: str):
        db = get_db(run_id=task.run_id, category=task.category)
        if task_type == "task_atom":
            if task.id == "1" or task.results.get("goal_update"):
                self.caches['task_list'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_task, task)
            await asyncio.to_thread(db.add_result, task)
        elif task_type == "task_plan_before_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_design(task, task.results.get("design_reflection"))
        elif task_type == "task_plan":
            if task.results.get("plan"):
                await asyncio.to_thread(db.add_result, task)
        elif task_type == "task_plan_reflection":
            if task.results.get("plan_reflection"):
                self.caches['task_list'].evict(tag=task.run_id)
                self.caches['upper_design'].evict(tag=task.run_id)
                self.caches['upper_search'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await asyncio.to_thread(db.add_sub_tasks, task)
        elif task_type == "task_execute_design":
            if task.results.get("design"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
        elif task_type == "task_execute_design_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_design(task, task.results.get("design_reflection"))
        elif task_type == "task_execute_search":
            if task.results.get("search"):
                self.caches['dependent_search'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_search(task, task.results.get("search"))
        elif task_type == "task_execute_write_before_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_design(task, task.results.get("design_reflection"))
        elif task_type == "task_execute_write":
            if task.results.get("write"):
                await asyncio.to_thread(db.add_result, task)
        elif task_type == "task_execute_write_reflection":
            write_reflection = task.results.get("write_reflection")
            if write_reflection:
                self.caches['text_latest'].evict(tag=task.run_id)
                self.caches['text_length'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                header_parts = [
                    task.id,
                    task.hierarchical_position,
                    task.goal,
                    task.length,
                ]
                header = " ".join(filter(None, header_parts))
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                content = f"## 任务\n{header}\n{timestamp}\n\n{write_reflection}"
                await asyncio.to_thread(text_file_append, get_text_file_path(task), content)
                await self.store_write(task, write_reflection)
        elif task_type == "task_execute_summary":
            if task.results.get("summary"):
                self.caches['text_summary'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_summary(task, task.results.get("summary"))
        elif task_type == "task_aggregate_design":
            if task.results.get("design"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_design(task, task.results.get("design"))
        elif task_type == "task_aggregate_search":
            if task.results.get("search"):
                self.caches['dependent_search'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_search(task, task.results.get("search"))
        elif task_type == "task_aggregate_summary":
            if task.results.get("summary"):
                self.caches['text_summary'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store_summary(task, task.results.get("summary"))
        else:
            raise ValueError("不支持的任务类型")

    async def store_design(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 design 内容 (向量索引与知识图谱)...")
        header_parts = [
            task.id,
            task.hierarchical_position,
            task.goal,
        ]
        header = " ".join(filter(None, header_parts))
        content = f"# 任务\n{header}\n\n{content}"
        storage_context = self._get_storage_context(task.run_id)
        doc_metadata = {
            "run_id": task.run_id,                      # 运行ID, 用于数据隔离
            "task_id": task.id,                         # 任务ID, 用于追溯来源
            "content_type": "design",                   # 内容类型, 用于过滤
            "status": "active",                         # 状态, 用于标记/取消文档
            "created_at": datetime.now().isoformat()    # 创建时间, 用于时序排序
        }
        doc = Document(id_=task.id, text=content, metadata=doc_metadata)
        node_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
        await asyncio.to_thread(
            VectorStoreIndex.from_documents, 
            [doc], 
            storage_context=storage_context, 
            embed_model=self.embed_model,
            transformations=[node_parser]
        )
        logger.info(f"[{task.id}] design 内容向量化完成, 开始构建知识图谱...")
        await asyncio.to_thread(
            KnowledgeGraphIndex.from_documents,
            [doc],
            storage_context=storage_context,
            embed_model=self.embed_model, 
            llm=self.llm_extract, 
            kg_extraction_prompt=load_prompts(task.category, "graph_cn", "kg_extraction_prompt_design")[0], 
            include_embeddings=True,        # 将图谱节点与向量嵌入关联, 支持混合搜索
            max_triplets_per_chunk=15,      # 每个文本块最多提取15个三元组
            transformations=[node_parser]   # 使用与向量索引相同的解析器, 保证块的一致性
        )
        logger.info(f"[{task.id}] design 内容存储完成。")

    async def store_search(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 search 内容 (向量索引)...")
        header_parts = [
            task.id,
            task.hierarchical_position,
            task.goal
        ]
        header = " ".join(filter(None, header_parts))
        content = f"# 任务\n{header}\n\n{content}"
        doc_metadata = {
            "run_id": task.run_id,                      # 运行ID, 用于数据隔离
            "task_id": task.id,                         # 任务ID, 用于追溯来源
            "content_type": "search",                   # 内容类型, 用于过滤
            "status": "active",                         # 状态, 用于标记/取消文档
            "created_at": datetime.now().isoformat()    # 创建时间, 用于时序排序
        }
        doc = Document(id_=task.id, text=content, metadata=doc_metadata)
        await asyncio.to_thread(
            VectorStoreIndex.from_documents, 
            [doc], 
            storage_context=self._get_storage_context(task.run_id), 
            embed_model=self.embed_model,
            transformations=[MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)]
        )
        logger.info(f"[{task.id}] search 内容存储完成。")

    async def store_write(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 write 内容 (构建知识图谱)...")
        doc_metadata = {
            "run_id": task.run_id,                      # 运行ID, 用于数据隔离
            "task_id": task.id,                         # 任务ID, 用于追溯来源
            "content_type": "write",                    # 内容类型, 用于过滤
            "status": "active",                         # 状态, 用于标记/取消文档
            "created_at": datetime.now().isoformat()    # 创建时间, 用于时序排序
        }
        doc = Document(id_=task.id, text=content, metadata=doc_metadata)
        await asyncio.to_thread(
            KnowledgeGraphIndex.from_documents,
            [doc],
            storage_context=self._get_storage_context(task.run_id),
            embed_model=self.embed_model, 
            llm=self.llm_extract, 
            kg_extraction_prompt=load_prompts(task.category, "graph_cn", "kg_extraction_prompt_write")[0], 
            include_embeddings=True,        # 将图谱节点与向量嵌入关联, 支持混合搜索
            max_triplets_per_chunk=15,      # 每个文本块最多提取15个三元组
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=100)]   # 使用与向量索引相同的解析器, 保证块的一致性
        )
        logger.info(f"[{task.id}] write 内容存储完成。")
    
    async def store_summary(self, task: Task, content: str) -> None:
        logger.info(f"[{task.id}] 正在存储 summary 内容 (向量索引)...")
        header_parts = [
            task.id,
            task.hierarchical_position,
            task.goal,
            task.length
        ]
        header = " ".join(filter(None, header_parts))
        content = f"# 任务\n{header}\n\n{content}"
        doc_metadata = {
            "run_id": task.run_id,                      # 运行ID, 用于数据隔离
            "task_id": task.id,                         # 任务ID, 用于追溯来源
            "content_type": "summary",                  # 内容类型, 用于过滤
            "status": "active",                         # 状态, 用于标记/取消文档
            "created_at": datetime.now().isoformat()    # 创建时间, 用于时序排序
        }
        doc = Document(id_=task.id, text=content, metadata=doc_metadata)
        await asyncio.to_thread(
            VectorStoreIndex.from_documents, 
            [doc], 
            storage_context=self._get_storage_context(task.run_id), 
            embed_model=self.embed_model,
            transformations=[MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)]
        )
        logger.info(f"[{task.id}] summary 内容存储完成。")

###############################################################################

    async def get_context_base(self, task: Task) -> Dict[str, Any]:
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
        db = get_db(run_id=task.run_id, category=task.category)
        dependent_design = await self.get_dependent_design(db, task)
        dependent_search = await self.get_dependent_search(db, task)
        text_latest = await self.get_text_latest(task)
        task_list = ""
        if len(task.id.split(".")) >= 2:
            task_list = await self.get_task_list(db, task)
        ret.update({
            "dependent_design": dependent_design,
            "dependent_search": dependent_search,
            "text_latest": text_latest,
            "task_list": task_list,
        })
        return ret
    
    async def get_context(self, task: Task) -> Dict[str, Any]:
        ret = await self.get_context_base(task)
        if not task.parent_id:
            return ret
        dependent_design = ret.get("dependent_design", "")
        dependent_search = ret.get("dependent_search", "")
        text_latest = ret.get("text_latest", "")
        task_list = ret.get("task_list", "")
        rag_context_coros = {}
        current_level = len(task.id.split("."))
        if current_level >= 3:
            rag_context_coros["upper_design"] = self.get_upper_design(task, dependent_design, dependent_search, text_latest, task_list)
            rag_context_coros["upper_search"] = self.get_upper_search(task, dependent_design, dependent_search, text_latest, task_list)
        if len(text_latest) > 500:
            rag_context_coros["text_summary"] = self.get_text_summary(task, dependent_design, dependent_search, text_latest, task_list)
        if not rag_context_coros:
            return ret
        results = await asyncio.gather(*rag_context_coros.values())
        ret.update({key: result for key, result in zip(rag_context_coros.keys(), results)})
        return ret

    async def get_dependent_design(self, db: Any, task: Task) -> str:
        cache_key = f"dependent_design:{task.run_id}:{task.id}"
        cached_result = await asyncio.to_thread(self.caches['dependent_design'].get, cache_key)
        if cached_result is not None:
            return cached_result
        result = await asyncio.to_thread(db.get_dependent_design, task)
        await asyncio.to_thread(self.caches['dependent_design'].set, cache_key, result, tag=task.run_id)
        return result

    async def get_dependent_search(self, db: Any, task: Task) -> str:
        cache_key = f"dependent_search:{task.run_id}:{task.id}"
        cached_result = await asyncio.to_thread(self.caches['dependent_search'].get, cache_key)
        if cached_result is not None:
            return cached_result
        result = await asyncio.to_thread(db.get_dependent_search, task)
        await asyncio.to_thread(self.caches['dependent_search'].set, cache_key, result, tag=task.run_id)
        return result

    async def get_task_list(self, db: Any, task: Task) -> str:
        cache_key = f"task_list:{task.run_id}:{task.parent_id}"
        cached_result = await asyncio.to_thread(self.caches['task_list'].get, cache_key)
        if cached_result is not None:
            return cached_result
        result = await asyncio.to_thread(db.get_task_list, task)
        await asyncio.to_thread(self.caches['task_list'].set, cache_key, result, tag=task.run_id)
        return result

    async def get_text_latest(self, task: Task, length: int = 3000) -> str:
        key = f"get_text_latest:{task.run_id}:{length}"
        cached_result = await asyncio.to_thread(self.caches['text_latest'].get, key)
        if cached_result is not None:
            return cached_result
        db = get_db(run_id=task.run_id, category=task.category)
        full_content = await asyncio.to_thread(db.get_latest_write_reflection, length)
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
        await asyncio.to_thread(self.caches['text_latest'].set, key, result, tag=task.run_id)
        return result

    async def get_text_length(self, task: Task) -> int:
        file_path = get_text_file_path(task)
        key = f"get_text_length:{file_path}"
        cached_result = await asyncio.to_thread(self.caches['text_length'].get, key)
        if cached_result is not None:
            return cached_result
        full_content = await asyncio.to_thread(text_file_read, file_path)
        length = len(full_content)
        await asyncio.to_thread(self.caches['text_length'].set, key, length, tag=task.run_id)
        return length

    async def get_aggregate_design(self, task: Task) -> Dict[str, Any]:
        ret = await self.get_context_base(task)
        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_design"] = await asyncio.to_thread(db.get_subtask_design, task.id)
        return ret

    async def get_aggregate_search(self, task: Task) -> Dict[str, Any]:
        ret = await self.get_context_base(task)
        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_search"] = await asyncio.to_thread(db.get_subtask_search, task.id)
        return ret

    async def get_aggregate_summary(self, task: Task) -> Dict[str, Any]:
        ret = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'hierarchical_position', 'goal', 'length'}
            ),
        }
        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_summary"] = await asyncio.to_thread(db.get_subtask_summary, task.id)
        return ret

    async def get_inquiry_plan(
        self,
        task: Task,
        dependent_design: str,
        dependent_search: str,
        text_latest: str,
        task_list: str,
        plan_type: Literal['search', 'design', 'write']
    ) -> Dict[str, Any]:
        if plan_type == 'search':
            SYSTEM_PROMPT, USER_PROMPT, InquiryPlan = load_prompts(task.category, "query_cn", "SYSTEM_PROMPT_search", "USER_PROMPT_search", "InquiryPlan")
        elif plan_type == 'design':
            SYSTEM_PROMPT, USER_PROMPT, SYSTEM_PROMPT_design_for_write, InquiryPlan = load_prompts(task.category, "query_cn", "SYSTEM_PROMPT_design", "USER_PROMPT_design", "SYSTEM_PROMPT_design_for_write", "InquiryPlan")
            if task.task_type == 'write' and task.results.get("atom_result") == "atom":
                SYSTEM_PROMPT = SYSTEM_PROMPT_design_for_write
        elif plan_type == 'write':
            SYSTEM_PROMPT, USER_PROMPT, InquiryPlan = load_prompts(task.category, "query_cn", "SYSTEM_PROMPT_write", "USER_PROMPT_write", "InquiryPlan")
        else:
            raise ValueError(f"不支持的探询计划类型: {plan_type}")
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
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context_dict_user)
        llm_params = get_llm_params(messages=messages, temperature=LLM_TEMPERATURES["reasoning"])
        message = await llm_acompletion(llm_params, response_model=InquiryPlan)
        return message.validated_data.model_dump()
    
    async def get_upper_search(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取上层搜索(upper_search)上下文...")
        cache_key = f"get_upper_search:{task.run_id}:{task.id}"
        cached_result = await asyncio.to_thread(self.caches['upper_search'].get, cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}] 命中上层搜索(upper_search)缓存。")
            return cached_result

        inquiry_plan = await self.get_inquiry_plan(task, dependent_design, dependent_search, text_latest, task_list, 'search')
        if not inquiry_plan or not inquiry_plan.get("questions"):
            logger.warning(f"[{task.id}] 生成的探询计划为空或无效, 跳过上层搜索。")
            return ""

        all_questions = list(inquiry_plan.get("questions", {}).keys())
        if not all_questions:
            logger.warning(f"[{task.id}] 探询计划中没有问题, 无法执行上层搜索查询。")
            return ""
        single_question = "\n".join(all_questions)
        logger.info(f"[{task.id}] 整合后的上层搜索查询问题:\n{single_question}")

        logger.info(f"[{task.id}] 正在初始化向量索引用于上层搜索...")
        storage_context = self._get_storage_context(task.run_id)
        vector_index = await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            storage_context.vector_store,
            embed_model=self.embed_model
        )

        active_filters = [MetadataFilter(key='content_type', value='search')]
        preceding_sibling_ids = get_preceding_sibling_ids(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        logger.info(f"[{task.id}] 上层搜索使用的过滤器: {active_filters}")

        vector_retriever = vector_index.as_retriever(
            filters=MetadataFilters(filters=active_filters),
            similarity_top_k=20,
        )

        logger.info(f"[{task.id}] 开始从向量数据库中检索上层搜索内容...")
        retrieved_nodes = await asyncio.to_thread(vector_retriever.retrieve, single_question)
        if not retrieved_nodes:
            logger.warning(f"[{task.id}] 上层搜索未检索到任何相关节点。")
            return ""
        logger.info(f"[{task.id}] 上层搜索成功检索到 {len(retrieved_nodes)} 个相关节点。")

        source_details = [
            re.sub(r'\s+', ' ', node.get_content()).strip() for node in retrieved_nodes
        ]
        result = "\n\n".join(filter(None, source_details))
        await asyncio.to_thread(self.caches['upper_search'].set, cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层搜索(upper_search)上下文完成。\n{result}")
        return result

    async def get_upper_design(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取上层设计(upper_design)上下文...")
        cache_key = f"get_upper_design:{task.run_id}:{task.id}"
        cached_result = await asyncio.to_thread(self.caches['upper_design'].get, cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}] 命中上层设计(upper_design)缓存。")
            return cached_result

        inquiry_plan = await self.get_inquiry_plan(task, dependent_design, dependent_search, text_latest, task_list, 'design')
        if not inquiry_plan or not inquiry_plan.get("questions"):
            logger.warning(f"[{task.id}] 生成的探询计划为空或无效, 跳过上层设计检索。")
            return ""

        all_questions = list(inquiry_plan.get("questions", {}).keys())
        if not all_questions:
            logger.warning(f"[{task.id}] 探询计划中没有问题, 无法执行上层设计查询。")
            return ""

        logger.info(f"[{task.id}] 正在初始化向量和知识图谱查询引擎...")
        kg_gen_query_prompt = load_prompts(task.category, "rag_cn", "kg_gen_query_prompt")[0]
        storage_context = self._get_storage_context(task.run_id)
        vector_index = await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            storage_context.vector_store,
            embed_model=self.embed_model
        )
        response_synthesizer = CompactAndRefine(
            llm=self.llm_synthesis,
            prompt_helper=PromptHelper(
                context_window=self.llm_synthesis.context_window,
                num_output=self.llm_synthesis.max_tokens,
                chunk_overlap_ratio=0.2
            )
        )
        active_filters = [MetadataFilter(key='content_type', value='design')]
        preceding_sibling_ids = get_preceding_sibling_ids(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        logger.info(f"[{task.id}] 4. 构建向量查询引擎...")
        vector_query_engine = vector_index.as_query_engine(
            filters=MetadataFilters(filters=active_filters),
            llm=self.llm_reasoning,
            response_synthesizer=response_synthesizer,
            similarity_top_k=150,
            node_postprocessors=[
                LLMRerank(llm=self.llm_reasoning, top_n=50)
            ]
        )
        logger.info(f"[{task.id}] 5. 构建知识图谱查询引擎...")
        kg_filter_context = {
            "run_id": task.run_id,
            "content_type": "design",
        }
        kg_query_gen_prompt = PromptTemplate(kg_gen_query_prompt.partial_format(**kg_filter_context))
        logger.info(f"[{task.id}]   - 加载知识图谱索引...")
        kg_index = await asyncio.to_thread(
            KnowledgeGraphIndex.from_documents,
            [], 
            storage_context=storage_context,
            llm=self.llm_reasoning,
            include_embeddings=True,
            embed_model=self.embed_model
        )
        logger.info(f"[{task.id}]   - 创建混合模式知识图谱检索器...")
        kg_retriever = kg_index.as_retriever(
            retriever_mode="hybrid",          # 混合模式: 结合了关键词、向量和图查询
            similarity_top_k=300,             # 1. 向量检索: 初步召回200个相关节点
            with_nl2graphquery=True,          # 2. NL2Graph: 将自然语言转为Cypher查询
            graph_traversal_depth=2,          # 3. 图遍历: 从召回的节点出发, 深入探索2层关系
            nl2graphquery_prompt=kg_query_gen_prompt, # 使用我们自定义的、带过滤条件的prompt
        )
        logger.info(f"[{task.id}]   - 组装知识图谱查询引擎...")
        kg_query_engine = RetrieverQueryEngine(
            retriever=kg_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                LLMRerank(llm=self.llm_reasoning, top_n=100)
            ]
        )
        logger.info(f"[{task.id}] 查询引擎初始化完成。")
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
        novel_length = await self.get_text_length(task)
        retrieval_mode = inquiry_plan.get("retrieval_mode", "simple")
        query_text = self.build_agent_query(inquiry_plan, final_instruction, rules_text)
        logger.info(f"[{task.id}] 小说长度: {novel_length}, 探询模式: {retrieval_mode}。")
        if novel_length > 100000 and retrieval_mode == 'complex':
            logger.info(f"[{task.id}] 使用 ReAct Agent 模式进行复杂查询。")
            react_system_prompt = load_prompts(task.category, "rag_cn", "react_system_prompt")[0]

            logger.info(f"[{task.id}] 正在创建向量搜索工具 (time_aware_vector_search)...")
            vector_tool = self._create_sorted_query_tool(
                vector_query_engine,
                name="time_aware_vector_search",
                description="功能: 检索上层的宏观小说设定、故事大纲、核心概念。",
                sort_by='time',
                formatter=self.format_response_with_sorting
            )

            logger.info(f"[{task.id}] 正在创建知识图谱搜索工具 (time_aware_knowledge_graph_search)...")
            kg_tool = self._create_sorted_query_tool(
                kg_query_engine,
                name="time_aware_knowledge_graph_search",
                description=f"功能: 探索实体、关系、路径。查询必须满足过滤条件: run_id='{task.run_id}', content_type='design'。",
                sort_by='time',
                formatter=self.format_response_with_sorting
            )

            tools = [vector_tool, kg_tool]
            logger.info(f"[{task.id}] 正在初始化 ReAct Agent 的记忆模块 (token_limit=4096)...")
            memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

            logger.info(f"[{task.id}] 正在组装 ReAct Agent...")
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=self.llm_reasoning,
                memory=memory,
                system_prompt=react_system_prompt,
            )

            logger.info(f"[{task.id}] ReAct Agent 开始执行思考和工具调用循环...")
            response = await agent.achat(query_text)
            result = str(response)
        else:
            logger.info(f"[{task.id}] 使用简单模式进行向量/知识图谱联合查询。")
            single_question = "\n".join(all_questions)
            synthesis_system_prompt, synthesis_user_prompt = load_prompts(task.category, "rag_cn", "synthesis_system_prompt", "synthesis_user_prompt")
            
            logger.info(f"[{task.id}] 正在执行向量查询...")
            vector_response = await asyncio.to_thread(vector_query_engine.query, single_question)
            formatted_vector_str = await asyncio.to_thread(self.format_response_with_sorting, vector_response, 'time')
            logger.info(f"[{task.id}] 向量查询完成, 检索到 {len(vector_response.source_nodes)} 个节点。")

            logger.info(f"[{task.id}] 正在执行知识图谱查询...")
            kg_response = await asyncio.to_thread(kg_query_engine.query, single_question)
            formatted_kg_str = await asyncio.to_thread(self.format_response_with_sorting, kg_response, 'time')
            logger.info(f"[{task.id}] 知识图谱查询完成, 检索到 {len(kg_response.source_nodes)} 个节点。")

            logger.info(f"[{task.id}] 正在整合向量和知识图谱的查询结果...")
            context_dict_user = {
                "query_text": query_text,
                "formatted_vector_str": formatted_vector_str,
                "formatted_kg_str": formatted_kg_str,
            }
            messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)
            llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=LLM_TEMPERATURES["synthesis"])
            final_message = await llm_acompletion(llm_params)
            result = final_message.content
            logger.info(f"[{task.id}] 结果整合完成。")
        await asyncio.to_thread(self.caches['upper_design'].set, cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取上层设计(upper_design)上下文完成。")
        return result
 
    async def get_text_summary(self, task: Task, dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}] 开始获取历史情节概要(text_summary)上下文...")
        cache_key = f"get_text_summary:{task.run_id}:{task.id}"
        cached_result = await asyncio.to_thread(self.caches['text_summary'].get, cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}] 命中历史情节概要(text_summary)缓存。")
            return cached_result

        inquiry_plan = await self.get_inquiry_plan(task, dependent_design, dependent_search, text_latest, task_list, 'write')
        if not inquiry_plan or not inquiry_plan.get("questions"):
            logger.warning(f"[{task.id}] 生成的探询计划为空或无效, 跳过历史情节概要检索。")
            return ""

        all_questions = list(inquiry_plan.get("questions", {}).keys())
        if not all_questions:
            logger.warning(f"[{task.id}] 探询计划中没有问题, 无法执行历史情节概要查询。")
            return ""

        kg_gen_query_prompt = load_prompts(task.category, "rag_cn", "kg_gen_query_prompt")[0]
        storage_context = self._get_storage_context(task.run_id)
        logger.info(f"[{task.id}] 2. 初始化响应合成器 (CompactAndRefine)...")
        response_synthesizer = CompactAndRefine(
            llm=self.llm_synthesis,
            prompt_helper=PromptHelper(
                context_window=self.llm_synthesis.context_window,
                num_output=self.llm_synthesis.max_tokens,
                chunk_overlap_ratio=0.2
            )
        )
        logger.info(f"[{task.id}] 3. 从存储中加载向量索引...")
        vector_index = await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            storage_context.vector_store,
            embed_model=self.embed_model
        )
        logger.info(f"[{task.id}] 4. 构建向量查询引擎...")
        active_filters = [MetadataFilter(key='content_type', value='summary')]
        preceding_sibling_ids = get_preceding_sibling_ids(task.id)
        if preceding_sibling_ids:
            active_filters.append(MetadataFilter(key='task_id', value=preceding_sibling_ids, operator='nin'))
        logger.info(f"[{task.id}] 历史情节概要检索使用的过滤器: {active_filters}")
        vector_query_engine = vector_index.as_query_engine(
            filters=MetadataFilters(filters=active_filters),
            llm=self.llm_reasoning,
            response_synthesizer=response_synthesizer,
            similarity_top_k=300,
            node_postprocessors=[
                LLMRerank(llm=self.llm_reasoning, top_n=100)
            ]
        )
        logger.info(f"[{task.id}] 5. 构建知识图谱查询引擎...")
        kg_filter_context = {
            "run_id": task.run_id,
            "content_type": "write",
        }
        kg_query_gen_prompt = PromptTemplate(kg_gen_query_prompt.partial_format(**kg_filter_context))
        logger.info(f"[{task.id}]   - 加载知识图谱索引...")
        kg_index = await asyncio.to_thread(
            KnowledgeGraphIndex.from_documents,
            [], 
            storage_context=storage_context,
            llm=self.llm_reasoning,
            include_embeddings=True,
            embed_model=self.embed_model
        )
        logger.info(f"[{task.id}]   - 创建混合模式知识图谱检索器...")
        kg_retriever = kg_index.as_retriever(
            retriever_mode="hybrid",
            similarity_top_k=600,
            with_nl2graphquery=True,
            graph_traversal_depth=2,
            nl2graphquery_prompt=kg_query_gen_prompt,
        )
        kg_query_engine = RetrieverQueryEngine(
            retriever=kg_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                LLMRerank(llm=self.llm_reasoning, top_n=200)
            ]
        )
        logger.info(f"[{task.id}] 查询引擎初始化完成。")
        final_instruction = "角色: 剧情连续性编辑。\n任务: 整合情节摘要(向量)与正文细节(图谱), 生成一份服务于续写的上下文报告。\n重点: 角色关系、关键伏笔、情节呼应。"
        rules_text = """
# 整合规则
1.  冲突解决: 向量(摘要)与图谱(细节)冲突时, 以图谱为准。
2.  排序依据:
    - 向量: 相关性。
    - 图谱: 章节顺序 (时间线)。
3.  输出要求: 整合为连贯叙述, 禁止罗列。

# 输出结构
- 核心上下文: [一段连贯的叙述, 总结最重要的背景信息]
- 关键要点: (可选) [以列表形式补充说明关键的角色状态、伏笔或设定]
"""
        novel_length = await self.get_text_length(task)
        retrieval_mode = inquiry_plan.get("retrieval_mode", "simple")
        query_text = self.build_agent_query(inquiry_plan, final_instruction, rules_text)
        logger.info(f"[{task.id}] 小说长度: {novel_length}, 探询模式: {retrieval_mode}。")
        if novel_length > 100000 and retrieval_mode == 'complex':
            logger.info(f"[{task.id}] 正在创建 ReAct Agent 及其工具...")
            react_system_prompt = load_prompts(task.category, "rag_cn", "react_system_prompt")[0]
            vector_tool = self._create_sorted_query_tool(
                vector_query_engine,
                name="time_aware_vector_search",
                description="功能: 检索小说情节摘要、角色关系、事件发展。范围: 所有层级的历史摘要。",
                sort_by='narrative',
                formatter=self.format_response_with_sorting
            )
            kg_tool = self._create_sorted_query_tool(
                kg_query_engine,
                name="time_aware_knowledge_graph_search",
                description=f"功能: 探索实体、关系、路径。查询必须满足过滤条件: run_id='{task.run_id}', content_type='write'。",
                sort_by='narrative',
                formatter=self.format_response_with_sorting
            )
            tools = [vector_tool, kg_tool]
            logger.info(f"[{task.id}] 正在组装 ReAct Agent...")
            memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=self.llm_reasoning,
                memory=memory,
                system_prompt=react_system_prompt,
            )
            logger.info(f"[{task.id}] ReAct Agent 开始执行思考和工具调用循环...")
            response = await agent.achat(query_text)
            result = str(response)
        else:
            logger.info(f"[{task.id}] 使用简单模式进行向量/知识图谱联合查询。")
            single_question = "\n".join(all_questions)
            synthesis_system_prompt, synthesis_user_prompt = load_prompts(task.category, "rag_cn", "synthesis_system_prompt", "synthesis_user_prompt")
            
            logger.info(f"[{task.id}] 正在执行向量查询(历史摘要)...")
            vector_response = await asyncio.to_thread(vector_query_engine.query, single_question)
            formatted_vector_str = await asyncio.to_thread(self.format_response_with_sorting, vector_response, 'narrative')
            logger.info(f"[{task.id}] 向量查询完成, 检索到 {len(vector_response.source_nodes)} 个节点。")

            logger.info(f"[{task.id}] 正在执行知识图谱查询(正文细节)...")
            kg_response = await asyncio.to_thread(kg_query_engine.query, single_question)
            formatted_kg_str = await asyncio.to_thread(self.format_response_with_sorting, kg_response, 'narrative')
            logger.info(f"[{task.id}] 知识图谱查询完成, 检索到 {len(kg_response.source_nodes)} 个节点。")

            context_dict_user = {
                "query_text": query_text,
                "formatted_vector_str": formatted_vector_str,
                "formatted_kg_str": formatted_kg_str,
            }
            messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)
            llm_params = get_llm_params(llm='reasoning', messages=messages, temperature=LLM_TEMPERATURES["synthesis"])
            final_message = await llm_acompletion(llm_params)
            result = final_message.content
        await asyncio.to_thread(self.caches['text_summary'].set, cache_key, result, tag=task.run_id)
        logger.info(f"[{task.id}] 获取历史情节概要(text_summary)上下文完成。")
        return result

    def build_agent_query(self, inquiry_plan: Dict[str, Any], final_instruction: str, rules_text: str) -> str:
        main_inquiry = inquiry_plan.get("main_inquiry", "请综合分析并回答以下问题。")
        query_text = f"# 核心探询目标\n{main_inquiry}\n\n# 具体信息需求 (按优先级降序排列)\n"
        has_priorities = False
        questions_dict = inquiry_plan.get("questions", {})
        # 按 high, medium, low 排序问题
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_questions = sorted(questions_dict.items(), key=lambda item: priority_order.get(item[1], 99))
        for question, priority in sorted_questions:
            if priority in ['high', 'low']:
                has_priorities = True # 检查是否存在高/低优先级, 以便后续添加规则
            query_text += f"- {question} (优先级: {priority})\n"
        instruction_block = f"\n# 任务指令与规则\n"
        instruction_block += f"## 最终目标\n{final_instruction}\n"
        instruction_block += f"\n## 执行规则\n"
        if has_priorities:
            instruction_block += "- 优先级: 你必须优先分析和回答标记为 `high` 优先级的信息需求。\n"
        adapted_rules_text = re.sub(r'^\s*#\s+', '### ', rules_text.lstrip(), count=1)
        instruction_block += adapted_rules_text
        query_text += instruction_block
        logger.debug(f"最终构建的 Agent 查询文本:\n{query_text}")
        return query_text

    def _create_sorted_query_tool(self, query_engine: Any, name: str, description: str, sort_by: Literal['time', 'narrative', 'relevance'], formatter: Callable) -> Any:
        """
        将一个查询引擎包装成一个 Agent 可用的、具备排序功能的工具 (FunctionTool)。
        """
        async def sorted_query(query_str: str) -> str:
            response: Response = await asyncio.to_thread(query_engine.query, query_str)
            return await asyncio.to_thread(formatter, response, sort_by)

        # 使用 LlamaIndex 的 FunctionTool.from_defaults 创建工具
        return FunctionTool.from_defaults(
            fn=sorted_query,
            name=name,
            description=description
        )

    def format_response_with_sorting(self, response: Response, sort_by: Literal['time', 'narrative', 'relevance']) -> str:
        """
        sort_by (Literal): 排序策略: 'time' (时间倒序), 'narrative' (章节顺序), 'relevance' (相关性)。
        """
        if not response.source_nodes:
            return f"未找到相关来源信息, 但综合回答是: \n{str(response)}"

        # 默认使用 LlamaIndex 返回的顺序 (通常是按相关性)
        sorted_nodes = response.source_nodes
        sort_description = ""

        if sort_by == 'narrative':
            # 定义一个“自然排序”的 key 函数, 能正确处理 '1.2' 和 '1.10' 这样的章节号
            def natural_sort_key(s: str) -> List[Any]:
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

            sorted_nodes = sorted(
                list(response.source_nodes),
                key=lambda n: natural_sort_key(n.metadata.get("task_id", "")),
                reverse=False  # 正序排列
            )
            sort_description = "按小说章节顺序排列 (从前到后)"
        elif sort_by == 'time':
            # 按元数据中的 'created_at' 时间戳对来源节点进行降序排序
            sorted_nodes = sorted(
                list(response.source_nodes),
                key=lambda n: n.metadata.get("created_at", "1970-01-01T00:00:00"),
                reverse=True  # 倒序排列, 最新的在前
            )
            sort_description = "按时间倒序排列 (最新的在前)"
        else: # 'relevance' 或其他默认情况
            sort_description = "按相关性排序"

        # 格式化每个来源节点的详细信息
        source_details = []
        for node in sorted_nodes:
            timestamp = node.metadata.get("created_at", "未知时间")
            task_id = node.metadata.get("task_id", "未知章节")
            score = node.get_score()
            score_str = f"{score:.4f}" if score is not None else "N/A"
            # 将文本中的多个空白符合并为一个空格, 以简化输出
            content = re.sub(r'\s+', ' ', node.get_content()).strip()
            source_details.append(f"来源信息 (章节: {task_id}, 时间: {timestamp}, 相关性: {score_str}):\n---\n{content}\n---")

        formatted_sources = "\n\n".join(source_details)

        # 将综合回答和详细来源信息组合成最终输出
        final_output = f"综合回答:\n{str(response)}\n\n详细来源 ({sort_description}):\n{formatted_sources}"
        return final_output


###############################################################################

_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance
