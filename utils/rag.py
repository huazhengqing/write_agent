#coding: utf8
import os
import re
import json
import torch
import hashlib
import asyncio
import collections
from loguru import logger
from diskcache import Cache
from datetime import datetime
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal, Optional
from llama_index.llms.litellm import LiteLLM
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.core.base.response.schema import Response
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.memgraph import MemgraphGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core import Document, StorageContext, VectorStoreIndex, KnowledgeGraphIndex
from utils.db import get_db
from utils.models import Task
from utils.llm import LLM_PARAMS_fast, LLM_PARAMS_reasoning, get_llm_messages, get_llm_params, llm_acompletion
from utils.prompt_loader import load_prompts


class RAG:
    def __init__(self):
        embed_model = os.getenv("embed_model")
        embed_BASE_URL = os.getenv("embed_BASE_URL")
        embed_API_KEY = os.getenv("embed_API_KEY")
        embed_dimensions = int(os.getenv("embed_dims", "1024"))
        if embed_model and embed_BASE_URL and embed_API_KEY:
            self.embed_model = LiteLLMEmbedding(
                model_name = embed_model,
                api_base = embed_BASE_URL,
                api_key = embed_API_KEY, 
                kwargs = {
                    "dimensions" : embed_dimensions
                }
            )
            self.embed_model.get_text_embedding("test")
        else:
            model_identifier = os.getenv("local_embed_model", "BAAI/bge-m3")
            model_folder_name = model_identifier.split('/')[-1]
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            local_model_path = os.path.join(project_root, "models", model_folder_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if os.path.isdir(local_model_path):
                self.embed_model = HuggingFaceEmbedding(model_name=local_model_path, device=device)

        self.qdrant_client = QdrantClient(
            host=os.getenv("qdrant_host", "localhost"),
            port=int(os.getenv("qdrant_port", "6333")),
        )

        self.graph_store = MemgraphGraphStore(
            url=os.getenv("memgraph_url", "bolt://localhost:7687"),
            username=os.getenv("memgraph_username", "memgraph"),
            password=os.getenv("memgraph_password", "memgraph"),
        )

        self.agent_llm = LiteLLM(**LLM_PARAMS_reasoning)
        self.extraction_llm = LiteLLM(**LLM_PARAMS_fast)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=f"write_vectors_{run_id}"
        )
        return StorageContext.from_defaults(
            vector_store=vector_store,
            graph_store=self.graph_store
        )

###############################################################################

    async def add(self, task: Task, task_type: str):
        """
        根据任务类型, 将任务的产出结果持久化到数据库、文件和 RAG 索引中, 并管理相关缓存。
        """
        db = get_db(run_id=task.run_id, category=task.category)

        # 处理原子任务的创建或更新
        if task_type == "task_atom":
            # 仅当是根任务或目标被更新时, 才写入数据库
            if task.id == "1" or task.results.get("goal_update"):
                await asyncio.to_thread(db.add_task, task)
                # 任务列表已变更, 清除缓存
                self.caches['task_list'].evict(tag=task.run_id)
        # 处理任务规划阶段, 批量添加子任务
        elif task_type == "task_plan":
            await asyncio.to_thread(db.add_sub_tasks, task)
        # 处理对任务规划的反思
        elif task_type == "task_plan_reflection":
            # 规划反思可能影响任务结构和上层上下文, 清除相关缓存
            self.caches['task_list'].evict(tag=task.run_id)
            self.caches['upper_design'].evict(tag=task.run_id)
            self.caches['upper_search'].evict(tag=task.run_id)
            await asyncio.to_thread(db.add_sub_tasks, task)
        # 处理设计任务的执行结果
        elif task_type == "task_execute_design":
            await asyncio.to_thread(db.add_result, task)
            await self.store(task, "design", task.results.get("design"))
        # 处理搜索任务的执行结果
        elif task_type == "task_execute_search":
            await asyncio.to_thread(db.add_result, task)
            await self.store(task, "search", task.results.get("search"))
        # 处理对设计结果的反思
        elif task_type == "task_execute_design_reflection":
            await asyncio.to_thread(db.add_result, task)
            # 将反思后的设计结果更新到 RAG 索引
            await self.store(task, "design", task.results.get("design_reflection"))
        # 处理写作任务的初步执行结果 (通常是草稿)
        elif task_type == "task_execute_write":
            await asyncio.to_thread(db.add_result, task)
        # 处理对写作结果的反思 (生成最终内容)
        elif task_type == "task_execute_write_reflection":
            # 正文内容已变更, 清除相关缓存
            self.caches['text_latest'].evict(tag=task.run_id)
            self.caches['text_length'].evict(tag=task.run_id)
            write_reflection = task.results.get("write_reflection")
            await asyncio.to_thread(db.add_result, task)
            # 将最终内容追加到主文本文件
            await asyncio.to_thread(self.text_file_append, self.get_text_file_path(task), write_reflection)
            # 将最终内容存入 RAG 索引
            await self.store(task, "write", write_reflection)
        # 处理为正文生成的摘要
        elif task_type == "task_execute_summary":
            # 摘要内容已变更, 清除缓存
            self.caches['text_summary'].evict(tag=task.run_id)
            await asyncio.to_thread(db.add_result, task)
            await self.store(task, "summary", task.results.get("summary"))
        # 处理对子任务设计的聚合
        elif task_type == "task_aggregate_design":
            await asyncio.to_thread(db.add_result, task)
            await self.store(task, "design", task.results.get("design"))
        # 处理对子任务搜索结果的聚合
        elif task_type == "task_aggregate_search":
            await asyncio.to_thread(db.add_result, task)
            await self.store(task, "search", task.results.get("search"))
        # 处理对子任务摘要的聚合
        elif task_type == "task_aggregate_summary":
            self.caches['text_summary'].evict(tag=task.run_id)
            await asyncio.to_thread(db.add_result, task)
            await self.store(task, "summary", task.results.get("summary"))
        else:
            raise ValueError("不支持的任务类型")

    def text_file_append(self, file_path: str, content: str):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n\n--- {timestamp} ---\n\n{content}")
            f.flush()
            os.fsync(f.fileno())

    def get_text_file_path(self, task: Task) -> str:
        return os.path.join("output", task.category, f"{task.run_id}.txt")

    async def store(self, task: Task, content_type: Literal['design', 'search', 'write', 'summary'], content: str):
        """
        将指定类型的内容存储到 RAG 索引中 (向量存储和知识图谱)。
        - 向量存储 (Vector Store): 用于语义相似性搜索。
          - 存储 'design', 'search', 'summary' 类型的内容。
        - 知识图谱 (Knowledge Graph): 用于提取和查询结构化事实 (实体和关系)。
          - 存储 'design', 'search', 'write' 类型的内容。
        """
        logger.info(f"开始存储: {task.run_id} {task.id} {task.task_type} {task.goal} {content_type}")

        if not content: 
            logger.warning("内容为空, 跳过存储。")
            return

        # 获取当前运行的存储上下文, 连接到正确的 Qdrant 集合和 Memgraph 实例
        storage_context = self._get_storage_context(task.run_id)
        
        # 构建文档元数据, 用于后续的过滤和排序
        doc_metadata = {
            "run_id": task.run_id,                      # 运行ID, 用于数据隔离
            "hierarchy_level": len(task.id.split(".")), # 任务层级, 用于范围查询
            "content_type": content_type,               # 内容类型, 用于过滤
            "task_id": task.id,                         # 任务ID, 用于追溯来源
            "status": "active",                         # 状态, 用于标记/取消文档
            "created_at": datetime.now().isoformat()    # 创建时间, 用于时序排序
        }
        # 将内容和元数据封装成 LlamaIndex 的 Document 对象
        doc = Document(id_=task.id, text=content, metadata=doc_metadata)

        # 1. 向量存储: 适用于需要进行语义搜索的内容 (设计、搜索结果、摘要)
        if content_type in ['design', 'search', 'summary']:
            logger.info(f"正在将 '{content_type}' 内容存入向量索引...")
            # 使用 to_thread 将同步的索引操作移到工作线程, 避免阻塞事件循环
            await asyncio.to_thread(
                VectorStoreIndex.from_documents, 
                [doc], 
                storage_context=storage_context, 
                embed_model=self.embed_model
            )

        # 2. 知识图谱存储: 适用于包含事实、实体和关系的内容 (设计、搜索结果、正文)
        if content_type in ['design', 'search', 'write']:
            logger.info(f"正在从 '{content_type}' 内容中提取并存入知识图谱...")
            # 加载针对不同内容类型的定制化图谱提取 Prompt
            design_prompt, write_prompt, search_prompt = load_prompts(task.category, "graph_cn", "design_prompt", "write_prompt", "search_prompt")
            kg_extraction_prompts = {
                "design": PromptTemplate(design_prompt),
                "write": PromptTemplate(write_prompt),
                "search": PromptTemplate(search_prompt),
            }
            kg_prompt = kg_extraction_prompts.get(content_type)
            
            if not kg_prompt:
                logger.warning(f"内容类型 '{content_type}' 没有找到对应的图谱提取Prompt, 将跳过图谱存储。")
                return

            # 使用 to_thread 将同步的图谱构建操作移到工作线程
            await asyncio.to_thread(
                KnowledgeGraphIndex.from_documents,
                [doc],
                storage_context=storage_context,
                max_triplets_per_chunk=15,      # 每个文本块最多提取15个三元组
                kg_extraction_prompt=kg_prompt, # 使用定制化Prompt指导LLM提取
                llm=self.extraction_llm,        # 使用为提取任务优化的轻量级LLM
                include_embeddings=True,        # 将图谱节点与向量嵌入关联, 支持混合搜索
            )
            
        logger.info(f"完成存储: {task.run_id} {task.id} {content_type}")

###############################################################################

    async def get_context_base(self, task: Task) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")
    
        ret = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'parent_id', 'task_type', 'goal', 'length', 'dependency'}
            ),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        }
        if not task.parent_id:
            logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
            return ret
    
        db = get_db(run_id=task.run_id, category=task.category)
        dependent_design_task = self.get_dependent_design(db, task)
        dependent_search_task = self.get_dependent_search(db, task)
        text_latest_task = self.get_text_latest(task)
        dependent_design, dependent_search, text_latest = await asyncio.gather(
            dependent_design_task, dependent_search_task, text_latest_task
        )
        ret["dependent_design"] = dependent_design
        ret["dependent_search"] = dependent_search
        ret["text_latest"] = text_latest
        
        context_tasks = {}
        current_level = len(task.id.split("."))
        if current_level >= 2:
            context_tasks["task_list"] = self.get_context_task_list(db, task)
    
        if current_level >= 3:
            context_tasks["upper_design"] = self.search_context(task, 'upper_design', dependent_design, dependent_search, text_latest)
            context_tasks["upper_search"] = self.search_context(task, 'upper_search', dependent_design, dependent_search, text_latest)
    
        if len(text_latest) > 500:
            context_tasks["text_summary"] = self.search_context(task, 'text_summary', dependent_design, dependent_search, text_latest)
    
        if context_tasks:
            results = await asyncio.gather(*context_tasks.values())
            for i, key in enumerate(context_tasks.keys()):
                result = results[i]
                ret[key] = result
    
        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret
    
    async def get_dependent_design(self, db: Any, task: Task) -> str:
        cache_key = f"dependent_design:{task.run_id}:{task.parent_id}"
        cached_result = self.caches['dependent_design'].get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await asyncio.to_thread(db.get_dependent_design, task)

        self.caches['dependent_design'].set(cache_key, result, tag=task.run_id)
        return result

    async def get_dependent_search(self, db: Any, task: Task) -> str:
        cache_key = f"dependent_search:{task.run_id}:{task.parent_id}"
        cached_result = self.caches['dependent_search'].get(cache_key)
        if cached_result is not None:
            return cached_result
        result = await asyncio.to_thread(db.get_dependent_search, task)
        self.caches['dependent_search'].set(cache_key, result, tag=task.run_id)
        return result

    async def get_context_task_list(self, db: Any, task: Task) -> str:
        cache_key = f"task_list:{task.run_id}:{task.parent_id}"
        cached_result = self.caches['task_list'].get(cache_key)
        if cached_result is not None:
            return cached_result
        result = await asyncio.to_thread(db.get_context_task_list, task)
        self.caches['task_list'].set(cache_key, result, tag=task.run_id)
        return result

    async def get_text_latest(self, task: Task, length: int = 3000) -> str:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal} {length}")

        key = f"get_text_latest:{task.run_id}:{length}"
        cached_result = self.caches['text_latest'].get(key)
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
                # 2. 如果后面没有换行符（说明我们在最后一段）, 则尝试在截取点之前找到最后一个换行符
                last_newline_before_start = full_content.rfind('\n', 0, start_pos)
                if last_newline_before_start != -1:
                    result = full_content[last_newline_before_start + 1:]
                else:
                    # 3. 如果全文都没有换行符, 或者只有一段很长的内容, 则直接硬截取
                    result = full_content[-length:]

        self.caches['text_latest'].set(key, result, tag=task.run_id)
        logger.info(f"完成 {result}")
        return result

    async def get_text_length(self, task: Task) -> int:
        file_path = self.get_text_file_path(task)

        key = f"get_text_length:{file_path}"
        cached_result = self.caches['text_length'].get(key)
        if cached_result is not None:
            return cached_result
        
        full_content = await asyncio.to_thread(self.text_file_read, file_path)
        length = len(full_content)
        
        self.caches['text_length'].set(key, length, tag=task.run_id)

        return length

    def text_file_read(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    async def get_context_aggregate_design(self, task: Task) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        ret = await self.get_context_base(task)

        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_design"] = await asyncio.to_thread(db.get_subtask_design, task.id)

        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret

    async def get_context_aggregate_search(self, task: Task) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        ret = await self.get_context_base(task)

        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_search"] = await asyncio.to_thread(db.get_subtask_search, task.id)

        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret

    async def get_context_aggregate_summary(self, task: Task) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        ret = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'goal', 'length'}
            ),
        }

        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_summary"] = await asyncio.to_thread(db.get_subtask_summary, task.id)

        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret

    async def get_query(self, task: Task, search_type: Literal['text_summary', 'upper_design', 'upper_search'], dependent_design: str, dependent_search: str, text_latest: str) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal} {search_type} \n dependent_design: \n{dependent_design} \n dependent_search: \n{dependent_search} \n text_latest: \n{text_latest}")
        (
            SYSTEM_PROMPT_design, USER_PROMPT_design,
            SYSTEM_PROMPT_design_for_write,
            SYSTEM_PROMPT_write, USER_PROMPT_write,
            SYSTEM_PROMPT_search, USER_PROMPT_search,
            InquiryPlan
        ) = load_prompts(
            task.category, "query_cn",
            "SYSTEM_PROMPT_design", "USER_PROMPT_design",
            "SYSTEM_PROMPT_design_for_write",
            "SYSTEM_PROMPT_write", "USER_PROMPT_write",
            "SYSTEM_PROMPT_search", "USER_PROMPT_search",
            "InquiryPlan"
        )
        PROMPTS = {
            "upper_design": (SYSTEM_PROMPT_design, USER_PROMPT_design),
            "upper_search": (SYSTEM_PROMPT_search, USER_PROMPT_search),
            "text_summary": (SYSTEM_PROMPT_write, USER_PROMPT_write),
        }
        if search_type not in PROMPTS:
            raise ValueError(f"不支持的查询生成类别: {search_type}")

        context_dict_user = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'task_type', 'goal', 'length'}
            ),
            "dependent_design": dependent_design,
            "dependent_search": dependent_search,
            "text_latest": text_latest
        }
        SYSTEM_PROMPT, USER_PROMPT = PROMPTS[search_type]
        if task.task_type == 'write' and search_type == 'upper_design':
            SYSTEM_PROMPT = SYSTEM_PROMPT_design_for_write
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context_dict_user)
        llm_params = get_llm_params(messages, temperature=0.1)
        llm_params['response_format'] = {
            "type": "json_object",
            "schema": InquiryPlan.model_json_schema()
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                message = await llm_acompletion(llm_params)
                content = message.content
                inquiry_plan_obj = InquiryPlan.model_validate_json(content)
                break  # 验证成功，跳出循环
            except Exception as e:
                logger.warning(f"探询计划生成或验证失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    try:
                        # 尝试删除错误的缓存条目
                        from litellm.caching.cache_key_generator import get_cache_key
                        cache_key = get_cache_key(**llm_params)
                        litellm.cache.delete(cache_key)
                        logger.info(f"已删除错误的探询计划缓存: {cache_key}。正在重试...")
                    except Exception as cache_e:
                        logger.error(f"删除缓存条目失败: {cache_e}")
                else:
                    logger.error("探询计划生成在多次重试后仍然失败。")
                    raise

        inquiry_plan = inquiry_plan_obj.model_dump()

        logger.info(f"完成 \n{json.dumps(inquiry_plan, indent=2, ensure_ascii=False)}")
        return inquiry_plan

    async def search_context(self, task: Task, search_type: Literal['text_summary', 'upper_design', 'upper_search'], dependent_design: str, dependent_search: str, text_latest: str) -> str:
        """
        根据指定的搜索类型, 为当前任务检索并合成上下文信息。

        该方法是 RAG 的核心调度器, 包含以下步骤:
        1.  生成探询计划 (Inquiry Plan): 使用 LLM 确定需要检索哪些信息。
        2.  缓存检查: 检查是否已有缓存的检索结果, 如果有则直接返回。
        3.  配置与引擎设置:
            - 获取特定于搜索类型的配置 (过滤器、工具描述等)。
            - 初始化向量查询引擎 (用于语义搜索)。
            - 初始化知识图谱查询引擎 (用于事实和关系搜索), 并注入一个定制的 Cypher 生成 Prompt。
        4.  选择检索策略:
            - 对于大型文档或复杂查询, 使用 ReAct Agent 进行多步推理和工具调用。
            - 否则, 使用简单的并行查询 + LLM 合成策略。
        5.  执行检索与合成: 调用相应的执行函数 (`_execute_react_agent` 或 `_execute_simple`)。
        6.  缓存结果: 将最终结果存入缓存并返回。
        """
        logger.info(f"开始 {task.run_id} {task.id} {search_type}")

        # 1. 使用 LLM 生成一个结构化的“探询计划”, 指导后续的检索方向和内容
        inquiry_plan = await self.get_query(task, search_type, dependent_design, dependent_search, text_latest)
        if not inquiry_plan or not inquiry_plan.get("information_needs"):
            logger.warning("生成的探询计划为空或无效, 跳过搜索。")
            return ""

        # 2. 基于探询计划和任务信息生成缓存键, 检查并返回缓存结果
        key_data = {
            "run_id": task.run_id,
            "inquiry_plan": inquiry_plan,
            "search_type": search_type,
            "task_level": len(task.id.split(".")),
        }
        cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode('utf-8')).hexdigest()
        cache = self.caches[search_type]
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"命中缓存, 直接返回结果: {cache_key}")
            return cached_result

        # 3. 获取搜索配置并初始化存储上下文
        config = self.get_search_config(task, inquiry_plan, search_type)
        storage_context = self._get_storage_context(task.run_id)

        # 4. 设置向量查询引擎, 并应用元数据过滤器
        vector_filters = MetadataFilters(
            filters=[MetadataFilter(key=f['key'], value=f['value']) for f in config['vector_filters_list']]
        )
        vector_index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store, 
            embed_model=self.embed_model
        )
        vector_query_engine = vector_index.as_query_engine(
            filters=vector_filters
        )

        # 5. 为知识图谱查询引擎定义一个定制化的 Cypher 查询生成 Prompt
        kg_query_gen_prompt_str = f"""
# 角色
你是一位精通 Cypher 的图数据库查询专家。

# 任务
根据用户提供的自然语言问题和图谱 Schema, 生成一条精确、高效、且符合所有规则的 Cypher 查询语句。

# 上下文
- 用户问题: '{{query_str}}'
- 图谱 Schema:
---
{{schema}}
---

# 核心规则 (必须严格遵守)
1.  强制过滤 (最重要!): 生成的查询 必须 包含一个 `WHERE` 子句, 以确保数据隔离。过滤条件为: `{config["kg_filters_str"]}`。
    - 正确示例: `MATCH (n) WHERE {config["kg_filters_str"]} AND n.property = 'value' RETURN n`
    - 错误示例 (缺少强制过滤): `MATCH (n) WHERE n.property = 'value' RETURN n`
2.  Schema 遵从: 只能使用图谱 Schema 中定义的节点标签和关系类型。严禁使用任何 Schema 之外的元素。
3.  单行输出: 最终的 Cypher 查询必须是单行文本, 不含换行符。
4.  效率优先: 在满足查询需求的前提下, 生成的查询应尽可能高效。

# 行动
现在, 请为上述用户问题生成 Cypher 查询语句。
"""
        # 6. 设置知识图谱查询引擎
        kg_query_gen_prompt = PromptTemplate(kg_query_gen_prompt_str)
        kg_index = KnowledgeGraphIndex.from_documents(
            [], 
            storage_context=storage_context, 
            llm=self.agent_llm, 
            include_embeddings=True
        )
        kg_query_engine = kg_index.as_query_engine(
            include_text=False,
            graph_query_synthesis_prompt=kg_query_gen_prompt
        )

        # 7. 判断是使用简单检索, 还是更复杂的 ReAct Agent 检索
        novel_length = await self.get_text_length(task)
        retrieval_mode = inquiry_plan.get("retrieval_mode", "simple")
        # 当小说长度超过阈值、探询计划要求复杂模式、且非上层资料搜索时, 启用 Agent
        use_agent = novel_length > 100000 and retrieval_mode == 'complex' and search_type != 'upper_search'
        if use_agent: 
            # 复杂模式: 使用 ReAct Agent 进行多步推理和工具调用
            logger.info(f"复杂检索 长度={novel_length} 检索模式={retrieval_mode} 检索类型={search_type}")
            result = await self._execute_react_agent(
                vector_query_engine=vector_query_engine,
                kg_query_engine=kg_query_engine,
                config=config
            )
        else:
            # 简单模式: 并行查询, 然后由 LLM 一次性合成
            result = await self._execute_simple(
                inquiry_plan=inquiry_plan,
                vector_query_engine=vector_query_engine,
                kg_query_engine=kg_query_engine,
                config=config
            )

        cache.set(cache_key, result, tag=task.run_id)
        logger.info(f"结果已存入缓存: {cache_key}")
        logger.info(f"完成 \n{result}")
        return result

    async def _execute_react_agent(self, vector_query_engine: Any, kg_query_engine: Any, config: Dict[str, Any]) -> str:
        """
        使用 ReAct Agent 执行复杂的、多步骤的检索任务。

        ReAct (Reasoning and Acting) Agent 能够通过思考和行动的循环来解决复杂问题。
        它会根据初始问题, 动态地决定使用哪个工具 (向量搜索或知识图谱搜索), 
        并根据工具返回的结果进行思考, 规划下一步行动, 直到找到最终答案。
        Args:
            vector_query_engine: 向量查询引擎实例。
            kg_query_engine: 知识图谱查询引擎实例。
        """
        logger.info(f"开始 复杂模式 ReActAgent \n{json.dumps(config, indent=2, ensure_ascii=False)}")

        # 1. 创建向量搜索工具
        #    - `_create_time_aware_tool` 将查询引擎包装成一个 Agent 可调用的工具。
        #    - `description` 告诉 Agent 这个工具能做什么, Agent 会根据这个描述来决定何时使用它。
        #    - `sort_by` 控制工具返回结果的排序方式。
        vector_tool = self._create_time_aware_tool(
            vector_query_engine,
            name="time_aware_vector_search",
            description=config['vector_tool_desc'],
            sort_by=config['vector_sort_by']
        )

        # 2. 创建知识图谱搜索工具
        #    - 同样, 为知识图谱查询引擎创建一个工具, 并提供清晰的描述。
        kg_tool = self._create_time_aware_tool(
            kg_query_engine,
            name="time_aware_knowledge_graph_search",
            description=config["kg_tool_desc"],
            sort_by=config['kg_sort_by']
        )

        # 3. 初始化 ReAct Agent
        #    - 将创建的工具列表提供给 Agent。
        #    - `llm` 指定用于 Agent 思考和推理的大语言模型。
        #    - `verbose` 设为 True 时, 会打印出 Agent 的完整思考过程 (Thought, Action, Observation), 便于调试。
        agent = ReActAgent(
            tools=[vector_tool, kg_tool],
            llm=self.agent_llm,
            verbose=os.getenv("deployment_environment") == "test"
        )

        # 4. 异步执行 Agent
        #    - `achat` 方法启动 Agent 的 "思考-行动" 循环。
        #    - Agent 会分析 `config["query_text"]` (核心探询目标), 并开始调用工具。
        response = await agent.achat(config["query_text"])
        ret = str(response)

        logger.info(f"完成\n{ret}")
        return ret

    async def _execute_simple(self, inquiry_plan: Dict[str, Any], vector_query_engine: Any, kg_query_engine: Any, config: Dict[str, Any]) -> str:
        """
        执行简单的并行检索与合成策略。

        此方法适用于非 Agent 模式, 流程如下:
        1.  将探询计划中的所有问题合并为一个查询。
        2.  并行地向向量查询引擎和知识图谱查询引擎发送该查询。
        3.  将两个引擎返回的结果格式化并排序。
        4.  构建一个包含原始目标、检索结果和整合规则的 Prompt。
        5.  调用 LLM 对检索到的信息进行最终的分析、整合和提炼, 生成最终答案。
        6.  包含重试和缓存清理机制, 确保结果的健壮性。
        Args:
            inquiry_plan (Dict[str, Any]): 包含信息需求的探询计划。
            vector_query_engine: 向量查询引擎实例。
            kg_query_engine: 知识图谱查询引擎实例。
            config (Dict[str, Any]): 包含查询文本、排序方式等信息的配置字典。
        """
        logger.info(f"开始 简单模式 \n{json.dumps(config, indent=2, ensure_ascii=False)}\n{json.dumps(inquiry_plan, indent=2, ensure_ascii=False)}")
        
        # 1. 从探询计划中提取所有问题, 并合并成一个单一的查询字符串
        all_questions = [q for need in inquiry_plan.get("information_needs", []) for q in need.get("questions", [])]
        if not all_questions:
            logger.warning("探询计划中没有问题, 无法执行查询。")
            return ""
        single_question = "\n".join(all_questions)

        # 2. 使用 asyncio.gather 并行执行向量查询和知识图谱查询
        vector_response_task = vector_query_engine.aquery(single_question)
        kg_response_task = kg_query_engine.aquery(single_question)
        vector_response, kg_response = await asyncio.gather(vector_response_task, kg_response_task)

        # 3. 格式化并根据配置对检索结果进行排序
        formatted_vector_str = self._format_response_with_sorting(vector_response, config['vector_sort_by'])
        formatted_kg_str = self._format_response_with_sorting(kg_response, config['kg_sort_by'])

        # 4. 构建用于最终信息合成的 Prompt
        synthesis_user_prompt = f"""
# 任务
- 遵循“探询计划与规则”。
- 整合“向量检索”和“知识图谱检索”的信息。
- 生成一个连贯、统一、直接回应“核心探询目标”的最终回答。

# 探询计划与规则
{config["query_text"]}

# 信息源
## 向量检索 (语义与上下文)
{formatted_vector_str}
---
## 知识图谱检索 (事实与关系)
{formatted_kg_str}
---

# 输出原则
- 分析: 严格遵循“探询计划与规则”中的“最终目标”和“执行规则”。
- 整合: 综合两个信息源, 注意各自侧重（语义 vs 事实）。
- 应用: 严格应用“执行规则”处理信息（如解决冲突、融合信息）。
- 内容:
    - 必须完全基于提供的信息源。
    - 必须是整合提炼后的答案, 禁止罗列。
    - 禁止任何关于你自身或任务过程的描述 (例如, “根据您的要求...”)。
"""

        # 5. 准备调用 LLM 进行信息合成
        synthesis_messages = [
            {"role": "system", "content": "角色：信息整合分析师。任务：遵循用户指令，整合并提炼向量检索和知识图谱的信息。输出：一个逻辑连贯、事实准确、完全基于所提供材料的最终回答。"},
            {"role": "user", "content": synthesis_user_prompt}
        ]
        llm_params = get_llm_params(synthesis_messages, temperature=self.agent_llm.temperature)
        
        # 6. 调用 LLM, 并加入重试和缓存清理逻辑
        max_retries = 3
        for attempt in range(max_retries):
            try:
                final_message = await llm_acompletion(llm_params)
                content = final_message.content
                # 验证返回的内容是否有效
                if not content or len(content.strip()) < 20:
                    raise ValueError("合成的回答为空或过短。")
                break # 验证成功, 跳出重试循环
            except Exception as e:
                logger.warning(f"简单模式合成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    try:
                        # 如果 LLM 调用失败, 尝试删除可能存在的错误缓存, 以便重试时能重新生成
                        from litellm.caching.cache_key_generator import get_cache_key
                        cache_key = get_cache_key(**llm_params)
                        litellm.cache.delete(cache_key)
                        logger.info(f"已删除错误的合成缓存: {cache_key}。正在重试...")
                    except Exception as cache_e:
                        logger.error(f"删除缓存失败: {cache_e}")
                else:
                    # 所有重试均失败后, 记录错误并抛出异常
                    logger.error("简单模式合成在多次重试后仍然失败。")
                    raise

        logger.info(f"完成\n{final_message.content}")
        return final_message.content

    def get_search_config(self, task: Task, inquiry_plan: Dict[str, Any], search_type: Literal['text_summary', 'upper_design', 'upper_search']) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {search_type} \n{json.dumps(inquiry_plan, indent=2, ensure_ascii=False)}")

        current_level = len(task.id.split("."))
        configs = {
            'text_summary': {
                'kg_filters_list': [
                    "n.status = 'active'",
                    f"n.run_id = '{task.run_id}'",
                    "n.content_type = 'write'"
                ],
                'vector_filters_list': [{'key': 'content_type', 'value': 'summary'}],
                'vector_tool_desc': "功能: 检索情节摘要、角色关系、事件发展。范围: 所有层级的历史摘要。",
                'final_instruction': "任务: 整合情节摘要(向量)与正文细节(图谱)，提供写作上下文。重点: 角色关系、关键伏笔、情节呼应。",
                'rules_text': """
# 整合规则
1.  冲突解决: 向量(摘要)与图谱(细节)冲突时, 以图谱为准。
2.  排序依据:
    - 向量: 相关性。
    - 图谱: 章节顺序 (时间线)。
3.  输出要求: 整合为连贯叙述, 禁止罗列。
""",
                'vector_sort_by': 'narrative',
                'kg_sort_by': 'narrative',
            },
            'upper_design': {
                'kg_filters_list': [
                    "n.status = 'active'",
                    f"n.run_id = '{task.run_id}'",
                    "n.content_type = 'design'",
                    f"n.hierarchy_level < {current_level}"
                ],
                'vector_filters_list': [{'key': 'content_type', 'value': 'design'}],
                'vector_tool_desc': f"功能: 检索小说设定、摘要、概念。范围: 任务层级 < {current_level}。",
                'final_instruction': "任务: 整合上层设计, 提供统一、无冲突的宏观设定和指导原则。",
                'rules_text': """
# 整合规则
1.  时序优先: 结果按时间倒序。遇直接矛盾, 采纳最新版本。
2.  矛盾 vs. 细化:
    - 矛盾: 无法共存的描述 (A是孤儿 vs A父母健在)。
    - 细化: 补充细节, 不推翻核心 (A会用剑 -> A擅长流风剑法)。细化应融合, 不是矛盾。
3.  输出要求:
    - 融合非冲突信息。
    - 报告被忽略的冲突旧信息。
    - 禁止罗列, 聚焦问题。
""",
                'vector_sort_by': 'time',
                'kg_sort_by': 'time',
            },
            'upper_search': {
                'kg_filters_list': [
                    "n.status = 'active'",
                    f"n.run_id = '{task.run_id}'",
                    "n.content_type = 'search'",
                    f"n.hierarchy_level < {current_level}"
                ],
                'vector_filters_list': [{'key': 'content_type', 'value': 'search'}],
                'vector_tool_desc': f"功能: 从外部研究资料库检索事实、概念、历史事件。范围: 任务层级 < {current_level}。",
                'final_instruction': "任务: 整合外部研究资料, 提供准确、有深度、经过批判性评估的背景支持。",
                'rules_text': """
# 整合规则
1.  评估: 批判性评估所有信息。
2.  冲突处理:
    - 识别并报告矛盾。
    - 列出冲突来源和时间戳。
    - 无法解决时, 保留不确定性。
3.  时效性: `created_at`是评估因素, 但非唯一标准。结果按相关性排序。
4.  输出要求: 组织为连贯报告, 禁止罗列。
""",
                'vector_sort_by': 'relevance',
                'kg_sort_by': 'relevance',
            }
        }

        config = configs[search_type]
        config["kg_filters_str"] = " AND ".join(config['kg_filters_list'])
        config["kg_tool_desc"] = f"功能: 探索实体、关系、路径。查询必须满足过滤条件: {config['kg_filters_str']}。"
        config["query_text"] = self._build_agent_query(inquiry_plan, config['final_instruction'], config['rules_text'])

        logger.info(f"结束 \n{json.dumps(config, indent=2, ensure_ascii=False)}")
        return config

    def _build_agent_query(self, inquiry_plan: Dict[str, Any], final_instruction: str, rules_text: str) -> str:
        """
        构建一个结构化的、详细的查询文本, 用于指导 ReAct Agent 或最终的合成 LLM。

        这个查询文本由三部分组成:
        1.  核心探询目标: 来自探询计划的主查询。
        2.  具体信息需求: 将探询计划中的问题列表化, 并标注优先级。
        3.  任务指令与规则: 包含最终目标和具体的执行规则 (如优先级处理、冲突解决等)。
        Args:
            inquiry_plan (Dict[str, Any]): LLM 生成的探询计划。
            final_instruction (str): 针对当前任务的最终目标描述。
            rules_text (str): 针对当前任务的特定整合规则。
        """
        logger.info(f"开始 \n{inquiry_plan}\n{final_instruction}\n{rules_text}")

        main_inquiry = inquiry_plan.get("main_inquiry", "请综合分析并回答以下问题。")

        # 1. 构建“核心探询目标”和“具体信息需求”部分
        query_text = f"# 核心探询目标\n{main_inquiry}\n\n# 具体信息需求\n"
        has_priorities = False
        for need in inquiry_plan.get("information_needs", []):
            description = need.get('description', '未知需求')
            priority = need.get('priority', 'medium')
            if priority in ['high', 'low']: # 检查是否存在高/低优先级, 以便后续添加规则
                has_priorities = True
            questions = need.get('questions', [])
            query_text += f"\n## {description} (优先级: {priority})\n"
            for q in questions:
                query_text += f"- {q}\n"

        # 2. 构建“任务指令与规则”部分
        instruction_block = f"\n# 任务指令与规则\n"
        instruction_block += f"## 最终目标\n{final_instruction}\n"
        instruction_block += f"\n## 执行规则\n"
        if has_priorities:
            instruction_block += "- 优先级: 你必须优先分析和回答标记为 `high` 优先级的信息需求。\n"

        # 动态调整传入规则的 Markdown 标题层级, 使其能正确嵌入到当前结构中
        adapted_rules_text = re.sub(r'^\s*#\s+', '### ', rules_text.lstrip(), count=1)
        instruction_block += adapted_rules_text

        query_text += instruction_block

        logger.info(f"结束 \n{query_text}")
        return query_text

    def _create_time_aware_tool(self, query_engine: Any, name: str, description: str, sort_by: Literal['time', 'narrative', 'relevance'] = 'relevance') -> "FunctionTool":
        """
        将一个查询引擎包装成一个 Agent 可用的、具备排序功能的工具 (FunctionTool)。
        Args:
            query_engine: LlamaIndex 的查询引擎实例 (如 VectorQueryEngine)。
            name (str): 工具的名称, Agent 会用这个名字来调用它。
            description (str): 工具的功能描述, Agent 根据这个描述来决定何时使用该工具。
            sort_by (Literal): 结果的排序方式, 会传递给 `_format_response_with_sorting`。
        Returns:
            FunctionTool: 一个可供 ReActAgent 使用的工具。
        """
        async def time_aware_query(query_str: str) -> str:
            # 实际执行查询的内部函数
            response: Response = await query_engine.aquery(query_str)
            # 查询后, 使用指定的排序方式格式化结果
            return self._format_response_with_sorting(response, sort_by)

        # 使用 LlamaIndex 的 FunctionTool.from_defaults 创建工具
        return FunctionTool.from_defaults(
            fn=time_aware_query,
            name=name,
            description=description
        )

    def _format_response_with_sorting(self, response: Response, sort_by: Literal['time', 'narrative', 'relevance']) -> str:
        """
        格式化查询响应, 并根据指定策略对来源节点进行排序。
        Args:
            response (Response): 查询引擎返回的响应对象。
            sort_by (Literal): 排序策略: 'time' (时间倒序), 'narrative' (章节顺序), 'relevance' (相关性)。
        """
        if not response.source_nodes:
            return f"未找到相关来源信息，但综合回答是：\n{str(response)}"

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

        logger.info(f"结束 \n{final_output}")
        return final_output

###############################################################################

_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance

###############################################################################
#                           测试代码 (Test Code)                              #
###############################################################################

if __name__ == '__main__':
    import unittest
    from unittest.mock import patch, MagicMock, AsyncMock

    # 为测试模拟 LlamaIndex 的 Response 和 Node 对象
    class MockNode:
        def __init__(self, content, metadata, score):
            self._content = content
            self.metadata = metadata
            self._score = score

        def get_content(self):
            return self._content

        def get_score(self):
            return self._score

    class MockResponse:
        def __init__(self, response_text, source_nodes):
            self._response_text = response_text
            self.source_nodes = source_nodes

        def __str__(self):
            return self._response_text

    class TestRAGMethods(unittest.IsolatedAsyncioTestCase):
        """
        针对 RAG 类中核心方法的单元测试。
        通过模拟外部依赖 (LLM, 数据库, 向量存储), 专注于测试内部逻辑。
        """
        def setUp(self):
            """在每个测试开始前, 模拟所有外部依赖。"""
            # 模拟环境变量, 避免在 RAG 初始化时进行真实连接
            self.env_patcher = patch.dict(os.environ, {
                "embed_model": "", # 强制使用本地模型路径
                "local_embed_model": "mock_model", # 提供一个假的本地模型名
                "qdrant_host": "localhost",
                "qdrant_port": "6333",
                "memgraph_url": "bolt://localhost:7687",
                "memgraph_username": "memgraph",
                "memgraph_password": "memgraph",
                "deployment_environment": "test" # 启用详细日志
            })
            self.env_patcher.start()

            # 模拟外部客户端和模型加载, 避免 I/O 操作
            self.hf_embedding_patcher = patch('utils.rag.HuggingFaceEmbedding', autospec=True)
            self.qdrant_client_patcher = patch('utils.rag.QdrantClient', autospec=True)
            self.memgraph_store_patcher = patch('utils.rag.MemgraphGraphStore', autospec=True)
            
            self.mock_hf_embedding = self.hf_embedding_patcher.start()
            self.mock_qdrant_client = self.qdrant_client_patcher.start()
            self.mock_memgraph_store = self.memgraph_store_patcher.start()

            # 现在可以安全地实例化 RAG 类
            self.rag_instance = RAG()

        def tearDown(self):
            """在每个测试结束后, 停止所有模拟。"""
            self.env_patcher.stop()
            self.hf_embedding_patcher.stop()
            self.qdrant_client_patcher.stop()
            self.memgraph_store_patcher.stop()

        @patch('utils.rag.llm_acompletion', new_callable=AsyncMock)
        async def test_execute_simple(self, mock_llm_completion):
            """测试 _execute_simple 方法的逻辑: 并行查询 -> 格式化 -> LLM 合成。"""
            print("\n--- 正在运行: test_execute_simple ---")
            
            # 1. 准备: 设置模拟数据和对象
            inquiry_plan = {"information_needs": [{"questions": ["主角是谁?", "主要情节是什么?"]}]}
            config = {"query_text": "# 核心目标\n回答问题。", "vector_sort_by": "relevance", "kg_sort_by": "relevance"}

            # 模拟查询引擎
            mock_vector_engine = MagicMock()
            mock_vector_engine.aquery = AsyncMock(return_value=MockResponse("向量搜索找到主角是爱丽丝。", [MockNode("爱丽丝是一个勇敢的战士。", {"task_id": "1.1", "created_at": "2023-01-01T12:00:00"}, 0.9)]))
            mock_kg_engine = MagicMock()
            mock_kg_engine.aquery = AsyncMock(return_value=MockResponse("知识图谱找到爱丽丝和鲍勃是朋友。", [MockNode("爱丽丝 -> 是朋友 -> 鲍勃", {"task_id": "1.2", "created_at": "2023-01-01T13:00:00"}, 0.8)]))

            # 模拟最终的 LLM 合成调用
            mock_final_message = MagicMock()
            mock_final_message.content = "主角是一个名叫爱丽丝的勇敢战士, 她是鲍勃的朋友。"
            mock_llm_completion.return_value = mock_final_message

            # 2. 执行: 运行被测试的方法
            result = await self.rag_instance._execute_simple(inquiry_plan=inquiry_plan, vector_query_engine=mock_vector_engine, kg_query_engine=mock_kg_engine, config=config)

            # 3. 断言: 检查行为和输出是否符合预期
            self.assertEqual(result, "主角是一个名叫爱丽丝的勇敢战士, 她是鲍勃的朋友。")
            mock_vector_engine.aquery.assert_awaited_once_with("主角是谁?\n主要情节是什么?")
            mock_kg_engine.aquery.assert_awaited_once_with("主角是谁?\n主要情节是什么?")
            mock_llm_completion.assert_awaited_once()
            user_prompt = mock_llm_completion.call_args[0][0]['messages'][1]['content']
            self.assertIn("向量搜索找到主角是爱丽丝。", user_prompt)
            self.assertIn("知识图谱找到爱丽丝和鲍勃是朋友。", user_prompt)
            print("--- 测试通过: test_execute_simple ---")

        @patch('utils.rag.ReActAgent', autospec=True)
        async def test_execute_react_agent(self, MockReActAgent):
            """测试 _execute_react_agent 方法的逻辑: 创建工具 -> 初始化 Agent -> 执行。"""
            print("\n--- 正在运行: test_execute_react_agent ---")

            # 1. 准备
            config = {"query_text": "查找关于爱丽丝的信息。", "vector_tool_desc": "向量搜索工具。", "kg_tool_desc": "知识图谱搜索工具。", "vector_sort_by": "relevance", "kg_sort_by": "relevance"}
            mock_vector_engine = MagicMock()
            mock_kg_engine = MagicMock()

            # 模拟 ReActAgent 实例及其 `achat` 方法
            mock_agent_instance = MockReActAgent.return_value
            mock_agent_instance.achat = AsyncMock(return_value="Agent 发现爱丽丝是一个勇敢的战士。")

            # 2. 执行
            result = await self.rag_instance._execute_react_agent(vector_query_engine=mock_vector_engine, kg_query_engine=mock_kg_engine, config=config)

            # 3. 断言
            self.assertEqual(result, "Agent 发现爱丽丝是一个勇敢的战士。")
            MockReActAgent.assert_called_once()
            agent_init_kwargs = MockReActAgent.call_args.kwargs
            self.assertEqual(len(agent_init_kwargs['tools']), 2)
            self.assertEqual(agent_init_kwargs['tools'][0].metadata.name, "time_aware_vector_search")
            mock_agent_instance.achat.assert_awaited_once_with("查找关于爱丽丝的信息。")
            print("--- 测试通过: test_execute_react_agent ---")

    # 运行测试
    unittest.main()
