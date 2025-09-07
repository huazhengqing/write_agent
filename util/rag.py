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
from util.db import get_db
from util.models import Task
from util.llm import LLM_PARAMS_fast, LLM_PARAMS_reasoning, get_llm_params, llm_acompletion


"""
存入时：
    设计 (design) : LlamaIndex (VectorIndex + KGIndex)
    正文 (text): LlamaIndex (KGIndex only)
    正文摘要 (summary) : LlamaIndex (VectorIndex only)

检索时：
    - 使用 LlamaIndex 的 ReActAgent 智能体, 结合向量和图谱工具进行多步推理和检索。

信息冗余会污染上下文，浪费 Token。
单步检索无法应对复杂的剧情逻辑，比如伏笔回收、角色动机反转等。

1. 如何解决“信息冗余”，实现精简输出？
问题： 同时从向量库（Qdrant）和图谱库（Memgraph）检索，必然会拿到重复或关联的信息。如何合并去重，得到最精简的结果？

2. 如何处理“复杂逻辑”，实现多步推理？
问题： 像“主角为什么在A事件后选择B路线，这和C伏笔有什么关系？”这类问题，需要多次查询和推理，单次检索无法完成。

在`rag.py`的`search_context`方法中，ReAct Agent的日志非常详细，如何能让它在生产环境中更简洁，只输出关键的决策和最终结果？

如何修改 `rag.py` 中的 `search_context` 方法来解析并执行新的“结构化探询计划”？

"""

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

        from prompts.story.graph_cn import design_prompt, text_prompt, search_prompt
        self.kg_extraction_prompts = {
            "design": PromptTemplate(design_prompt),
            "text": PromptTemplate(text_prompt),
            "search": PromptTemplate(search_prompt),
        }

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cache_base_dir = os.path.join(project_root, ".cache", "rag")
        os.makedirs(cache_base_dir, exist_ok=True)
        self.caches: Dict[str, Cache] = {
            'dependent_design': Cache(os.path.join(cache_base_dir, "dependent_design"), size_limit=int(32 * (1024**2))),
            'dependent_search': Cache(os.path.join(cache_base_dir, "dependent_search"), size_limit=int(32 * (1024**2))),
            'text_latest': Cache(os.path.join(cache_base_dir, "text_latest"), size_limit=int(32 * (1024**2))),      'text_length': Cache(os.path.join(cache_base_dir, "text_length"), size_limit=int(1 * (1024**2))),
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
        if task_type == "task_atom":
            return await self.add_task_atom(task)
        
        if task_type == "task_plan":
            self.caches['task_list'].evict(tag=task.run_id)
            self.caches['upper_design'].evict(tag=task.run_id)
            self.caches['upper_search'].evict(tag=task.run_id)
            return await self.add_task_plan(task)

        task_result = task.results.get("result")
        if not task_result:
            logger.error(f"task_result为空, 无法添加记忆")
            return
        
        if task_type == "task_execute":
            if task.task_type == "write":
                self.caches['text_latest'].evict(tag=task.run_id)
                self.caches['text_length'].evict(tag=task.run_id)
                self.caches['text_summary'].evict(tag=task.run_id)
                return await self.add_task_execute_write(task)
            elif task.task_type == "design":
                return await self.add_result(task)
            elif task.task_type == "search":
                return await self.add_result(task)
            else:
                raise ValueError(f"不支持的任务类型: {task.task_type}")
        elif task_type == "task_aggregate":
            if task.task_type == "design":
                return await self.add_result(task)
            elif task.task_type == "search":
                return await self.add_result(task)
            elif task.task_type == "write":
                self.caches['text_summary'].evict(tag=task.run_id)
                return await self.add_result(task)
            else:
                raise ValueError(f"不支持的任务类型: {task.task_type}")
        else:
            raise ValueError(f"不支持的类型 {task_type}")

    async def add_task_atom(self, task: Task):
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        if task.id == "1" or task.results.get("goal_update"):
            db = get_db(run_id=task.run_id, category=task.category)
            await asyncio.to_thread(db.add_task, task)

            self.caches['task_list'].evict(tag=task.run_id)

        logger.info(f"完成")

    async def add_task_plan(self, task: Task):
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        db = get_db(run_id=task.run_id, category=task.category)
        await asyncio.to_thread(db.add_sub_tasks, task)

        logger.info(f"完成")

    async def add_task_execute_write(self, task: Task):
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        task_result = task.results.get("result")
        if not task_result:
            logger.error(f"task_result为空, 无法添加记忆")
            return

        await asyncio.to_thread(self.text_file_append, self.get_text_file_path(task), task_result)

        summary = await self.summary_text(task)

        task.results["text"] = task.results["result"]
        task.results["result"] = summary

        db = get_db(run_id=task.run_id, category=task.category)
        await asyncio.to_thread(db.add_text, task)

        await self.store(task, task.results["text"], content_type='text')
        await self.store(task, summary, content_type='write')

        logger.info(f"完成")

    def text_file_append(self, file_path: str, content: str):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{content}")
            f.flush()
            os.fsync(f.fileno())

    def get_text_file_path(self, task: Task) -> str:
        return os.path.join("output", task.category, f"{task.run_id}.txt")
    
    async def summary_text(self, task: Task):
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")
        
        prompt_module = __import__(f"prompts.{task.category}.write_aggregate_cn", fromlist=["SYSTEM_PROMPT", "USER_PROMPT"])
        SYSTEM_PROMPT = prompt_module.SYSTEM_PROMPT
        USER_PROMPT = prompt_module.USER_PROMPT

        context = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'goal'}
            ),
            "subtask_write": task.results.get("result")
        }
        
        safe_context = collections.defaultdict(str, context)
        user_prompt_content = USER_PROMPT.format_map(safe_context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_content}
        ]

        llm_params = get_llm_params(messages, temperature=0.2)
        
        message = await llm_acompletion(llm_params)

        logger.info(f"结束")
        return message.content


    async def add_result(self, task: Task):
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        task_result = task.results.get("result")
        if not task_result:
            logger.error(f"task_result为空, 无法添加记忆")
            return

        db = get_db(run_id=task.run_id, category=task.category)
        await asyncio.to_thread(db.add_result, task)
        
        await self.store(task, task_result, content_type=task.task_type)

        logger.info(f"完成")

    async def store(self, task: Task, content: str, content_type: Literal['design', 'search', 'write', 'text']):
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal} {content_type}")

        if not content: 
            return

        storage_context = self._get_storage_context(task.run_id)
        doc_metadata = {
            "run_id": task.run_id,
            "hierarchy_level": len(task.id.split(".")),
            "content_type": content_type, 
            "task_id": task.id, 
            "status": "active", 
            "created_at": datetime.now().isoformat()
        }
        doc = Document(id_=task.id, text=content, metadata=doc_metadata)

        if content_type in ['design', 'search', 'write']:
            await asyncio.to_thread(
                VectorStoreIndex.from_documents, 
                [doc], 
                storage_context=storage_context, 
                embed_model=self.embed_model
            )

        if content_type in ['design', 'search', 'text']:
            # 根据内容类型选择最合适的图谱提取Prompt
            kg_prompt = self.kg_extraction_prompts.get(content_type)
            if not kg_prompt:
                logger.warning(f"内容类型 '{content_type}' 没有找到对应的图谱提取Prompt, 将跳过图谱存储。")
                return

            await asyncio.to_thread(
                KnowledgeGraphIndex.from_documents,
                [doc],
                storage_context=storage_context,
                max_triplets_per_chunk=15,
                kg_extraction_prompt=kg_prompt, # 使用定制化Prompt
                llm=self.extraction_llm, # 使用为提取优化的LLM
                include_embeddings=True, # 将图谱节点与向量嵌入关联
            )
            
        logger.info(f"完成")

###############################################################################

    async def get_context_base(self, task: Task) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        db = get_db(run_id=task.run_id, category=task.category)

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

        dependent_design_task = self.get_dependent_design(db, task)
        dependent_search_task = self.get_dependent_search(db, task)
        text_latest_task = self.get_text_latest(task)

        dependent_design, dependent_search, text_latest = await asyncio.gather(
            dependent_design_task, dependent_search_task, text_latest_task
        )
        
        ret["dependent_design"] = dependent_design
        ret["dependent_search"] = dependent_search
        ret["text_latest"] = text_latest
        
        design_plan_task = self.get_query(task, "design", dependent_design, dependent_search, text_latest)
        content_plan_task = self.get_query(task, "write", dependent_design, dependent_search, text_latest)
        search_plan_task = self.get_query(task, "search", dependent_design, dependent_search, text_latest)

        design_plan, content_plan, search_plan = await asyncio.gather(
            design_plan_task, content_plan_task, search_plan_task
        )

        context_tasks = {}
        if design_plan and design_plan.get("information_needs"):
            context_tasks["upper_design"] = self.search_context(task, design_plan, 'upper_design')
        if search_plan and search_plan.get("information_needs"):
            context_tasks["upper_search"] = self.search_context(task, search_plan, 'upper_search')
        if content_plan and content_plan.get("information_needs"):
            context_tasks["text_summary"] = self.search_context(task, content_plan, 'text_summary')
        
        if len(task.id.split(".")) >= 2:
            context_tasks["task_list"] = self.get_context_task_list(db, task)

        if context_tasks:
            results = await asyncio.gather(*context_tasks.values())
            for i, key in enumerate(context_tasks.keys()):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"获取上下文 '{key}' 失败: {result}")
                    ret[key] = f"错误: 获取 {key} 上下文失败。"
                else:
                    ret[key] = result

        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret
    
    async def get_dependent_design(self, db: Any, task: Task) -> str:
        cache_key = f"dependent_design:{task.run_id}:{task.parent_id}"
        cached_result = self.caches['dependent_design'].get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await asyncio.to_thread(db.get_dependent, task, "design")

        self.caches['dependent_design'].set(cache_key, result, tag=task.run_id)
        return result

    async def get_dependent_search(self, db: Any, task: Task) -> str:
        cache_key = f"dependent_search:{task.run_id}:{task.parent_id}"
        cached_result = self.caches['dependent_search'].get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await asyncio.to_thread(db.get_dependent, task, "search")

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

        file_path = self.get_text_file_path(task)
        key = f"get_text_latest:{file_path}:{length}"
        cached_result = self.caches['text_latest'].get(key)
        if cached_result is not None:
            return cached_result

        full_content = await asyncio.to_thread(self.text_file_read, file_path)
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
            if task.task_type == "design":
                db = get_db(run_id=task.run_id, category=task.category)
                ret["subtask_design"] = await asyncio.to_thread(db.get_subtask_results, task.id, "design")

        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret

    async def get_context_aggregate_search(self, task: Task) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        ret = await self.get_context_base(task)

        if task.sub_tasks:
            if task.task_type == "search":
                db = get_db(run_id=task.run_id, category=task.category)
                ret["subtask_search"] = await asyncio.to_thread(db.get_subtask_results, task.id, "search")

        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret

    async def get_context_aggregate_write(self, task: Task) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal}")

        ret = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'id', 'goal', 'length'}
            ),
        }

        if task.sub_tasks:
            if task.task_type == "write":
                db = get_db(run_id=task.run_id, category=task.category)
                ret["subtask_write"] = await asyncio.to_thread(db.get_subtask_results, task.id, "write")

        logger.info(f"完成\n{json.dumps(ret, indent=2, ensure_ascii=False)}")
        return ret

    async def get_query(self, task: Task, category: str, dependent_design: str, dependent_search: str, text_latest: str) -> Dict[str, Any]:
        logger.info(f"开始 {task.run_id} {task.id} {task.task_type} {task.goal} {category} \n dependent_design: \n{dependent_design} \n dependent_search: \n{dependent_search} \n text_latest: \n{text_latest}")

        prompt_module = __import__(f"prompts.{task.category}.query_cn", fromlist=[
            "SYSTEM_PROMPT_design", "USER_PROMPT_design",
            "SYSTEM_PROMPT_design_for_write",
            "SYSTEM_PROMPT_write", "USER_PROMPT_write",
            "SYSTEM_PROMPT_search", "USER_PROMPT_search",
            "InquiryPlan"
        ])
        InquiryPlan = getattr(prompt_module, "InquiryPlan")
        PROMPTS = {
            "design": (getattr(prompt_module, "SYSTEM_PROMPT_design", None), getattr(prompt_module, "USER_PROMPT_design", None)),
            "write": (prompt_module.SYSTEM_PROMPT_write, prompt_module.USER_PROMPT_write),
            "search": (prompt_module.SYSTEM_PROMPT_search, prompt_module.USER_PROMPT_search),
        }
        
        if category not in PROMPTS:
            raise ValueError(f"不支持的查询生成类别: {category}")

        prompt_input = {
            "task": task.model_dump_json(
                indent=2,
                exclude_none=True,
                include={'task_type', 'goal', 'length'}
            ),
            "dependent_design": dependent_design,
            "dependent_search": dependent_search,
            "text_latest": text_latest
        }
        system_prompt, user_prompt_template = PROMPTS[category]

        # 核心修改：根据主任务类型选择不同的 design prompt
        if task.task_type == 'write' and category == 'design':
            system_prompt = prompt_module.SYSTEM_PROMPT_design_for_write

        safe_context = collections.defaultdict(str, prompt_input)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format_map(safe_context)}
        ]

        llm_params = get_llm_params(messages, temperature=0.1)
        llm_params['response_format'] = {
            "type": "json_object",
            "schema": InquiryPlan.model_json_schema()
        }

        message = await llm_acompletion(llm_params)
        response_content = message.content

        inquiry_plan_obj = InquiryPlan.model_validate_json(response_content)
        inquiry_plan = inquiry_plan_obj.model_dump()

        logger.info(f"完成 \n{json.dumps(inquiry_plan, indent=2, ensure_ascii=False)}")
        return inquiry_plan

    async def search_context(self, task: Task, inquiry_plan: Dict[str, Any], search_type: Literal['text_summary', 'upper_design', 'upper_search']) -> str:
        logger.info(f"开始 {task.run_id} {task.id} {search_type} \n{inquiry_plan}")

        if not inquiry_plan or not inquiry_plan.get("information_needs"):
            logger.warning("收到的探询计划为空或无效, 跳过搜索。")
            return ""

        config = self.get_search_config(task, inquiry_plan, search_type)

        key_data = {
            "run_id": task.run_id,
            "inquiry_plan": inquiry_plan,
            "search_type": config['kg_content_type'],
            "task_level": len(task.id.split(".")),
        }
        cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode('utf-8')).hexdigest()
        cache = self.caches[config['cache_name']]
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        storage_context = self._get_storage_context(task.run_id)

        #
        # vector
        #
        vector_index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store, 
            embed_model=self.embed_model
        )
        vector_query_engine = vector_index.as_query_engine(
            filters=config['vector_filters']
        )

        #
        # kg
        #
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
1.  **强制过滤 (最重要!)**: 生成的查询 **必须** 包含一个 `WHERE` 子句, 以确保数据隔离。过滤条件为: `{config["kg_filters_str"]}`。
    - **正确示例**: `MATCH (n) WHERE {config["kg_filters_str"]} AND n.property = 'value' RETURN n`
    - **错误示例 (缺少强制过滤)**: `MATCH (n) WHERE n.property = 'value' RETURN n`
2.  **Schema 遵从**: 只能使用图谱 Schema 中定义的节点标签和关系类型。严禁使用任何 Schema 之外的元素。
3.  **单行输出**: 最终的 Cypher 查询必须是单行文本, 不含换行符。
4.  **效率优先**: 在满足查询需求的前提下, 生成的查询应尽可能高效。

# 行动
现在, 请为上述用户问题生成 Cypher 查询语句。
"""
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

        # 判断是简单检索，还是复杂检索
        novel_length = await self.get_text_length(task)
        retrieval_mode = inquiry_plan.get("retrieval_mode", "simple")
        # 'upper_search' 总是使用简单模式，因为它处理的是外部事实，整合逻辑比多步推理更重要。
        use_agent = novel_length > 100000 and retrieval_mode == 'complex' and search_type != 'upper_search'

        if use_agent:
            result = await self._execute_react_agent(
                vector_query_engine=vector_query_engine,
                kg_query_engine=kg_query_engine,
                config=config
            )
        else:
            result = await self._execute_simple(
                inquiry_plan=inquiry_plan,
                vector_query_engine=vector_query_engine,
                kg_query_engine=kg_query_engine,
                config=config
            )

        cache.set(cache_key, result, tag=task.run_id)

        logger.info(f"完成 \n{result}")
        return result

    async def _execute_react_agent(self, vector_query_engine: Any, kg_query_engine: Any, config: Dict[str, Any]) -> str:
        logger.info(f"开始 复杂模式 ReActAgent \n{json.dumps(config, indent=2, ensure_ascii=False)}")

        vector_tool = self._create_time_aware_tool(
            vector_query_engine,
            name="time_aware_vector_search",
            description=config['vector_tool_desc'],
            sort_by=config['vector_sort_by']
        )

        kg_tool = self._create_time_aware_tool(
            kg_query_engine,
            name="time_aware_knowledge_graph_search",
            description=config["kg_tool_desc"],
            sort_by=config['kg_sort_by']
        )

        agent = ReActAgent(
            tools=[vector_tool, kg_tool],
            llm=self.agent_llm,
            verbose=os.getenv("deployment_environment") == "test"
        )

        response = await agent.achat(config["query_text"])
        ret = str(response)

        logger.info(f"完成\n{ret}")
        return ret

    async def _execute_simple(self, inquiry_plan: Dict[str, Any], vector_query_engine: Any, kg_query_engine: Any, config: Dict[str, Any]) -> str:
        """
        执行一个简单的“检索-然后-综合”流程。
        之所以不直接使用 LlamaIndex 的 ResponseSynthesizer 或 RouterQueryEngine，原因如下：
        1.  **复杂的、上下文感知的整合规则**: 我们的整合逻辑不仅仅是总结文本。它需要根据 `search_type` (如 'upper_design', 'text_summary') 应用不同的、非常具体的规则。
            例如，对于 'upper_design'，规则是“当设计冲突时，采纳最新版本”；对于 'upper_search'，规则是“报告信息间的矛盾，保留不确定性”。
            这种动态的、基于规则的决策逻辑很难用标准的 ResponseSynthesizer 实现。
        2.  **对异构数据源的显式控制**: 我们从向量库（语义相似）和知识图谱（结构化关系）获取信息。通过分别查询然后自定义整合，我们可以精确控制如何处理和呈现来自这两种不同来源的信息。
            例如，我们可以明确告诉最终的 LLM：“向量工具提供的是摘要，图谱工具提供的是细节，如果冲突，以细节为准。”
        3.  **保留和利用元数据**: 我们的 `_format_response_with_sorting` 函数根据元数据（如创建时间、任务ID）对结果进行排序。这个排序后的、带有丰富元数据的结果被完整地提供给最终的 LLM，
            这对于执行“采纳最新版本”等规则至关重要。
        
        总而言之，我们自己构建了一个“自定义的、基于规则的响应合成器”，通过一次额外的 LLM 调用来实现比通用组件更强大、更具控制力的整合能力。
        """
        logger.info(f"开始 简单模式 \n{json.dumps(config, indent=2, ensure_ascii=False)}")
        
        all_questions = [q for need in inquiry_plan.get("information_needs", []) for q in need.get("questions", [])]
        if not all_questions:
            logger.warning("探询计划中没有问题, 无法执行查询。")
            return ""
        single_question = "\n".join(all_questions)

        vector_response_task = vector_query_engine.aquery(single_question)
        kg_response_task = kg_query_engine.aquery(single_question)
        vector_response, kg_response = await asyncio.gather(vector_response_task, kg_response_task)

        formatted_vector_str = self._format_response_with_sorting(vector_response, config['vector_sort_by'])
        formatted_kg_str = self._format_response_with_sorting(kg_response, config['kg_sort_by'])
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
- **分析**: 严格遵循“探询计划与规则”中的“最终目标”和“执行规则”。
- **整合**: 综合两个信息源, 注意各自侧重（语义 vs 事实）。
- **应用**: 严格应用“执行规则”处理信息（如解决冲突、融合信息）。
- **内容**:
    - 必须完全基于提供的信息源。
    - 必须是整合提炼后的答案, 禁止罗列。
    - 禁止任何关于你自身或任务过程的描述 (例如, “根据您的要求...”)。
"""
        synthesis_messages = [
            {"role": "system", "content": "角色：信息整合分析师。任务：遵循用户指令，整合并提炼向量检索和知识图谱的信息。输出：一个逻辑连贯、事实准确、完全基于所提供材料的最终回答。"},
            {"role": "user", "content": synthesis_user_prompt}
        ]

        llm_params = get_llm_params(synthesis_messages, temperature=self.agent_llm.temperature)
        final_message = await llm_acompletion(llm_params)

        logger.info(f"完成\n{final_message.content}")
        return final_message.content

    def get_search_config(self, task: Task, inquiry_plan: Dict[str, Any], search_type: Literal['text_summary', 'upper_design', 'upper_search']) -> Dict[str, Any]:
        current_level = len(task.id.split("."))
        configs = {
            'text_summary': {
                'cache_name': 'text_summary',
                'kg_content_type': 'text',
                'kg_filters_list': [
                    "n.status = 'active'",
                    f"n.run_id = '{task.run_id}'",
                    "n.content_type = 'text'"
                ],
                'vector_filters': MetadataFilters(filters=[MetadataFilter(key="content_type", value="write")]),
                'vector_tool_desc': "功能: 检索情节摘要、角色关系、事件发展。范围: 所有层级的历史摘要。",
                'final_instruction': "任务: 整合情节摘要(向量)与正文细节(图谱)，提供写作上下文。重点: 角色关系、关键伏笔、情节呼应。",
                'rules_text': """
                    # 整合规则
                    1.  **冲突解决**: 向量(摘要)与图谱(细节)冲突时, 以图谱为准。
                    2.  **排序依据**:
                        - 向量: 相关性。
                        - 图谱: 章节顺序 (时间线)。
                    3.  **输出要求**: 整合为连贯叙述, 禁止罗列。
                """,
                'vector_sort_by': 'narrative',
                'kg_sort_by': 'narrative',
            },
            'upper_design': {
                'cache_name': 'upper_design',
                'kg_content_type': 'design',
                'kg_filters_list': [
                    "n.status = 'active'",
                    f"n.run_id = '{task.run_id}'",
                    "n.content_type = 'design'",
                    f"n.hierarchy_level < {current_level}"
                ],
                'vector_filters': MetadataFilters(filters=[MetadataFilter(key="content_type", value="design")]),
                'vector_tool_desc': f"功能: 检索小说设定、摘要、概念。范围: 任务层级 < {current_level}。",
                'final_instruction': "任务: 整合上层设计, 提供统一、无冲突的宏观设定和指导原则。",
                'rules_text': """
                    # 整合规则
                    1.  **时序优先**: 结果按时间倒序。遇直接矛盾, 采纳最新版本。
                    2.  **矛盾 vs. 细化**:
                        - 矛盾: 无法共存的描述 (A是孤儿 vs A父母健在)。
                        - 细化: 补充细节, 不推翻核心 (A会用剑 -> A擅长流风剑法)。细化应融合, 不是矛盾。
                    3.  **输出要求**:
                        - 融合非冲突信息。
                        - 报告被忽略的冲突旧信息。
                        - 禁止罗列, 聚焦问题。
                """,
                'vector_sort_by': 'time',
                'kg_sort_by': 'time',
            },
            'upper_search': {
                'cache_name': 'upper_search',
                'kg_content_type': 'search',
                'kg_filters_list': [
                    "n.status = 'active'",
                    f"n.run_id = '{task.run_id}'",
                    "n.content_type = 'search'",
                    f"n.hierarchy_level < {current_level}"
                ],
                'vector_filters': MetadataFilters(filters=[MetadataFilter(key="content_type", value="search")]),
                'vector_tool_desc': f"功能: 从外部研究资料库检索事实、概念、历史事件。范围: 任务层级 < {current_level}。",
                'final_instruction': "任务: 整合外部研究资料, 提供准确、有深度、经过批判性评估的背景支持。",
                'rules_text': """
                    # 整合规则
                    1.  **评估**: 批判性评估所有信息。
                    2.  **冲突处理**:
                        - 识别并报告矛盾。
                        - 列出冲突来源和时间戳。
                        - 无法解决时, 保留不确定性。
                    3.  **时效性**: `created_at`是评估因素, 但非唯一标准。结果按相关性排序。
                    4.  **输出要求**: 组织为连贯报告, 禁止罗列。
                """,
                'vector_sort_by': 'relevance',
                'kg_sort_by': 'relevance',
            }
        }

        config = configs[search_type]
        config["kg_filters_str"] = " AND ".join(config['kg_filters_list'])
        config["kg_tool_desc"] = f"功能: 探索实体、关系、路径。查询必须满足过滤条件: {config['kg_filters_str']}。"
        config["query_text"] = self._build_agent_query(inquiry_plan, config['final_instruction'], config['rules_text'])

        return config

    def _build_agent_query(self, inquiry_plan: Dict[str, Any], final_instruction: str, rules_text: str) -> str:
        main_inquiry = inquiry_plan.get("main_inquiry", "请综合分析并回答以下问题。")

        # 1. 构建核心探询目标和具体信息需求部分
        query_text = f"# 核心探询目标\n{main_inquiry}\n\n# 具体信息需求\n"
        has_priorities = False
        for need in inquiry_plan.get("information_needs", []):
            description = need.get('description', '未知需求')
            priority = need.get('priority', 'medium')
            if priority in ['high', 'low']:
                has_priorities = True
            questions = need.get('questions', [])
            query_text += f"\n## {description} (优先级: {priority})\n"
            for q in questions:
                query_text += f"- {q}\n"

        # 2. 构建结构化的指令与规则部分
        instruction_block = f"\n# 任务指令与规则\n"
        instruction_block += f"## 最终目标\n{final_instruction}\n"
        instruction_block += f"\n## 执行规则\n"
        if has_priorities:
            instruction_block += "- **优先级**: 你必须优先分析和回答标记为 `high` 优先级的信息需求。\n"

        # 动态调整传入规则的标题层级, 使其适应新的结构
        adapted_rules_text = re.sub(r'^\s*#\s+', '### ', rules_text.lstrip(), count=1)
        instruction_block += adapted_rules_text

        query_text += instruction_block
        return query_text

    def _create_time_aware_tool(self, query_engine: Any, name: str, description: str, sort_by: Literal['time', 'narrative', 'relevance'] = 'relevance') -> "FunctionTool":
        """
        创建一个包装了查询引擎的工具。该工具会异步查询，并根据指定的策略对结果进行排序和格式化。
        - 'time': 按创建时间倒序，用于处理版本冲突。
        - 'narrative': 按任务ID（章节顺序）正序，用于理解故事线。
        - 'relevance': 按向量搜索的默认相关性排序。
        """
        async def time_aware_query(query_str: str) -> str:
            response: Response = await query_engine.aquery(query_str)
            return self._format_response_with_sorting(response, sort_by)

        return FunctionTool.from_defaults(
            fn=time_aware_query,
            name=name,
            description=description
        )

    def _format_response_with_sorting(self, response: Response, sort_by: Literal['time', 'narrative', 'relevance']) -> str:
        """
        一个辅助函数, 用于根据指定的策略对 LlamaIndex 的响应进行排序和格式化。
        这确保了无论是简单查询还是Agent工具调用, 输出格式都保持一致。
        """
        if not response.source_nodes:
            return f"未找到相关来源信息，但综合回答是：\n{str(response)}"

        # 默认按LlamaIndex返回的相关性排序
        sorted_nodes = response.source_nodes
        sort_description = ""

        if sort_by == 'narrative':
            # 定义一个自然排序的key函数，能正确处理 '1.2' 和 '1.10'
            def natural_sort_key(s: str) -> List[Any]:
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

            sorted_nodes = sorted(
                list(response.source_nodes),
                key=lambda n: natural_sort_key(n.metadata.get("task_id", "")),
                reverse=False  # 正序排列
            )
            sort_description = "按小说章节顺序排列 (从前到后)"
        elif sort_by == 'time':
            # 按时间戳对来源节点进行降序排序
            sorted_nodes = sorted(
                list(response.source_nodes),
                key=lambda n: n.metadata.get("created_at", "1970-01-01T00:00:00"),
                reverse=True  # 倒序排列
            )
            sort_description = "按时间倒序排列 (最新的在前)"
        else: # relevance
            sort_description = "按相关性排序"

        source_details = []
        for node in sorted_nodes:
            timestamp = node.metadata.get("created_at", "未知时间")
            task_id = node.metadata.get("task_id", "未知章节")
            score = node.get_score()
            score_str = f"{score:.4f}" if score is not None else "N/A"
            # 移除文本中的换行符以简化输出
            content = re.sub(r'\s+', ' ', node.get_content()).strip()
            source_details.append(f"来源信息 (章节: {task_id}, 时间: {timestamp}, 相关性: {score_str}):\n---\n{content}\n---")

        formatted_sources = "\n\n".join(source_details)

        final_output = f"综合回答:\n{str(response)}\n\n详细来源 ({sort_description}):\n{formatted_sources}"
        return final_output


###############################################################################

_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance
