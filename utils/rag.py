#coding: utf8
import os
import json
import hashlib
import asyncio
from loguru import logger
from diskcache import Cache
from datetime import datetime
from typing import Dict, Any, List, Literal, Optional, Callable, Union, TYPE_CHECKING
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.db import get_db
from utils.models import Task
from utils.prompt_loader import load_prompts
from utils.llm import LLM_PARAMS_fast, LLM_PARAMS_reasoning, get_llm_messages, get_llm_params, llm_acompletion, LLM_TEMPERATURES
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.indices.prompt_helper import PromptHelper


class RAG:
    def __init__(self):
        self._embed_model: Optional[Union[LiteLLMEmbedding, HuggingFaceEmbedding]] = None
        self._qdrant_client: Optional[QdrantClient] = None
        self._qdrant_aclient: Optional["AsyncQdrantClient"] = None
        self._graph_store: Optional[MemgraphGraphStore] = None
        self._agent_llm: Optional[LiteLLM] = None
        self._extraction_llm: Optional[LiteLLM] = None

        if TYPE_CHECKING:
            from qdrant_client import QdrantClient, AsyncQdrantClient
            from llama_index.graph_stores.memgraph import MemgraphGraphStore
            from llama_index.core import StorageContext

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

    @property
    def embed_model(self) -> Union[LiteLLMEmbedding, HuggingFaceEmbedding]:
        if self._embed_model is None:
            logger.info("正在延迟加载 embed_model...")
            embed_model_name = os.getenv("embed_model")
            embed_BASE_URL = os.getenv("embed_BASE_URL")
            embed_API_KEY = os.getenv("embed_API_KEY")
            embed_dimensions = int(os.getenv("embed_dims", "1024"))
            if embed_model_name and embed_BASE_URL and embed_API_KEY:
                logger.info(f"使用远程嵌入模型: {embed_model_name}")
                self._embed_model = LiteLLMEmbedding(
                    model_name=embed_model_name,
                    api_base=embed_BASE_URL,
                    api_key=embed_API_KEY,
                    kwargs={"dimensions": embed_dimensions},
                )
                self._embed_model.get_text_embedding("test")
            else:
                model_identifier = os.getenv("local_embed_model", "BAAI/bge-m3")
                logger.info(f"使用本地嵌入模型: {model_identifier}")
                model_folder_name = model_identifier.split("/")[-1]
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..")
                )
                local_model_path = os.path.join(
                    project_root, "models", model_folder_name
                )
                if not os.path.isdir(local_model_path):
                    raise FileNotFoundError(
                        f"本地嵌入模型路径不存在: {local_model_path}. "
                        "请确认模型已下载或配置了远程嵌入模型 API。"
                    )
                # import torch
                # device = "cuda" if torch.cuda.is_available() else "cpu"
                # self._embed_model = HuggingFaceEmbedding(model_name=local_model_path, device=device)
        return self._embed_model

    @property
    def qdrant_client(self) -> "QdrantClient":
        from qdrant_client import QdrantClient
        if self._qdrant_client is None:
            logger.info("正在延迟加载 qdrant_client...")
            self._qdrant_client = QdrantClient(
                host=os.getenv("qdrant_host", "localhost"),
                port=int(os.getenv("qdrant_port", "6333")),
            )
        return self._qdrant_client

    @property
    def qdrant_aclient(self) -> "AsyncQdrantClient":
        from qdrant_client import AsyncQdrantClient
        if self._qdrant_aclient is None:
            logger.info("正在延迟加载 qdrant_aclient...")
            self._qdrant_aclient = AsyncQdrantClient(
                host=os.getenv("qdrant_host", "localhost"),
                port=int(os.getenv("qdrant_port", "6333")),
            )
        return self._qdrant_aclient

    @property
    def graph_store(self) -> "MemgraphGraphStore":
        from llama_index.graph_stores.memgraph import MemgraphGraphStore
        if self._graph_store is None:
            logger.info("正在延迟加载 graph_store...")
            self._graph_store = MemgraphGraphStore(
                url=os.getenv("memgraph_url", "bolt://localhost:7687"),
                username=os.getenv("memgraph_username", "memgraph"),
                password=os.getenv("memgraph_password", "memgraph"),
            )
        return self._graph_store

    @property
    def agent_llm(self) -> LiteLLM:
        if self._agent_llm is None:
            logger.info("正在延迟加载 agent_llm...")
            self._agent_llm = LiteLLM(**LLM_PARAMS_reasoning)
        return self._agent_llm

    @property
    def extraction_llm(self) -> LiteLLM:
        if self._extraction_llm is None:
            logger.info("正在延迟加载 extraction_llm...")
            self._extraction_llm = LiteLLM(**LLM_PARAMS_fast)
        return self._extraction_llm

    def _get_storage_context(self, run_id: str) -> "StorageContext":
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.core import StorageContext
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            aclient=self.qdrant_aclient,
            collection_name=f"write_vectors_{run_id}"
        )
        return StorageContext.from_defaults(
            vector_store=vector_store,
            graph_store=self.graph_store
        )

###############################################################################

    async def add(self, task: Task, task_type: str):
        db = get_db(run_id=task.run_id, category=task.category)
        if task_type == "task_atom":
            if task.id == "1" or task.results.get("goal_update"):
                self.caches['task_list'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_task, task)
        elif task_type == "task_plan_before_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store(task, "design", task.results.get("design_reflection"))
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
                await self.store(task, "design", task.results.get("design_reflection"))
        elif task_type == "task_execute_search":
            if task.results.get("search"):
                self.caches['dependent_search'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store(task, "search", task.results.get("search"))
        elif task_type == "task_execute_write_before_reflection":
            if task.results.get("design_reflection"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store(task, "design", task.results.get("design_reflection"))
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
                await asyncio.to_thread(self.text_file_append, self.get_text_file_path(task), content)
                await self.store(task, "write", write_reflection)
        elif task_type == "task_execute_summary":
            if task.results.get("summary"):
                self.caches['text_summary'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store(task, "summary", task.results.get("summary"))
        elif task_type == "task_aggregate_design":
            if task.results.get("design"):
                self.caches['dependent_design'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store(task, "design", task.results.get("design"))
        elif task_type == "task_aggregate_search":
            if task.results.get("search"):
                self.caches['dependent_search'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store(task, "search", task.results.get("search"))
        elif task_type == "task_aggregate_summary":
            if task.results.get("summary"):
                self.caches['text_summary'].evict(tag=task.run_id)
                await asyncio.to_thread(db.add_result, task)
                await self.store(task, "summary", task.results.get("summary"))
        else:
            raise ValueError("不支持的任务类型")

    def text_file_append(self, file_path: str, content: str):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{content}")
            f.flush()
            os.fsync(f.fileno())

    def get_text_file_path(self, task: Task) -> str:
        return os.path.join("output", task.category, f"{task.run_id}.txt")

    async def store(self, task: Task, content_type: Literal['design', 'search', 'write', 'summary'], content: str):
        """
        - 向量存储 (Vector Store): 存储 'design', 'search', 'summary' 类型的内容。
        - 知识图谱 (Knowledge Graph): 存储 'design', 'search', 'write' 类型的内容。
        """
        if not content: 
            logger.warning("内容为空, 跳过存储。")
            return

        if content_type in ["design", "search", "summary"]:
            header_parts = [
                task.id,
                task.hierarchical_position,
                task.goal,
            ]
            if task.length:
                header_parts.append(task.length)
            header = " ".join(filter(None, header_parts))
            content = f"# 任务\n{header}\n\n{content}"

        from llama_index.core import Document, StorageContext, VectorStoreIndex, KnowledgeGraphIndex
        from llama_index.core.prompts import PromptTemplate
        from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser

        storage_context = self._get_storage_context(task.run_id)
        
        doc_metadata = {
            "run_id": task.run_id,                      # 运行ID, 用于数据隔离
            "task_id": task.id,                         # 任务ID, 用于追溯来源
            "hierarchy_level": len(task.id.split(".")), # 任务层级, 用于范围查询
            "content_type": content_type,               # 内容类型, 用于过滤
            "status": "active",                         # 状态, 用于标记/取消文档
            "created_at": datetime.now().isoformat()    # 创建时间, 用于时序排序
        }
        doc = Document(id_=task.id, text=content, metadata=doc_metadata)

        node_parser = None
        if content_type in ['design', 'summary', 'search']:
            logger.info(f"为 '{content_type}' 类型使用 MarkdownNodeParser。")
            node_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)
        else: 
            logger.info(f"为 '{content_type}' 类型使用 SentenceSplitter。")
            # 对于小说等叙事性强的文本, 增加重叠区域(chunk_overlap)可以更好地保留上下文联系,
            # 避免句子或段落在切分时被割裂, 从而提升检索和生成质量。
            # 将重叠从50增加到100 (约为 chunk_size 的 20%) 是一个常见的优化策略。
            node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

        if content_type in ['design', 'search', 'summary']:
            logger.info(f"正在将 '{content_type}' 内容存入向量索引...")
            await asyncio.to_thread(
                VectorStoreIndex.from_documents, 
                [doc], 
                storage_context=storage_context, 
                embed_model=self.embed_model,
                transformations=[node_parser]
            )
            logger.info(f"完成 向量索引")

        if content_type in ['design', 'search', 'write']:
            logger.info(f"正在从 '{content_type}' 内容中提取并存入知识图谱...")
            kg_extraction_prompt_design, kg_extraction_prompt_write, kg_extraction_prompt_search = load_prompts(task.category, "graph_cn", "kg_extraction_prompt_design", "kg_extraction_prompt_write", "kg_extraction_prompt_search")
            prompts = {
                "design": PromptTemplate(kg_extraction_prompt_design),
                "write": PromptTemplate(kg_extraction_prompt_write),
                "search": PromptTemplate(kg_extraction_prompt_search),
            }
            kg_extraction_prompt = prompts.get(content_type)
            await asyncio.to_thread(
                KnowledgeGraphIndex.from_documents,
                [doc],
                storage_context=storage_context,
                embed_model=self.embed_model, 
                llm=self.extraction_llm, 
                kg_extraction_prompt=kg_extraction_prompt, 
                include_embeddings=True,        # 将图谱节点与向量嵌入关联, 支持混合搜索
                max_triplets_per_chunk=15,      # 每个文本块最多提取15个三元组
                transformations=[node_parser]   # 使用与向量索引相同的解析器, 保证块的一致性
            )
            logger.info(f"完成 知识图谱")
            

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
            task_list = await self.get_context_task_list(db, task)
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
            rag_context_coros["upper_design"] = self.search_context(task, 'upper_design', dependent_design, dependent_search, text_latest, task_list)
            # rag_context_coros["upper_search"] = self.search_context(task, 'upper_search', dependent_design, dependent_search, text_latest, task_list)
    
        if len(text_latest) > 500:
            rag_context_coros["text_summary"] = self.search_context(task, 'text_summary', dependent_design, dependent_search, text_latest, task_list)
        
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

    async def get_context_task_list(self, db: Any, task: Task) -> str:
        cache_key = f"task_list:{task.run_id}:{task.parent_id}"
        cached_result = await asyncio.to_thread(self.caches['task_list'].get, cache_key)
        if cached_result is not None:
            return cached_result
        result = await asyncio.to_thread(db.get_context_task_list, task)
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
        file_path = self.get_text_file_path(task)

        key = f"get_text_length:{file_path}"
        cached_result = await asyncio.to_thread(self.caches['text_length'].get, key)
        if cached_result is not None:
            return cached_result
        
        full_content = await asyncio.to_thread(self.text_file_read, file_path)
        length = len(full_content)
        
        await asyncio.to_thread(self.caches['text_length'].set, key, length, tag=task.run_id)

        return length

    def text_file_read(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    async def get_context_aggregate_design(self, task: Task) -> Dict[str, Any]:
        ret = await self.get_context_base(task)
        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_design"] = await asyncio.to_thread(db.get_subtask_design, task.id)
        return ret

    async def get_context_aggregate_search(self, task: Task) -> Dict[str, Any]:
        ret = await self.get_context_base(task)
        if task.sub_tasks:
            db = get_db(run_id=task.run_id, category=task.category)
            ret["subtask_search"] = await asyncio.to_thread(db.get_subtask_search, task.id)
        return ret

    async def get_context_aggregate_summary(self, task: Task) -> Dict[str, Any]:
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

    async def get_query(self, task: Task, search_type: Literal['text_summary', 'upper_design', 'upper_search'], dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> Dict[str, Any]:
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
                include={'task_type', 'hierarchical_position', 'goal', 'length'}
            ),
            "dependent_design": dependent_design,
            "dependent_search": dependent_search,
            "text_latest": text_latest,
            "task_list": task_list
        }
        SYSTEM_PROMPT, USER_PROMPT = PROMPTS[search_type]
        if task.task_type == 'write' and search_type == 'upper_design' and task.results["atom_result"] == "atom":
            SYSTEM_PROMPT = SYSTEM_PROMPT_design_for_write
        messages = get_llm_messages(SYSTEM_PROMPT, USER_PROMPT, None, context_dict_user)
        llm_params = get_llm_params(messages, temperature=LLM_TEMPERATURES["reasoning"])
        message = await llm_acompletion(llm_params, response_model=InquiryPlan)
        inquiry_plan_obj = message.validated_data
        inquiry_plan = inquiry_plan_obj.model_dump()
        return inquiry_plan

    async def search_context(self, task: Task, search_type: Literal['text_summary', 'upper_design', 'upper_search'], dependent_design: str, dependent_search: str, text_latest: str, task_list: str) -> str:
        logger.info(f"[{task.id}][{search_type}] 开始执行 search_context...")

        key_data = {
            "run_id": task.run_id,
            "task_id": task.id,
            "task_level": len(task.id.split(".")),
            "search_type": search_type
        }
        cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode('utf-8')).hexdigest()
        cache = self.caches[search_type]
        cached_result = await asyncio.to_thread(cache.get, cache_key)
        if cached_result is not None:
            logger.info(f"[{task.id}][{search_type}] 缓存命中, 直接返回结果。")
            return cached_result

        inquiry_plan = await self.get_query(task, search_type, dependent_design, dependent_search, text_latest, task_list)
        if not inquiry_plan or not inquiry_plan.get("questions"):
            logger.warning("生成的探询计划为空或无效, 跳过搜索。")
            return ""

        from llama_index.core import VectorStoreIndex
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
        get_search_config, kg_gen_query_prompt = load_prompts(task.category, "rag_cn", "get_search_config", "kg_gen_query_prompt")

        config = get_search_config(task, inquiry_plan, search_type)
        storage_context = self._get_storage_context(task.run_id)
        retrieval_strategies = {
            # 上层设计: 大范围召回(25), 精排(rerank)出最相关的8个。确保宏观方向的绝对准确, 避免遗漏关键设定。
            'upper_design': {'similarity_top_k': 45, 'rerank_top_n': 15, 'use_rerank': True},
            # 情节摘要: 超大范围召回(40), 精排(rerank)出最关键的10个。为长篇小说提供足够深度的上下文, 保证情节的长期连贯性和伏笔呼应。
            'text_summary': {'similarity_top_k': 60, 'rerank_top_n': 20, 'use_rerank': True},
            # 外部资料: 较大范围召回(20), 精排(rerank)出最相关的5个。在保证信息覆盖面的同时, 控制外部资料的噪音。
            'upper_search': {'similarity_top_k': 30, 'rerank_top_n': 10, 'use_rerank': True},
        }
        strategy = retrieval_strategies.get(search_type, {'similarity_top_k': 5, 'rerank_top_n': 3, 'use_rerank': False})
        logger.info(f"[{task.id}][{search_type}] 采用策略: {strategy}")

        logger.info(f"[{task.id}][{search_type}] 配置节点后处理器 (reranker)...")
        node_postprocessors = []
        if strategy['use_rerank']:
            from llama_index.core.postprocessor import LLMRerank
            reranker = LLMRerank(
                llm=self.agent_llm,
                top_n=strategy['rerank_top_n'],
                verbose=os.getenv("deployment_environment") == "test"
            )
            node_postprocessors.append(reranker)
            logger.info(f"[{task.id}][{search_type}] Reranker 已配置, top_n={strategy['rerank_top_n']}.")

        logger.info(f"[{task.id}][{search_type}] 配置 Agent LLM 的上下文管理器和响应合成器...")
        agent_llm_context_window = LLM_PARAMS_reasoning["context_window"]
        agent_llm_max_completion_tokens = LLM_PARAMS_reasoning["max_completion_tokens"]
        prompt_helper_agent = PromptHelper(
            context_window=agent_llm_context_window,
            num_output=agent_llm_max_completion_tokens,
            chunk_overlap_ratio=0.2
        )
        response_synthesizer_agent = CompactAndRefine(
            llm=self.agent_llm,
            prompt_helper=prompt_helper_agent
        )
        logger.info(f"[{task.id}][{search_type}] Agent LLM 配置完成。")

        logger.info(f"[{task.id}][{search_type}] 构建向量查询引擎...")
        vector_filters = MetadataFilters(
            filters=[MetadataFilter(key=f['key'], value=f['value']) for f in config['vector_filters_list']]
        )
        vector_index = await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            storage_context.vector_store,
            embed_model=self.embed_model
        )
        vector_query_engine = vector_index.as_query_engine(
            filters=vector_filters,
            llm=self.agent_llm,
            response_synthesizer=response_synthesizer_agent,
            similarity_top_k=strategy['similarity_top_k'],
            node_postprocessors=node_postprocessors
        )
        logger.info(f"[{task.id}][{search_type}] 向量查询引擎构建完成。")

        logger.info(f"[{task.id}][{search_type}] 构建知识图谱查询引擎...")
        from llama_index.core.prompts import PromptTemplate
        from llama_index.core import KnowledgeGraphIndex
        kg_query_gen_prompt = PromptTemplate(kg_gen_query_prompt.format(kg_filters_str=config["kg_filters_str"]))
        kg_index = await asyncio.to_thread(
            KnowledgeGraphIndex.from_documents,
            [],
            storage_context=storage_context,
            llm=self.agent_llm,
            include_embeddings=True,
            embed_model=self.embed_model
        )
        kg_query_engine = kg_index.as_query_engine(
            include_text=False,
            graph_query_synthesis_prompt=kg_query_gen_prompt,
            llm=self.agent_llm,
            response_synthesizer=response_synthesizer_agent
        )
        logger.info(f"[{task.id}][{search_type}] 知识图谱查询引擎构建完成。")

        logger.info(f"[{task.id}][{search_type}] 判断并选择执行模式 (简单模式 vs ReAct Agent)...")
        novel_length = await self.get_text_length(task)
        retrieval_mode = inquiry_plan.get("retrieval_mode", "simple")
        if novel_length > 100000 and retrieval_mode == 'complex' and search_type != 'upper_search':
            logger.info(f"[{task.id}][{search_type}] 选择 ReAct Agent 模式 (小说长度={novel_length}, 检索模式='{retrieval_mode}')。")
            result = await self._execute_react_agent(
                task=task,
                vector_query_engine=vector_query_engine,
                kg_query_engine=kg_query_engine,
                config=config
            )
        else:
            logger.info(f"[{task.id}][{search_type}] 选择简单模式 (小说长度={novel_length}, 检索模式='{retrieval_mode}')。")
            result = await self._execute_simple(
                task=task,
                inquiry_plan=inquiry_plan,
                vector_query_engine=vector_query_engine,
                kg_query_engine=kg_query_engine,
                config=config
            )

        await asyncio.to_thread(cache.set, cache_key, result, tag=task.run_id)
        return result
    
    async def _execute_simple(self, task: Task, inquiry_plan: Dict[str, Any], vector_query_engine: Any, kg_query_engine: Any, config: Dict[str, Any]) -> str:
        logger.info(f"检索: 简单模式 \n{json.dumps(config, indent=2, ensure_ascii=False)}\n{json.dumps(inquiry_plan, indent=2, ensure_ascii=False)}")
        
        all_questions = list(inquiry_plan.get("questions", {}).keys())
        if not all_questions:
            logger.warning(f"[{task.id}] 探询计划中没有问题, 无法执行简单模式查询。")
            return ""
        single_question = "\n".join(all_questions)
        
        get_search_config, format_response_with_sorting, kg_gen_query_prompt, synthesis_system_prompt, synthesis_user_prompt = load_prompts(
            task.category, "rag_cn", "get_search_config", "format_response_with_sorting", "kg_gen_query_prompt", "synthesis_system_prompt", "synthesis_user_prompt"
        )

        logger.info(f"[{task.id}]   - 开始向量查询...")
        vector_response = await asyncio.to_thread(vector_query_engine.query, single_question)
        logger.info(f"[{task.id}]   - 向量查询完成, 正在格式化结果...")
        formatted_vector_str = await asyncio.to_thread(format_response_with_sorting, vector_response, config['vector_sort_by'])

        logger.info(f"[{task.id}]   - 开始知识图谱查询...")
        kg_response = await asyncio.to_thread(kg_query_engine.query, single_question)
        logger.info(f"[{task.id}]   - 知识图谱查询完成, 正在格式化结果...")
        formatted_kg_str = await asyncio.to_thread(format_response_with_sorting, kg_response, config['kg_sort_by'])
        
        context_dict_user = {
            "query_text": config["query_text"],
            "formatted_vector_str": formatted_vector_str,
            "formatted_kg_str": formatted_kg_str,
        }
        synthesis_messages = get_llm_messages(synthesis_system_prompt, synthesis_user_prompt, None, context_dict_user)
        llm_params = get_llm_params(synthesis_messages, temperature=LLM_TEMPERATURES["synthesis"])
        final_message = await llm_acompletion(llm_params)
        return final_message.content

    async def _execute_react_agent(self, task: Task, vector_query_engine: Any, kg_query_engine: Any, config: Dict[str, Any]) -> str:
        logger.info(f"检索: ReActAgent 模型 \n{json.dumps(config, indent=2, ensure_ascii=False)}")

        from llama_index.core.agent import ReActAgent
        from llama_index.core.memory import ChatMemoryBuffer
        get_search_config, format_response_with_sorting, react_system_prompt = load_prompts(task.category, "rag_cn", "get_search_config", "format_response_with_sorting", "react_system_prompt")

        logger.info(f"[{task.id}] 创建 Agent 可用的工具 (Tools)...")
        vector_tool = self._create_sorted_query_tool(
            vector_query_engine,
            name="time_aware_vector_search",
            description=config['vector_tool_desc'],
            sort_by=config['vector_sort_by'],
            formatter=format_response_with_sorting
        )
        logger.info(f"[{task.id}]   - 向量搜索工具已创建。")
        kg_tool = self._create_sorted_query_tool(
            kg_query_engine,
            name="time_aware_knowledge_graph_search",
            description=config["kg_tool_desc"],
            sort_by=config['kg_sort_by'],
            formatter=format_response_with_sorting
        )
        logger.info(f"[{task.id}]   - 知识图谱工具已创建。")
        tools = [vector_tool, kg_tool]

        logger.info(f"[{task.id}] 初始化聊天记忆并创建 ReActAgent 实例...")
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.agent_llm,
            memory=memory,
            system_prompt=react_system_prompt,
            verbose=os.getenv("deployment_environment") == "test"
        )
        logger.info(f"[{task.id}] ReActAgent 实例创建完成。")

        logger.info(f"[{task.id}] 开始执行 Agent...")
        response = await agent.achat(config["query_text"])
        final_answer = str(response)
        logger.info(f"[{task.id}] Agent 执行完成, 最终答案:\n{final_answer}")
        return final_answer

    def _create_sorted_query_tool(self, query_engine: Any, name: str, description: str, sort_by: Literal['time', 'narrative', 'relevance'], formatter: Callable) -> Any:
        """
        将一个查询引擎包装成一个 Agent 可用的、具备排序功能的工具 (FunctionTool)。
        """
        from llama_index.core.tools import FunctionTool
        from llama_index.core.base.response.schema import Response

        async def sorted_query(query_str: str) -> str:
            response: Response = await asyncio.to_thread(query_engine.query, query_str)
            return await asyncio.to_thread(formatter, response, sort_by)

        # 使用 LlamaIndex 的 FunctionTool.from_defaults 创建工具
        return FunctionTool.from_defaults(
            fn=sorted_query,
            name=name,
            description=description
        )

###############################################################################

_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance
