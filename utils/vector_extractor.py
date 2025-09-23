import re
from loguru import logger
from typing import Any, List, Literal

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import JSONNodeParser, SentenceSplitter
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.llms.litellm import LiteLLM


summary_query_str = """
这个表格是关于什么的?请给出一个非常简洁的摘要(想象你正在为这个表格添加一个新的标题和摘要)。
如果上下文中提供了表格的真实/现有标题, 请输出它。
如果上下文中提供了表格的真实/现有ID, 请输出它。
并输出是否应该保留该表格。
"""


mermaid_summary_prompt = """
# 角色
你是一位精通图表解读的分析师。

# 任务
阅读下方的 Mermaid 图表代码, 并生成一段简洁、流畅、易于理解的自然语言摘要。

# 核心原则
1.  **识别核心**: 找出图表中的关键实体(节点)和它们之间的核心关系(连接)。
2.  **概括整体**: 不要逐条罗列连接关系, 而是从整体上描述图表所表达的结构、流程或层级。例如, "此图表展示了一个三层架构, ... " 或 "该流程图描述了用户从登录到完成购买的完整步骤, ..."。
3.  **解释意图**: 如果可能, 推断并解释图表的设计意图或它所解决的问题。
4.  **忠于图表**: 摘要必须完全基于图表内容, 禁止引入外部信息。

# Mermaid 图表代码
---------------------
{mermaid_code}
---------------------

# 内容摘要
"""


class CustomMarkdownNodeParser(MarkdownElementNodeParser):

    _mermaid_summary_prompt: PromptTemplate = PrivateAttr()

    def __init__(self, llm: LiteLLM, summary_query_str: str, mermaid_summary_prompt: str, **kwargs: Any):
        super().__init__(llm=llm, summary_query_str=summary_query_str, **kwargs)
        self._mermaid_summary_prompt = PromptTemplate(mermaid_summary_prompt)

    def _create_summary_and_content_nodes(
        self, summary_text: str, content_text: str, metadata: dict
    ) -> List[BaseNode]:
        """辅助函数, 用于创建摘要节点和内容节点, 并将它们链接起来。"""
        summary_node = TextNode(text=summary_text, metadata=metadata)
        content_node = TextNode(text=content_text, metadata=metadata)
        logger.debug(f"创建了摘要节点 (ID: {summary_node.id_}) 和内容节点 (ID: {content_node.id_})。")

        summary_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=content_node.id_
        )
        content_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=summary_node.id_
        )
        logger.debug("已在摘要节点和内容节点之间建立双向关系。")
        return [summary_node, content_node]

    def _parse_table(self, element: Any, node: TextNode) -> List[BaseNode]:
        """
        解析表格元素。
        此方法覆盖默认行为, 将摘要放入一个单独节点的内容中, 而不是元数据中, 以避免元数据过长。
        """
        table_md = element.element
        
        try:
            # markdown-it-py 的表格解析插件应提供 to_pandas 方法
            table_df = element.table
            table_text_for_summary = "\n\n".join(["Table:", table_df.to_string()])
        except Exception:
            # 如果由于某种原因它不是可转换为 pandas 的对象, 则回退
            logger.warning("无法将表格转换为 pandas DataFrame 进行摘要, 使用原始 markdown。")
            table_text_for_summary = table_md

        if self.llm is None:
            return [TextNode(text=table_md, metadata=node.metadata)]

        logger.debug("正在为 Markdown 表格生成摘要...")
        summary_response = self.llm.predict(
            PromptTemplate(self.summary_query_str),
            context_str=table_text_for_summary,
        )
        logger.debug(f"Markdown 表格摘要生成完毕:\n{summary_response}")

        return self._create_summary_and_content_nodes(
            summary_text=f"表格摘要:\n{summary_response}",
            content_text=table_md,
            metadata=node.metadata.copy(),
        )

    def _parse_mermaid(self, mermaid_code: str, metadata: dict) -> List[BaseNode]:
        """解析 Mermaid 图表, 生成摘要节点和代码节点。"""
        if not mermaid_code.strip():
            return []

        logger.debug("正在为 Mermaid 图表生成摘要...")
        summary_response = self.llm.predict(
            self._mermaid_summary_prompt, mermaid_code=mermaid_code
        )
        logger.debug(f"Mermaid 图表摘要生成完毕:\n{summary_response}")

        return self._create_summary_and_content_nodes(
            summary_text=f"Mermaid图表摘要:\n{summary_response}",
            content_text=f"```mermaid\n{mermaid_code}\n```",
            metadata=metadata,
        )

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """从单个节点中获取节点列表。"""
        logger.debug(f"CustomMarkdownNodeParser: 开始从节点 (ID: {node.id_}) 提取子节点...")
        text = node.get_content()
        parts = re.split(r"(```mermaid\n.*?\n```)", text, flags=re.DOTALL)

        final_nodes: List[BaseNode] = []
        source_document = node.source_node or node.as_related_node_info()

        for part in parts:
            if not part.strip():
                continue

            if part.startswith("```mermaid"):
                logger.debug("在 Markdown 中检测到 Mermaid 图表, 正在提取...")
                mermaid_code = part.removeprefix("```mermaid\n").removesuffix("\n```")
                mermaid_nodes = self._parse_mermaid(mermaid_code, node.metadata)
                logger.debug(f"  - Mermaid 图表部分提取了 {len(mermaid_nodes)} 个节点。")
                final_nodes.extend(mermaid_nodes)
            else:
                logger.debug("在 Markdown 中检测到常规文本部分, 正在解析...")
                # 1. 提取元素
                elements = self.extract_elements(part, table_filters=[self.filter_table], node_id=node.node_id)
                elements = self.extract_html_tables(elements)

                # 2. 从元素创建节点
                for element in elements:
                    if element.type == "table":
                        # 使用自定义的表格解析逻辑
                        table_nodes = self._parse_table(element, node)
                        final_nodes.extend(table_nodes)
                    else:
                        # 处理文本和其他类型的元素
                        final_nodes.append(
                            TextNode(text=element.element, metadata=node.metadata, id_=element.id)
                        )

        # 为所有新创建的节点设置源文档关系和元数据
        for n in final_nodes:
            n.relationships[NodeRelationship.SOURCE] = source_document
            n.metadata.update(node.metadata)

        logger.debug(f"CustomMarkdownNodeParser 完成处理, 共生成 {len(final_nodes)} 个子节点。")
        return final_nodes


###############################################################################


def get_vector_node_parser(content_format: Literal["md", "txt", "json"], content_length: int = 0) -> NodeParser:
    chunk_size = 2048
    chunk_overlap = 400
    if content_length > 0 and content_length < chunk_size:
        return SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
    if content_format == "json":
        return JSONNodeParser(
            include_metadata=True,
            max_depth=5, 
            levels_to_keep=2
        )
    elif content_format == "txt":
        return SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
    return CustomMarkdownNodeParser(
        llm=Settings.llm,
        summary_query_str=summary_query_str,
        mermaid_summary_prompt=mermaid_summary_prompt,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
