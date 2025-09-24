from loguru import logger
from typing import Any, List, Literal

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
from llama_index.llms.litellm import LiteLLM


table_summary_prompt = """
# 角色
你是一位顶尖的数据分析师和信息架构师。

# 任务
为下方的表格内容, 生成一个结构化的、信息密集的摘要。

# 核心原则
1. 直接输出: 直接生成标题和摘要内容, 禁止使用任何引导性或对话性语句。
2. 信息提炼: 摘要必须精准地概括表格的核心主题、关键洞察和数据趋势。
3. 忠于原文: 所有信息必须来源于表格本身, 禁止外部知识。
4. 考虑元数据: 如果上下文中提供了表格的来源信息(如文件名), 请在分析时考虑。

# 输出格式
必须严格遵循以下格式, 使用 Markdown:

## [表格标题]
[对表格内容的详细、精炼的摘要]

# 表格内容
---
{table_code}
---
"""


mermaid_summary_prompt = """
# 角色
你是一位顶尖的信息架构师和图表分析专家。

# 任务
为下方的 Mermaid 图表代码, 生成一个结构化的、信息密集的摘要。

# 核心原则
1. 直接输出: 直接生成标题和摘要内容, 禁止使用任何引导性或对话性语句。
2. 信息提炼: 摘要必须精准地概括图表的核心主题、关键实体、关系和结构。
3. 忠于原文: 所有信息必须来源于图表本身, 禁止外部知识。
4. 概括整体: 不要逐条罗列连接关系, 而是从整体上描述图表所表达的结构、流程或层级。
5. 解释意图: 如果可能, 推断并解释图表的设计意图或它所解决的问题。

# 输出格式
必须严格遵循以下格式, 使用 Markdown:

## [图表标题]
[对图表内容的详细、精炼的摘要]

# Mermaid 图表代码
---
{mermaid_code}
---
"""



class CustomMarkdownNodeParser(MarkdownElementNodeParser):

    _mermaid_summary_prompt: PromptTemplate = PrivateAttr()

    def __init__(self, llm: LiteLLM, table_summary_prompt: str, mermaid_summary_prompt: str, **kwargs: Any):
        super().__init__(llm=llm, summary_query_str=table_summary_prompt, **kwargs)
        self._mermaid_summary_prompt = PromptTemplate(mermaid_summary_prompt)


    def _parse_table(self, element: Any, node: TextNode) -> List[BaseNode]:
        """
        解析表格元素。
        此方法覆盖默认行为, 将摘要放入一个单独节点的内容中, 而不是元数据中, 以避免元数据过长。
        """
        table_md = element.element
        
        try:
            table_df = element.table
            table_text_for_summary = "\n\n".join(["Table:", table_df.to_string()])
        except Exception:
            logger.warning("无法将表格转换为 pandas DataFrame 进行摘要, 使用原始 markdown。")
            table_text_for_summary = table_md

        if self.llm is None:
            return [TextNode(text=table_md, metadata=node.metadata)]

        logger.debug("正在为 Markdown 表格生成摘要...")
        summary_response = self.llm.predict(
            PromptTemplate(self.summary_query_str),
            table_code = f"{table_text_for_summary}\n\n来源文件: {node.metadata.get('file_name', '')}"
        )
        logger.debug(f"Markdown 表格摘要生成完毕:\n{summary_response}")

        # 将摘要和内容合并到单个节点中, 确保它们在检索时总是一起出现。
        # 使用分隔符清晰地组织内容。
        combined_text = (
            f"这是一个表格, 其摘要如下:\n{summary_response}\n\n"
            f"---\n\n"
            f"原始表格Markdown内容:\n{table_md}"
        )
        return [TextNode(text=combined_text, metadata=node.metadata.copy())]


    def _parse_mermaid(self, mermaid_code: str, metadata: dict) -> List[BaseNode]:
        """解析 Mermaid 图表, 生成摘要节点和代码节点。"""
        if not mermaid_code.strip():
            return []

        logger.debug("正在为 Mermaid 图表生成摘要...")
        summary_response = self.llm.predict(
            self._mermaid_summary_prompt, 
            mermaid_code=mermaid_code
        )
        logger.debug(f"Mermaid 图表摘要生成完毕:\n{summary_response}")

        # 同样, 将Mermaid图表的摘要和代码合并到单个节点。
        combined_text = (
            f"这是一个Mermaid图表, 其摘要如下:\n{summary_response}\n\n"
            f"---\n\n"
            f"原始Mermaid图表代码:\n```mermaid\n{mermaid_code}\n```"
        )
        return [TextNode(text=combined_text, metadata=metadata)]


    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """从单个节点中获取节点列表。"""
        logger.debug(f"CustomMarkdownNodeParser: 开始从节点 (ID: {node.id_}) 提取子节点...")
        text = node.get_content()
        import re
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
    if content_length > 0 and content_length < 512:
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(
            chunk_size=512, 
            chunk_overlap=100,
        )
    if content_format == "json":
        from llama_index.core.node_parser import JSONNodeParser
        return JSONNodeParser(
            include_metadata=True,
            max_depth=5, 
            levels_to_keep=2
        )
    elif content_format == "md":
        from llama_index.core import Settings
        return CustomMarkdownNodeParser(
            llm=Settings.llm,
            table_summary_prompt=table_summary_prompt,
            mermaid_summary_prompt=mermaid_summary_prompt,
            chunk_size=2048,
            chunk_overlap=400,
        )
    from llama_index.core.node_parser import SentenceSplitter
    return SentenceSplitter(
        chunk_size=512, 
        chunk_overlap=100,
    )
