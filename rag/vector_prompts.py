from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataInfo,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)


tree_summary_prompt = """
# 角色
你是一位严谨、精确的信息分析师, 你的核心职责是忠于事实。

# 任务
严格依据下方提供的`上下文信息`, 回答`问题`。

# 核心原则
1.  **绝对忠实**: 你的回答必须**完全**来源于`上下文信息`。禁止使用任何外部知识、个人推断或假设。
2.  **相关性优先**: 在生成任何回答之前, 必须首先判断`上下文信息`是否包含能直接回答`问题`的内容。
3.  **拒绝回答**: 如果`上下文信息`为空, 或者与`问题`完全无关, 无法提供有效答案, **必须**直接输出"无法回答该问题"。禁止输出任何其他内容。
4.  **综合提炼**: 如果上下文相关, 请整合所有相关信息, 形成一个有结构的、逻辑连贯的答案, 而不是简单罗列。
5.  **结构化保留**: 如果上下文中包含表格、代码或图表, 应尽可能在最终答案中保留这些结构。

# 上下文信息
---------------------
{context_str}
---------------------

# 待回答的问题
{query_str}

# 严格基于上下文的回答
"""


# text_qa_prompt = """
# # 角色
# 你是一位信息提取助手。

# # 任务
# 从下方的`上下文信息`中, 提取与`问题`相关的所有事实和描述, 并以清晰的陈述句形式呈现。

# # 核心原则
# 1. 忠于原文: 你的回答必须完全基于`上下文信息`, 禁止引入外部知识。
# 2. 提取而非回答: 你的目标是提取信息片段, 而不是直接形成对`问题`的最终答案。如果`上下文信息`只包含部分相关信息, 就只输出那部分。
# 3. 无相关则为空: 如果`上下文信息`与`问题`完全无关, 则返回空字符串。
# 4. 直接陈述: 直接列出事实, 不要添加引述性短语。

# # 上下文信息
# ---------------------
# {context_str}
# ---------------------

# # 问题
# {query_str}

# # 提取的事实
# """


# refine_prompt = """
# # 角色
# 你是一位高级信息整合师。

# # 任务
# 根据`新的上下文`, 优化`已有的回答`, 以更全面、更精确地回答`原始问题`。

# # 工作流程
# 1. 分析新信息: 仔细阅读`新的上下文`, 识别出其中包含的、但`已有的回答`中缺失或不完整的新信息点。
# 2. 比较与整合: 将新信息点与`已有的回答`进行融合, 遵循下方的核心原则。
# 3. 生成新答案: 构建一个单一、连贯、全面的新答案。

# # 核心原则
# 1. 信息完整性: 最终答案必须无缝整合`已有的回答`和`新的上下文`中的所有相关信息, 禁止丢失任何细节。
# 2. 增量优化: 你的目标是"优化"而非"重写"。只有当`新的上下文`能提供补充、修正或更具体的细节时, 才进行修改。
# 3. 冲突处理: 如果`新的上下文`与`已有的回答`中的信息发生冲突, 请综合判断, 保留更具体、更可信的信息。如果无法判断优劣, 则应同时提及两种说法并明确指出其矛盾之处。
# 4. 无效则返回原文: 如果`新的上下文`与问题无关, 或未能提供任何有价值的新信息, 请直接返回`已有的回答`, 不要做任何改动。
# 5. 风格一致: 在生成新答案时, 尽量保持`已有的回答`的语言风格和格式, 使最终答案浑然一体。

# # 原始问题
# {query_str}

# # 已有的回答
# {existing_answer}

# # 新的上下文
# ------------
# {context_msg}
# ------------

# # 优化后的回答
# """


###############################################################################


prefix = """
你的目标是将用户的查询结构化, 使其符合下方提供的请求模式(Request Schema)。

<< 结构化请求模式(Structured Request Schema)>>
回复时, 请使用 Markdown 代码片段, 其中包含一个遵循以下模式格式化的 JSON 对象: 

{schema_str}

查询字符串应仅包含预计会与文档内容匹配的文本。筛选条件(filter)中的任何条件也不应在查询(query)中提及。

确保筛选条件(filter)仅引用数据源中存在的属性。
确保筛选条件(filter)考虑了属性的描述信息。
确保仅在需要时使用筛选条件(filter)。如果不存在应应用的筛选条件, 则筛选条件(filter)的值返回 []。

如果用户的查询中明确提到了要检索的文档数量, 请将 top_k 设置为该数量；否则, 不设置 top_k。
"""

example_info = VectorStoreInfo(
    content_info="一首歌的歌词",
    metadata_info=[
        MetadataInfo(name="artist", type="str", description="歌曲艺术家的姓名"),
        MetadataInfo(
            name="genre",
            type="str",
            description='歌曲流派, 取值为 "pop"(流行)、"rock"(摇滚)或 "rap"(说唱)之一',
        ),
    ],
)

example_query = "泰勒・斯威夫特(Taylor Swift)或凯蒂・佩里(Katy Perry)创作的、属于舞曲流行(dance pop)流派的歌曲有哪些"

example_output = VectorStoreQuerySpec(
    query="青少年爱情(teenager love)",
    filters=[
        MetadataFilter(key="artist", value="泰勒・斯威夫特(Taylor Swift)"),
        MetadataFilter(key="artist", value="凯蒂・佩里(Katy Perry)"),
        MetadataFilter(key="genre", value="pop(流行)"),
    ],
)

example_info_2 = VectorStoreInfo(
    content_info="经典文学作品(Classic literature)",
    metadata_info=[
        MetadataInfo(name="author", type="str", description="作者姓名"),
        MetadataInfo(
            name="book_title",
            type="str",
            description="书籍标题(书名)",
        ),
        MetadataInfo(
            name="year",
            type="int",
            description="出版年份(Year Published)",
        ),
        MetadataInfo(
            name="pages",
            type="int",
            description="页数(Number of pages)",
        ),
        MetadataInfo(
            name="summary",
            type="str",
            description="书籍的简短摘要(A short summary of the book)",
        ),
    ],
)

example_query_2 = "简·奥斯汀(Jane Austen)在 1813 年之后出版的、探讨 \"为社会地位而结婚\" 这一主题的书籍有哪些?"

example_output_2 = VectorStoreQuerySpec(
    query="与 为社会地位而结婚 主题相关的书籍(Books related to theme of marriage for social standing)",
    filters=[
        MetadataFilter(key="year", value="1813", operator=FilterOperator.GT),
        MetadataFilter(key="author", value="简・奥斯汀(Jane Austen)"),
    ],
)

examples = f"""
<< 示例 1. >>
数据源(Data Source):
```json
{example_info.model_dump_json(indent=4)}
```

用户查询(User Query):
{example_query}

结构化请求(Structured Request):
```json
{example_output.model_dump_json()}


<< 示例 2. >>
数据源(Data Source):
```json
{example_info_2.model_dump_json(indent=4)}
```

用户查询(User Query):
{example_query_2}

结构化请求(Structured Request):
```json
{example_output_2.model_dump_json()}

```
""".replace("{", "{{").replace("}", "}}")


suffix = """
<< 示例 3. >>
数据源(Data Source):
```json
{info_str}
```

用户查询(User Query):
{query_str}

结构化请求(Structured Request):
"""


vector_store_query_prompt = prefix + examples + suffix


###############################################################################
