

## git
git config --global http.proxy http://127.0.0.1:10809
git config --global https.proxy https://127.0.0.1:10809


git config --global --unset http.proxy
git config --global --unset https.proxy


[user]
    name = huazhengqing
    email = 18836038@qq.com




## 
echo "配置 Docker 镜像加速..."
DOCKER_DAEMON_FILE="/etc/docker/daemon.json"
if [ ! -f "$DOCKER_DAEMON_FILE" ] || ! grep -q "registry-mirrors" "$DOCKER_DAEMON_FILE"; then
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://3xoj0j3i.mirror.aliyuncs.com",
    "https://docker.m.daocloud.io",
    "https://mirror.azure.cn",
    "https://ghcr.hub1.nat.tf",
    "https://f1361db2.m.daocloud.io"
  ]
}
EOF
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    echo "Docker 镜像加速配置完成。"
else
    echo "Docker 镜像加速已配置, 跳过配置。"
fi

docker info | grep "Registry Mirrors" -A 5





聊天交互, 提示词, 代码注释都要使用中文
写提示词时,要求: 清晰、精确、易于理解, 在保持质量的同时, 尽可能简洁, 不要有各种"黑话"和比喻, 以关键词为主








信息冗余会污染上下文, 浪费 Token。
单步检索无法应对复杂的剧情逻辑, 比如伏笔回收、角色动机反转等。


1. 如何解决"信息冗余", 实现精简输出?
问题:  同时从向量库(Qdrant)和图谱库(Memgraph)检索, 必然会拿到重复或关联的信息。如何合并去重, 得到最精简的结果?
LlamaIndex 方案: 响应合成器 (Response Synthesizer)
工作流程: 
并行检索 (Retrieve): LlamaIndex 向 Qdrant 和 Memgraph 同时发出查询, 各自取回一批相关的"知识片段"(在 LlamaIndex 中称为 Node)。
合成 (Synthesize): 这是关键步骤。LlamaIndex 会将所有取回的 Node(无论来自哪里, 可能包含重复信息)全部交给一个 LLM, 并给出一个特殊的指令: "请基于以下所有信息, 综合、去重、提炼, 然后生成一个连贯、精简的回答。"
输出 (Response): 最终你得到的不是一堆零散的资料, 而是一个由 LLM 帮你整理好的、高度浓缩的段落。
优势: 
自动去重:  利用 LLM 强大的语言理解能力, 自然地将"林烬是主角"和"主角是林烬"这类重复信息融合。
逻辑连接:  它能将来自图谱的"林烬 -> 拥有 -> 轩辕剑"和来自向量库的"轩辕剑是一把上古神兵, 能吸收使用者情绪"这两条信息, 合成为"主角林烬拥有的轩辕剑是一把能吸收情绪的上古神兵"。
Token 高效:  最终给到你下一个任务(比如写作)的上下文, 是经过提炼的"精华", 而不是原始、冗长的资料堆砌, 极大地节省了 Token。



2. 如何处理"复杂逻辑", 实现多步推理?
问题:  像"主角为什么在A事件后选择B路线, 这和C伏笔有什么关系?"这类问题, 需要多次查询和推理, 单次检索无法完成。
LlamaIndex 方案: 智能代理 (Agent) + 工具 (Tools)
工作流程 (以 ReAct 代理为例):
定义工具 (Tools): 我们把不同的检索能力包装成"工具"。
工具A (向量搜索): 描述为"用于查找设定、摘要等语义相似的内容"。
工具B (图谱搜索): 描述为"用于探索角色、事件之间的复杂关系"。

提出复杂问题:  你向代理提出问题: "林烬为什么在背叛帝国后, 反而去寻找他师父留下的那把断剑?这和之前的伏笔有什么关系?"
代理开始思考和行动 (Thought & Action Loop):
思考 1: "这个问题很复杂。我需要先搞清楚‘林烬’和‘帝国’的关系。"
行动 1: 代理决定使用 工具B (图谱搜索), 并自己生成一个查询 (林烬)-[]->(帝国)。
观察 1: 得到结果"林烬被帝国追杀"。
思考 2: "好的, 他是被追杀。现在我要查‘断剑’和‘伏笔’。这听起来更像是一个概念, 我用向量搜索。"
行动 2: 代理决定使用 工具A (向量搜索), 自己生成查询"师父 断剑 伏笔 秘密"。
观察 2: 得到结果"摘要: 师父临终前曾说, 断剑中藏着足以颠覆帝国的秘密..."。
思考 3: "信息齐了。林烬被帝国追杀, 而师父留下的断剑里有颠覆帝国的秘密。所以他去找剑是为了复仇和自保。我能回答最终问题了。"
最终回答:  代理将整个推理链条整合, 生成最终的、富有逻辑的答案。

优势: 
自动化推理:  你不需要手动设计复杂的检索流程, 代理会自己规划步骤、选择工具、执行查询。
处理复杂因果:  完美契合小说创作中追踪伏笔、分析角色动机等复杂需求。
极致的 Token 效率:  在多步推理中, 只有最终的、经过整合的结论会作为上下文输出, 中间过程的检索结果和思考链条都是代理内部消耗的, 不会浪费你的 Token 预算。




LlamaIndex 不仅能很好地支持你的需求, 它简直就是为解决这类复杂 RAG 场景而设计的。通过: 
响应合成器 来解决信息冗余和 Token 效率问题。
智能代理 (Agent) 来解决多步推理和复杂逻辑关联问题。
我们可以构建一个既自动化又强大的记忆系统, 它能真正理解你小说的内在逻辑, 为你提供最高质量的上下文。


result_serializerPREFECT_RESULTS_DEFAULT_SERIALIZERpickle"pickle""json""compressed/pickle""compressed/json"

result_serializerPREFECT_RESULTS_DEFAULT_SERIALIZERpickle"pickle""json""compressed/pickle""compressed/json"



result_storage_key="{parameters[task].run_id}/{parameters[task].id}/atom.pickle", 



import importlib.util
from typing import List, Dict, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class DuckDuckGoSearchToolSpec(BaseToolSpec):
    """DuckDuckGoSearch tool spec."""

    spec_functions = ["duckduckgo_instant_search", "duckduckgo_full_search"]

    def __init__(self) -> None:
        if not importlib.util.find_spec("duckduckgo_search"):
            raise ImportError(
                "DuckDuckGoSearchToolSpec requires the duckduckgo_search package to be installed."
            )
        super().__init__()

    def duckduckgo_instant_search(self, query: str) -> List[Dict]:
        """
        Make a query to DuckDuckGo api to receive an instant answer.

        Args:
            query (str): The query to be passed to DuckDuckGo.

        """
        from duckduckgo_search import DDGS

        with DDGS() as ddg:
            return list(ddg.answers(query))

    def duckduckgo_full_search(
        self,
        query: str,
        region: Optional[str] = "wt-wt",
        max_results: Optional[int] = 10,
    ) -> List[Dict]:
        """
        Make a query to DuckDuckGo search to receive a full search results.

        Args:
            query (str): The query to be passed to DuckDuckGo.
            region (Optional[str]): The region to be used for the search in [country-language] convention, ex us-en, uk-en, ru-ru, etc...
            max_results (Optional[int]): The maximum number of results to be returned.

        """
        from duckduckgo_search import DDGS

        params = {
            "keywords": query,
            "region": region,
            "max_results": max_results,
        }

        with DDGS() as ddg:
            return list(ddg.text(**params))



from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = DuckDuckGoSearchToolSpec()

agent = FunctionAgent(
    tools=DuckDuckGoSearchToolSpec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("What's going on with the superconductor lk-99"))
print(await agent.run("what are the latest developments in machine learning"))




from llama_index.tools.google import GmailToolSpec
from llama_index.tools.google import GoogleCalendarToolSpec
from llama_index.tools.google import GoogleSearchToolSpec
google_spec = GoogleSearchToolSpec(key="your-key", engine="your-engine")


from llama_index.tools.bing_search import BingSearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = BingSearchToolSpec(api_key="your-key")

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)

print(await agent.run("what's the latest news about superconductors"))
print(await agent.run("what does lk-99 look like"))
print(await agent.run("is there any videos of it levitating"))




from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = ArxivToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(), llm=OpenAI(model="gpt-4.1")
)

await agent.run("What's going on with the superconductor lk-99")
await agent.run("what are the latest developments in machine learning")



from llama_index_desearch.tools import DesearchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

desearch_tool = DesearchToolSpec(
    api_key=os.environ["DESEARCH_API_KEY"],
)
agent = FunctionAgent(
    tools=desearch_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("Can you find the latest news on quantum computing?"))



from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tavily_tool = TavilyToolSpec(
    api_key="your-key",
)
agent = FunctionAgent(
    tools=tavily_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

await agent.run("What happened in the latest Burning Man festival?")









