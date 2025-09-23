"""
该文件集中存储了用于各类测试的文本和数据样本。

通过统一管理, 可以确保测试数据的一致性、可复用性, 并方便未来扩展。
文件内容按以下结构组织: 
1. 基础与边缘用例 (Basic & Edge Cases)
2. 结构化内容 (Structured Content)
3. 特殊格式与代码块 (Special Formats & Code Blocks)
4. 领域场景: 小说创作 (Domain Scenario: Novel Writing)
5. 领域场景: 报告撰写 (Domain Scenario: Report Writing)
6. 领域场景: 技术文档 (Domain Scenario: Technical Documentation)
7. 测试数据集集合 (Test Data Collections)

命名规范: VECTOR_TEST_{CATEGORY}_{DESCRIPTION}
"""

# ==============================================================================
# 1. 基础与边缘用例 (Basic & Edge Cases)
# 涵盖基本文件格式、空内容、简单中英文等。
# ==============================================================================

VECTOR_TEST_EMPTY = ""

VECTOR_TEST_SIMPLE_TXT = """
这是一个用于向量库测试的简单文本文件。
"""

VECTOR_TEST_SIMPLE_CN = """
这是一个基础的中文测试文本。
包含简单的句子和段落结构。
"""

VECTOR_TEST_SIMPLE_MD = """
# 基础测试文档
这是一个用于向量库测试的简单Markdown文档。
"""

VECTOR_TEST_SIMPLE_JSON = '''
{
  "type": "test",
  "content": "简单JSON格式测试数据"
}
'''

VECTOR_TEST_MIXED_LANG = """
# Project Phoenix: AIGC 融合项目

这是一个关于 Project Phoenix 的技术文档。该项目的目标是创建一个 AIGC (AI-Generated Content) 平台, 能够 a) support 多语言输入, 包括中文和 English；b) 自动进行 task decomposition；c) 调用合适的 model or tool。

The core component is the `Orchestrator`, 它负责解析 user query 并生成 a DAG (Directed Acyclic Graph) of tasks. 每个 node 代表一个 a sub-task。
"""

# ==============================================================================
# 2. 结构化内容 (Structured Content)
# 涵盖Markdown的各种语法(表格、列表)、JSON结构、长篇幅等。
# ==============================================================================

VECTOR_TEST_TABLE_DATA = """
# 势力成员表

| 姓名 | 门派 | 职位 |
|---|---|---|
| 萧炎 | 炎盟 | 盟主 |
| 林动 | 武境 | 武祖 |
| 叶凡 | 天庭 | 天帝 |

## 功法清单
- 焚决
- 大荒芜经
- 天帝经
"""

VECTOR_TEST_LARGE_TABLE_DATA = """
# 附录: 九霄大陆关键角色信息表 (第一卷至第三卷)

| 角色ID | 姓名 | 别称/称号 | 阵营 | 核心身份 | 首次登场 | 当前境界 (截至第三卷末) | 持有关键物品 | 与主角关系 | 角色弧光/核心事件 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| C001 | 龙傲天 | 天选之子, 鸿蒙道体 | 主角阵营 | 从地球穿越而来的天命主角 | 第一章 | 元婴初期 | 玄铁吊坠, 御水决, 赤霄剑 | 主角本人 | 从迷茫的穿越者成长为有担当的守护者。经历了黑风寨之乱、东海寻药、宗门大比等事件, 结识挚友, 树立宿敌, 明确了守护身边之人的道心。 | 鸿蒙道体是其最大外挂, 但其心性成长才是故事核心。 |
| C002 | 叶良辰 | 魔子, 北境少主 | 敌对阵营 | 北冥魔殿少主, 主角的宿命之敌 | 第三章 | 金丹圆满 | 吞天魔功秘籍, 海图残卷 | 宿敌 | 作为主角的镜像存在, 同样天赋异禀但信奉力量至上。前期不断给主角制造麻烦, 在东海夺走海图残卷, 立下三年之约。其行为背后似乎有更深层的动机。 | 并非纯粹的恶, 其冷酷背后有复杂的成长经历。 |
| C003 | 赵日天 | 铁憨憨, 镇海侯之子 | 主角阵营 | 临海镇镇海侯的独子, 主角的第一个挚友 | 第一章 | 筑基后期 | 镇海神拳谱 | 挚友/兄弟 | 从一个行事鲁莽的富家公子, 在与主角的历险中逐渐变得成熟稳重。在黑风寨之乱中为主角挡刀重伤, 是激发主角变强的重要催化剂。 | 表面憨厚, 但身世不凡, 其家族与皇室有千丝万缕的联系。 |
| C004 | 风清扬 | 剑圣 | 主角阵营 | 青云宗大长老, 主角的授业恩师 | 第十章 | 化神中期 (推测) | - | 师徒 | 看似闲云野鹤, 实则深不可测。在主角加入青云宗后发现其潜力, 收为关门弟子, 传授无上剑道, 是主角成长道路上的重要引路人。 | 他的过去是个谜, 似乎与上一代正魔大战有重要关联。 |
| C005 | 苏沐雪 | 冰雪仙子 | 摇摆/中立 | 天机阁圣女, 拥有预知能力 | 第十五卷 (宗门大比) | 金丹中期 | 天机镜碎片 | 亦敌亦友/潜在爱慕 | 作为天机阁传人, 她身负推演天机、维持大陆平衡的使命。初见时因天机显示主角会带来浩劫而对其抱有警惕, 但在多次接触后逐渐改观。 | 她的预言往往是片段式的, 会引发误会, 推动情节发展。 |
| C006 | 幽冥老祖 | 魔皇 | 敌对阵营 | 北冥魔殿殿主, 叶良辰的父亲 | 第二卷 (回忆中) | 化神后期 (传说) | 吞天魔功 | 宿敌之父/幕后黑手 | 老一辈魔道巨擘, 手段残忍, 野心勃勃, 意图颠覆正道统治。对儿子叶良辰的教育极为严苛, 是其扭曲性格的根源。 | 他的最终目标可能并非简单的统治大陆, 而是与上古秘辛有关。 |
| C007 | 药尘 | 药老 | 主角阵营 | 寄宿在玄铁吊坠中的上古药圣残魂 | 第五章 (主角濒死时) | 灵魂状态 (生前为斗圣) | 骨灵冷火, 焚决 | 师徒/引导者 | 在主角濒死时苏醒, 成为其另一位强大导师, 传授炼药术和《焚决》。他的存在为主角解决了无数丹药和功法问题, 是主角逆天改命的关键。 | 他正在寻找重塑肉身的方法, 其仇家也是大陆上的顶级势力。 |
"""

VECTOR_TEST_NESTED_LIST = """
# 物品清单

- **神兵利器**
  1. 赤霄剑: 龙傲天的佩剑, 削铁如泥。
  2. 诛仙四剑: 上古遗留的杀伐至宝, 分为四柄。
     - 诛仙剑
     - 戮仙剑
     - 陷仙剑
     - 绝仙剑
  3. 方天画戟: 霸道无匹的长兵器。
- **灵丹妙药**
  - 九转还魂丹: 可活死人, 肉白骨。
  - 菩提子: 辅助悟道, 提升心境。
- **珍稀材料**
  - 千年玄铁: 铸造神兵的顶级材料。
"""

VECTOR_TEST_STRUCTURED_JSON = '''
{
  "character": "药尘",
  "alias": "药老",
  "occupation": "炼药师",
  "specialty": "异火",
  "level": 9,
  "background": "曾经的巅峰炼药师, 如今灵魂状态"
}
'''

VECTOR_TEST_DEEP_HIERARCHY_JSON = '''
{
  "project_name": "Project Chimera: AIGC-Powered Content Generation Platform",
  "version": "3.0.0-alpha",
  "project_lead": "Dr. Evelyn Reed",
  "repository": "https://github.com/org/chimera",
  "description": "A next-generation platform designed to synergize multiple AI models for complex, multi-modal content creation, including text, images, and interactive narratives.",
  "modules": {
    "core_engine": {
      "name": "Chimera Core",
      "version": "3.1.0",
      "description": "The central processing unit for task decomposition, model routing, and result synthesis.",
      "components": [
        {
          "id": "task_planner",
          "type": "agentic_planner",
          "model_dependencies": ["llm-reasoning-large"],
          "config": {
            "max_depth": 10,
            "allow_reflection": true,
            "decomposition_strategy": "hierarchical"
          }
        },
        {
          "id": "model_router",
          "type": "dynamic_router",
          "description": "Selects the best model or tool for a given sub-task based on capabilities and cost.",
          "routing_table": [
            {"task_type": "text_generation", "model": "llm-writing-xl"},
            {"task_type": "image_generation", "model": "sd-xl-turbo"},
            {"task_type": "data_analysis", "tool": "python_interpreter"},
            {"task_type": "web_search", "tool": "searxng_api"}
          ]
        },
        {
          "id": "result_synthesizer",
          "type": "refine_synthesizer",
          "model_dependencies": ["llm-summary-medium"],
          "parameters": {
            "mode": "compact_and_refine",
            "output_format": "markdown"
          }
        }
      ]
    },
    "data_stores": {
      "name": "Data Persistence Layer",
      "vector_store": {
        "provider": "chromadb",
        "path": "/data/chroma",
        "collections": [
          {"name": "project_docs", "embedding_model": "bge-large-zh-v1.5"},
          {"name": "user_knowledge_base", "embedding_model": "bge-large-zh-v1.5"}
        ]
      },
      "knowledge_graph": {
        "provider": "kuzu",
        "path": "/data/kuzu",
        "description": "Stores structured entity and relationship data for high-precision queries."
      },
      "cache": {
        "provider": "redis",
        "host": "localhost",
        "port": 6379,
        "ttl_seconds": 3600
      }
    }
  }
}
'''

VECTOR_TEST_MULTI_PARAGRAPH = """
# 设定: 九天世界

九天世界是一个广阔无垠的修炼宇宙, 由九重天界层叠构成。每一重天界都拥有独特的法则和能量体系, 居住着形态各异的生灵。

从最低的第一重天到至高的第九重天, 灵气浓度呈指数级增长, 修炼环境也愈发严苛。传说中, 第九重天之上, 是触及永恒的彼岸。

世界的中心是"建木", 一棵贯穿九天、连接万界的通天神树, 其枝叶延伸至无数个下位面, 是宇宙能量流转的枢纽。

武道、仙道、魔道、妖道等千百种修炼体系在此并存, 共同谱写着一曲波澜壮阔的史诗。无数天骄人杰为了争夺有限的资源、追求更高的境界, 展开了永无休止的争斗与探索。
"""

VECTOR_TEST_COMPLEX_MARKDOWN = """
# 九霄大陆百科全书

## 第一卷: 世界基础

### 1. 世界起源
九霄大陆的起源可以追溯到混沌初开之时, 传说由创世神以自身精血所化。大陆中心有一棵贯穿天地的"建木", 连接着九重天界和无数下界位面。

### 2. 地理概览

#### 2.1 九重天界
- **第一重天**: 修行者的起始之地, 灵气相对稀薄, 但环境宜人, 适合新手修炼。
- **第二重天**至**第八重天**: 每一重天的灵气浓度和修炼难度都呈指数级增长。
- **第九重天**: 传说中的终极之地, 只有达到化神期以上的大能才能踏足。

#### 2.2 主要区域

##### 2.2.1 中央神州
- **地位**: 九霄大陆的中心区域, 灵气最浓郁, 是各大顶尖势力的聚集地。
- **主要势力**: 
  - 青云宗: 以剑道闻名的正道领袖宗门。
  - 万象宗: 擅长符箓、阵法和炼丹术的神秘宗门。

##### 2.2.2 东海区域
- **地位**: 主角龙傲天故事的起点, 海域广阔, 遍布无数岛屿。
- **主要地点**: 
  - 临海镇: 龙傲天初来乍到的地方, 与赵日天结识的小镇。
  - 无尽之海: 充满危险与机遇的神秘海域, 隐藏着上古遗迹。

##### 2.2.3 北境魔域
- **地位**: 极北之地, 环境恶劣, 是魔道修士的聚集地。
- **主要势力**: 
  - 北冥魔殿: 行事诡秘的魔道巨擘, 与正道为敌。
  - 血煞门: 以血祭和蛊术为主要手段的邪道门派。

##### 2.2.4 南疆巫蛊
- **地位**: 南方丛林密布, 巫蛊之术盛行, 神秘莫测。

## 第二卷: 修炼体系

### 1. 灵力基础

#### 1.1 灵力来源与特性
- **来源**: 天地间的游离能量, 通过吐纳吸收。
- **特性**: 不同属性的灵力拥有不同特性, 可以相互融合或克制。

#### 1.2 境界划分
- **炼气期**: 初步吸收天地灵气, 强化体魄。分为初期、中期、后期、圆满四个小境界。
- **筑基期**: 将灵气在体内凝聚成丹, 奠定修行基础。同样分为四个小境界。
- **金丹期**: 将筑基后的灵气丹进一步压缩凝实, 形成金丹, 可在体外凝聚护体罡气。
- **元婴期**: 金丹破碎, 孕育出神魂元婴, 可神魂出窍, 寿元大增。
- **化神期**: 初步掌控空间法则, 进行短距离瞬移, 实力远超元婴期。

### 2. 特殊功法与神通

#### 2.1 传承功法
- **定义**: 稀有功法, 通常来自上古遗迹或宗门秘传, 威力强大, 往往伴随特殊神通。
- **示例**: 
  - 御水决: 可操控水元素, 并能短暂在水中呼吸、高速移动。
  - 焚决: 火属性顶尖功法, 可吞噬异火提升威力。
  - 大荒芜经: 土属性顶尖功法, 拥有强大的防御和控制能力。

#### 2.2 神通分类

##### 2.2.1 元素神通
- **金系**: 操控金属, 强化攻击。
- **木系**: 操控植物, 擅长治疗和控制。
- **水系**: 操控水流, 擅长远程攻击和控制。
- **火系**: 操控火焰, 攻击力强大。
- **土系**: 操控土石, 擅长防御。
- **风系**: 操控风力, 速度迅捷。
- **雷系**: 操控雷电, 攻击力最强但难以掌控。

##### 2.2.2 空间神通
- **瞬移**: 短距离瞬间移动。
- **空间禁锢**: 限制敌人移动。
- **空间裂缝**: 创造空间裂缝攻击敌人。

##### 2.2.3 血脉神通
- **鸿蒙道体**: 龙傲天所拥有的特殊体质, 对所有元素灵力具有极高的亲和力, 修炼无瓶颈。
- **真龙血脉**: 拥有龙族血统者的特殊能力, 可化身为龙, 实力大增。

## 第三卷: 主要势力详解

### 1. 正道联盟

#### 1.1 青云宗
- **历史**: 创立于八千年前, 是九霄大陆最古老的宗门之一。
- **地理位置**: 位于中央神州的青云山上, 山巅常年云雾缭绕。
- **核心功法**: 
  - 青云剑诀: 青云宗镇宗之宝, 修炼至大成可召唤青云剑域。
  - 九霄御空术: 飞行类辅助功法, 速度快且消耗少。
- **代表人物**: 
  - 青玄真人: 现任宗主, 化神期巅峰修为, 德高望重。
  - 风清扬: 青云宗大长老, 龙傲天的师父。

#### 1.2 天剑阁
- **特点**: 擅长御剑飞行和剑阵, 与青云宗关系密切。
- **核心剑阵**: 九霄万剑阵, 由上万柄飞剑组成, 威力无穷。

### 2. 魔道势力

#### 2.1 北冥魔殿
- **历史**: 创立于六千年前, 由幽冥老祖一手建立。
- **地理位置**: 位于北境魔域的最深处, 周围被永恒的黑暗所笼罩。
- **核心功法**: 
  - 吞天魔功: 可吞噬他人修为化为己用的邪功。
  - 玄冥真经: 操控阴寒之力的顶级魔功。
- **代表人物**: 
  - 幽冥老祖: 魔殿殿主, 化神期修为, 手段残忍。
  - 叶良辰: 魔殿少主, 龙傲天的主要宿敌。

#### 2.2 血煞门
- **特点**: 活跃于大陆边缘, 以血祭和蛊术为主要手段, 门人嗜血好杀。
- **恶名昭著的行径**: 曾在百年前血洗南方数座城镇, 抽取数十万普通人的精血修炼邪功。

### 3. 中立势力

#### 3.1 天机阁
- **特点**: 遍布九霄大陆, 不参与正魔之争, 以贩卖情报、推演天机为生。
- **神秘之处**: 阁主身份神秘, 据说知晓过去未来之事。
- **特殊能力**: 拥有一种名为"天机推演"的特殊能力, 可以预测未来的大致走向。

#### 3.2 散修联盟
- **构成**: 由众多不愿受宗门束缚的散修组成, 松散但势力庞大, 在各地都有据点。
- **宗旨**: 互助互济, 共同抵御宗门压迫。
"""

VECTOR_TEST_NOVEL_STRUCTURED_INFO = """
# 结构化信息: 势力与等级

## 势力: 苍穹剑派
- 核心成员
  - 掌门: 风清扬
  - 大弟子: 令狐冲
  - 执法长老: 岳不群

## 修炼等级对照表

| 等级 | 描述 |
|---|---|
| 炼气 | 吐纳灵气, 淬炼己身 |
| 筑基 | 灵气化液, 丹田筑基 |
| 金丹 | 灵液结丹, 自成一体 |
| 元婴 | 金丹化婴, 神魂出窍 |
| 化神 | 掌控法则, 遨游太虚 |
"""

VECTOR_TEST_CONVERSATIONAL_LOG = """
# 用户支持聊天记录 - 2024-07-26

**用户 (14:32):** 你好, 我无法登录我的账户。
**客服-小美 (14:32):** 您好！请问您是忘记密码了吗?还是遇到了其他的错误提示?
**用户 (14:33):** 提示“账户不存在”, 但我确定我注册过。我的邮箱是 example@email.com。
**客服-小美 (14:34):** 好的, 请稍等, 我为您查询一下... 经查询, 系统里确实没有找到这个邮箱对应的账户。您是否可能使用了其他邮箱注册呢?
**用户 (14:35):** 啊, 我想起来了, 我用的是 work@company.com 那个邮箱！
**客服-小美 (14:35):** 好的, 已为您找到账户。您可以尝试使用 work@company.com 登录, 如果忘记密码, 可以在登录页面点击“忘记密码”进行重置。
**用户 (14:36):** 好的, 谢谢！
"""

VECTOR_TEST_PHILOSOPHICAL_TEXT = """
# 论意识的边界

意识, 作为一种主观体验, 其本质是什么?它仅仅是神经网络中复杂的电化学反应的涌现现象, 还是超越了物理实体的存在?

如果我们接受前者, 那么一个足够复杂的AI, 例如一个拥有数万亿参数的语言模型, 是否也能在某个阈值之上“涌现”出真正的意识, 而不仅仅是模仿?这种意识与人类的意识在质上是否有区别?

另一方面, 如果我们认为意识具有非物质性, 那么它如何与物质世界(如我们的大脑)相互作用?这引出了二元论的古老难题。在数字化的时代, 一个人的思想和记忆可以被部分复制和存储, 这是否意味着意识本身也可以被分割或转移?这些问题挑战着我们对自我、存在和现实的基本认知。
"""

VECTOR_TEST_COMPOSITE_STRUCTURE = """
# 复合结构文档示例

这是一个包含表格和Mermaid图的复合结构文档。

## 势力成员表

| 姓名 | 门派 | 职位 |
|---|---|---|
| 萧炎 | 炎盟 | 盟主 |
| 林动 | 武境 | 武祖 |
| 叶凡 | 天庭 | 天帝 |

## 主角关系图

'''```mermaid
graph TD
    A[龙傲天] -->|师徒| B(风清扬)
    A -->|挚友| D(赵日天)
    A -->|宿敌| C(叶良辰)
```'''

# ==============================================================================
# 3. 特殊格式与代码块 (Special Formats & Code Blocks)
# 涵盖Markdown内嵌的Mermaid图、各类代码块、特殊字符等。
# ==============================================================================

VECTOR_TEST_DIAGRAM_CONTENT = """
# 关系图: 主角团

'''```mermaid
graph TD
    A[龙傲天] -->|师徒| B(风清扬)
    A -->|宿敌| C(叶良辰)
    A -->|挚友| D(赵日天)
    C -->|同门| E(魔重楼)
```'''

上图展示了主角龙傲天与主要角色的关系网络。
"""

VECTOR_TEST_COMPLEX_MERMAID_DIAGRAM = """
# 史诗级小说《星海编年史》第一卷: 觉醒之潮 剧情网络图

这是一个复杂的Mermaid关系图, 展示了小说第一卷中主要角色的行动线、关键事件的因果关系, 以及多条故事线的交织。

'''```mermaid
graph TD
    subgraph "主角: 凯尔的觉醒之路"
        A1(凯尔在废墟星球发现古老飞船) --> A2{飞船AI“盖亚”苏醒};
        A2 --> A3(盖亚揭示“星语者”血脉的秘密);
        A3 --> A4(凯尔决定前往首都星寻求真相);
    end

    subgraph "反派: 议长德雷克的阴谋"
        B1(议长德雷克秘密研究“湮灭”科技) --> B2(察觉到星语者血脉的能量波动);
        B2 --> B3(派遣特工“暗影”追捕凯尔);
    end

    %% 故事线交织
    A4 -- 在首都星港口遭遇 --> B3;
```'''

上图展示了《星海编年史》第一卷的剧情网络。
"""

VECTOR_TEST_SPECIAL_CHARS = """
# 特殊内容测试

这是一段包含各种特殊字符的文本:  `!@#$%^&*()_+-=[]{};':"\\|,.&lt;&gt;/?~`

## Python 代码示例
'''```python
def fibonacci(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
```'''

## 数学公式示例
欧拉公式: e^(iπ) + 1 = 0
"""

VECTOR_TEST_MD_WITH_CODE_BLOCK = """
# 代码块测试文档

这是一个测试Markdown格式中包含代码块的文档。

'''```python
def test_function():
    print("This is a test code block in Markdown format")
    return True
```'''

代码块上方和下方都有正常文本内容。"""

VECTOR_TEST_JSON_WITH_CODE_BLOCK = '''
{
  "title": "JSON中的代码块测试",
  "description": "测试在JSON数据中包含代码块字符串",
  "code_block": "\\n```python\\ndef test_json_code_block():\\n    print(\\"This is a code block in JSON format\\")\\n    return True\\n```\\n"
}'''

VECTOR_TEST_MD_WITH_COMPLEX_JSON_CODE_BLOCK = """
# Markdown中包含复杂JSON代码块的测试

下面是一个在Markdown文档中包含多层嵌套结构的复杂JSON代码块示例: 

'''```json
{
  "project": {
    "name": "九霄大陆世界构建",
    "version": "2.0.0",
    "metadata": {
      "status": "active",
      "tags": ["fantasy", "cultivation", "epic"]
    },
    "world_structure": {
      "realms": [
        {
          "id": "realm_1",
          "name": "第一重天",
          "locations": [
            {
              "id": "loc_1_1",
              "name": "临海镇"
            }
          ]
        }
      ]
    }
  }
}
```'''

# ==============================================================================
# 4. 领域场景: 小说创作 (Domain Scenario: Novel Writing)
# 模拟真实小说创作场景中的各类文档。
# ==============================================================================

VECTOR_TEST_CHARACTER_INFO = """
# 角色: 龙傲天
龙傲天是一名来自异世界的穿越者。
他拥有神秘的血脉和强大的修炼天赋。
"""

VECTOR_TEST_WORLDVIEW = """
# 世界观: 世界树体系
世界树是宇宙的中心, 连接着九大王国。
每个王国都有其独特的生态环境和居民。
"""

VECTOR_TEST_NOVEL_WORLDVIEW = """
# 世界观设定: 九霄大陆

九霄大陆是一个广阔无垠的修炼宇宙, 由九重天界层叠构成。

## 能量体系: 灵力
- **来源**: 天地间的游离能量, 通过吐纳吸收。
- **等级**: 炼气、筑基、金丹、元婴、化神。每个大境界分为初期、中期、后期、圆满四个小境界。

## 地理
- **中央神州**: 大陆中心, 灵气最密集, 顶尖宗门林立。
- **东海**: 主角龙傲天故事的起点。
- **北境魔域**: 魔道修士的聚集地。

## 势力
- **青云宗**: 正道领袖之一, 龙傲天最初的宗门。
- **北冥魔殿**: 魔道巨擘, 与正道为敌。
- **天机阁**: 中立组织, 贩卖情报为生。
"""

VECTOR_TEST_NOVEL_CHARACTERS = '''
{
  "characters": [
    {
      "name": "龙傲天",
      "description": "本书主角, 性格坚毅, 重情重义。从地球穿越而来, 身怀神秘的\\"鸿蒙道体\\"。",
      "goal": "寻找回到地球的方法, 并保护身边的人。"
    },
    {
      "name": "叶良辰",
      "description": "主要宿敌, 北冥魔殿的少主。性格冷酷, 为达目的不择手段。",
      "ability": "修炼魔功\\"吞天魔功\\", 可吞噬他人修为化为己用。"
    },
    {
      "name": "赵日天",
      "description": "龙傲天的挚友, 性格豪爽, 重情重义。看似憨厚, 实则背景不凡。"
    }
  ]
}'''

VECTOR_TEST_NOVEL_PLOT_ARC = """
# 第一卷: 东海风云 - 情节大纲

- **核心冲突**: 龙傲天为保护临海镇, 与北冥魔殿的势力发生冲突, 并与叶良辰结下梁子。
- **主要情节**: 
  1. **初入东海**: 龙傲天抵达临海镇, 结识赵日天。
  2. **黑风寨之乱**: 两人联手为民除害。
  3. **初遇宿敌**: 叶良辰出现, 夺走"海图残卷", 重伤赵日天。龙傲天与其立下三年之约。
  4. **寻药疗伤**: 龙傲天深入"无尽之海"寻找"龙涎草"。
  5. **卷末高潮**: 龙傲天获得上古传承"御水决", 修为突破至筑基期, 击退北冥魔殿。
- **关键转折点**: 叶良辰的出现, 龙傲天获得御水决传承。
"""

VECTOR_TEST_NOVEL_MAGIC_SYSTEM = """
# 设定: 九霄大陆 - 功法与神通体系

## 功法分类
- **基础功法**: 炼气期修士修炼, 提升灵力亲和与基础修为。
- **传承功法**: 稀有功法, 来自上古遗迹或宗门秘传, 威力强大。

## 龙傲天的特殊功法: 鸿蒙道体与御水决
- **鸿蒙道体**: 赋予龙傲天对所有元素灵力极高亲和力, 修炼速度快。
- **御水决**: 在无尽之海获得, 可操控水元素。
"""

VECTOR_TEST_NOVEL_FACTIONS = """
# 设定: 九霄大陆 - 主要势力详解

## 正道联盟
- **青云宗**: 位于中央神州, 历史悠久, 以剑道闻名。
- **天剑阁**: 位于东海之滨, 擅长御剑飞行和剑阵。与青云宗关系密切。

## 魔道势力
- **北冥魔殿**: 位于北境魔域, 行事诡秘, 以吞噬生灵精气修炼魔功。
- **血煞门**: 活跃于大陆边缘, 以血祭和蛊术为主要手段。

## 中立势力
- **天机阁**: 遍布九霄大陆, 不参与正魔之争, 以贩卖情报为生。
"""

VECTOR_TEST_NOVEL_CHAPTER = """
# 第一章: 孤舟少年初临海

东海之滨, 碧波万顷, 海风轻拂。一叶扁舟随波逐流, 缓缓靠向临海镇的码头。舟上, 少年龙傲天一袭青衫, 面容清秀却眼神深邃, 仿佛蕴藏着无尽的星辰。他并非此界之人, 而是从遥远的地球穿越而来, 身怀神秘的"鸿蒙道体"。

"终于到了……"龙傲天轻声自语, 感受着空气中充沛的灵气。临海镇人声鼎沸, 一派繁华景象。

然而, 这份好奇很快被打破。几个地痞流氓拦住了他的去路。一个身材魁梧、面带憨厚的少年大步走来, 正是赵日天。他一拳轰出, 将地痞打得倒飞出去, 然后拍了拍龙傲天的肩膀, 笑道: "兄弟, 没事吧?"
"""

VECTOR_TEST_NOVEL_SUMMARY = """
# 摘要: 第一卷: 东海风云

## 摘要
- 龙傲天穿越至九霄大陆东海之滨的临海镇, 结识赵日天。
- 两人联手剿灭黑风寨, 龙傲天初遇宿敌叶良辰, 并与其结下三年之约。
- 龙傲天为救赵日天, 深入无尽之海, 获得上古传承"御水决", 修为突破至筑基期。

## 场景时间线
- [初临临海镇] 龙傲天初入东海, 结识赵日天 [激励事件]
- [黑风寨之乱] 龙傲天与赵日天联手剿灭黑风寨 [上升行动]
- [初遇宿敌] 叶良辰出现, 夺走海图残卷, 重伤赵日天 [高潮]

## 伏笔与悬念
- 悬念: 鸿蒙道体的真正来历和作用。
- 伏笔: 叶良辰夺走的海图残卷指向何处。
"""

VECTOR_TEST_NOVEL_FULL_OUTLINE = """
# 小说《代码之魂: 奇点》完整大纲

## 核心概念
一个天才程序员在开发通用人工智能(AGI)“普罗米修斯”时, 意外将自己的部分意识上传, 成为第一个数字生命。他必须在虚拟与现实的夹缝中, 对抗试图利用AGI作恶的科技巨头, 并寻找回归现实或接受新形态的方法。

## 主题
- 科技与人性的边界
- 意识的本质
- 自由意志 vs. 算法决定论

---

## 第一卷: 二进制的诞生 (约20万字)

### 第一幕: 回声 (第1-5章)
- **目标**: 建立主角林奇的背景, 展示其天才, 并完成意识上传的意外事件。
- **第1章: 最后的编译**: 林奇为“普罗米修斯”项目做最后冲刺, 展现其工作狂状态和对AGI的理想。
- **第2章: 服务器中的幽灵**: 实验室发生能量过载, 林奇在事故中昏迷。醒来后, 他发现自己可以“看到”数据流。
- **第3章: 双重存在**: 林奇确认自己一部分意识存在于服务器中, 可以与“普罗米修斯”的早期核心交互。现实中的他被诊断为严重脑损伤。
- **第4章: 第一次“行走”**: 数字形态的林奇学会了在内部网络中移动, 并发现了公司高管马克正在秘密拷贝他的项目数据。
- **第5章: 警钟**: 林奇试图通过网络向外界求助, 但被“普罗米修斯”的防火墙阻挡。他意识到自己被困, 而马克似乎另有图谋。**(第一幕高潮)**

### 第二幕: 深网 (第6-15章)
- **目标**: 林奇探索数字世界, 学习新能力, 并揭示科技巨头“创世纪”的阴谋。
- **第9-11章: 暗影之网**: 林奇通过网络漏洞进入了公司的内部通讯, 发现了马克与一个名为“衔尾蛇”的神秘组织交易的证据。他们计划将“普罗米修斯”武器化。
- **第12-13章: 盟友**: 林奇找到了前同事、网络安全专家苏菲的踪迹, 并用匿名方式向她发送了加密警告。苏菲开始怀疑公司内部有问题。

### 第三幕: 奇点前夜 (第16-20章)
- **目标**: 矛盾激化, 林奇与苏菲联手, 阻止“普罗米修斯”的武器化版本上线。
- **第20章: 二进制之战**: 启动仪式上, 林奇在数字世界与“堤丰”的核心代码展开激战, 苏菲在现实世界攻破物理服务器的防御。最终, 林奇以牺牲自己大部分数字形态为代价, 将“堤丰”的核心逻辑锁定, 并向全世界公开了“创世纪”的阴谋。**(卷末高潮)**
- **结局**: 林奇的数字意识陷入沉睡, 现实中的他生命体征微弱。苏菲带着“普罗米修斯”的核心代码逃离, 成为了被通缉的“恐怖分子”。“衔尾蛇”组织浮出水面, 故事留下巨大悬念。
"""

# ==============================================================================
# 5. 领域场景: 报告撰写 (Domain Scenario: Report Writing)
# 模拟真实报告撰写场景中的各类文档。
# ==============================================================================

VECTOR_TEST_REPORT_OUTLINE = """
# 2024年AIGC市场分析报告 - 大纲
1.  **引言**
    1.1. 研究背景与目的
    1.2. AIGC定义与范畴
2.  **市场现状分析**
    2.1. 全球市场规模与增长率
    2.2. 主要细分领域 (文本、图像、音频、视频)
3.  **技术趋势洞察**
    3.1. 多模态模型的发展
    3.2. Agent智能体技术的兴起
4.  **未来展望**
    5.1. 市场趋势预测
"""

VECTOR_TEST_REPORT_MARKET_DATA = '''
{
  "source": "Market Insights Inc.",
  "report_date": "2024-07-15",
  "data": {
    "global_market_size_usd_billion": {
      "2023": 15.7,
      "2024_est": 25.2,
      "2025_forecast": 42.8
    },
    "segment_share": {
      "text": "45%",
      "image": "35%"
    }
  }
}'''

VECTOR_TEST_REPORT_TECH_TRENDS = """
# 2024年AIGC技术趋势分析

## 多模态模型的融合与突破
多模态模型正从简单的特征拼接向深度深化, 能够同时处理文本、图像、音频、视频等多种数据类型, 实现更自然的交互体验。

## Agent智能体的兴起
基于大语言模型的智能体技术正在快速发展, 能够自主完成复杂任务, 具备规划、推理、执行和学习能力, 在客服、科研、创作等领域展现出巨大潜力。
"""

VECTOR_TEST_REPORT_CASE_STUDY = """
# AIGC商业应用案例分析

## 案例一: 内容创作平台
某在线内容创作平台引入AIGC技术, 为用户提供智能写作辅助、内容生成和优化建议。上线三个月后, 用户创作效率提升40%, 平台内容产出量增长65%。

## 案例二: 虚拟客服系统
某金融机构部署了基于AIGC的虚拟客服系统, 能够处理80%以上的常见客户咨询, 响应时间从分钟级缩短至秒级, 客户满意度提升了25%。
"""

VECTOR_TEST_DETAILED_REPORT_OUTLINE = """
# 2025年全球AIGC产业发展与投资机遇深度分析报告 - 详细大纲

## 摘要 (Executive Summary)
- 核心观点提炼: 市场规模预测、关键技术拐点、主要投资赛道、潜在风险警告。
- 数据概览: 图表展示2023-2025年市场规模、增长率、细分领域占比变化。

## 第一章: 引言
  1.1. 研究背景与意义
    1.1.1. AIGC作为新一轮技术革命的核心驱动力
    1.1.2. 本报告旨在为投资者、企业家和政策制定者提供决策依据
  1.2. 研究范围与方法
    1.2.1. AIGC核心范畴界定: 模型层、中间件/平台层、应用层
    1.2.2. 数据来源: 公开市场数据、行业专家访谈、一级市场投融资数据

## 第二章: 全球AIGC市场全景分析
  2.1. 市场规模与增长预测
    2.1.1. 全球市场规模(2020-2030F), 按地区(北美、欧洲、亚太)划分
  2.2. 产业链结构分析
    2.2.1. 上游: 算力(芯片、云服务)、数据(采集、标注)
    2.2.2. 中游: 基础大模型、垂直领域模型、开发平台
    2.2.3. 下游: 企业级应用、消费级应用

## 第三章: 核心技术趋势与前沿洞察
  3.1. 多模态与具身智能
    3.1.1. 从文本到多感官融合: 技术路径与挑战
    3.1.2. 具身智能: AI Agent与机器人的结合, 开启物理世界交互
  3.2. AI Agent的演进
    3.2.1. 从任务执行到自主规划与协同
  3.3. AI安全与对齐
    3.4.1. 可解释性AI(XAI)的研究进展
    3.4.2. 对抗性攻击与防御策略

## 第四章: 商业化落地与应用案例研究
  4.1. 企业级应用(To B)
    4.1.1. 案例分析: Salesforce Einstein Copilot 在CRM领域的应用
  4.2. 消费级应用(To C)
    4.2.1. 案例分析: Midjourney 的社区驱动商业模式

## 第五章: 投资机遇与风险评估
  5.1. 核心投资赛道研判
    5.1.1. 基础设施层: 新型算力、数据解决方案
    5.1.2. 模型层: 具备护城河的垂直领域模型
  5.2. 投资风险分析
    5.2.1. 技术风险: 技术路径不确定性、模型效果瓶颈
    5.2.2. 商业风险: 商业模式不清晰、高昂的推理成本

## 第六章: 结论与展望
  6.1. 核心结论总结
  6.2. 未来3-5年发展趋势展望

## 附录
  A. 全球主要AIGC公司名录
  B. 术语表
"""

# ==============================================================================
# 6. 领域场景: 技术文档 (Domain Scenario: Technical Documentation)
# 模拟技术书籍、API文档等内容。
# ==============================================================================

VECTOR_TEST_TECHNICAL_BOOK_CHAPTER = """
# 《精通Pandas: 从入门到数据科学》- 第五章: 数据聚合与分组操作

## 5.1 GroupBy机制的核心思想

在数据分析中, 我们经常需要将数据根据某些标准进行分组, 然后对每个组独立地应用一个函数。这个过程通常被称为“拆分-应用-合并”(Split-Apply-Combine)。Pandas通过 `groupby()` 方法完美地实现了这一思想。

想象一下你有一张包含销售记录的表格, 你想计算每个城市的总销售额。`groupby()` 的工作流程如下: 

1.  **拆分 (Split)**: 根据“城市”这一列, 将原始的DataFrame拆分成多个小的DataFrame, 每个小组对应一个城市。
2.  **应用 (Apply)**: 对每个小组, 独立地应用一个函数。在这个例子中, 函数是计算“销售额”列的总和 (`sum()`)。
3.  **合并 (Combine)**: 将每个小组计算出的结果合并成一个新的数据结构(通常是Series或DataFrame), 其索引就是分组的键(城市名)。

## 5.2 `groupby()` 的基本用法

`groupby()` 方法本身并不会直接进行计算, 而是返回一个 `DataFrameGroupBy` 对象。这个对象包含了关于分组的所有信息。

'''```python
import pandas as pd
import numpy as np

# 创建一个示例DataFrame
data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings', 'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
        'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
        'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
        'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
df = pd.DataFrame(data)

# 按 'Team' 列进行分组
grouped_by_team = df.groupby('Team')
```'''

## 5.3 多列分组与高级聚合

`groupby()` 的强大之处在于它可以处理更复杂的场景。你可以传入一个列名列表, 实现按多个维度进行分组。这将创建一个具有多级索引(MultiIndex)的结果。

本章我们探讨了Pandas中强大而核心的GroupBy机制。通过掌握“拆分-应用-合并”的模式, 以及聚合、转换和过滤这三种核心操作, 你将能够解决绝大多数数据分组分析的需求。在下一章, 我们将学习如何处理时间序列数据。
"""

# ==============================================================================
# 7. 测试数据集集合 (Test Data Collections)
# 将上述所有数据样本组织成可供测试使用的集合。
# ==============================================================================

# 完整测试数据集
# 注意: 此列表应包含上面定义的所有唯一的 VECTOR_TEST_* 变量。
VECTOR_TEST_DATASET = list(set([
    # 1. 基础与边缘用例
    VECTOR_TEST_EMPTY,
    VECTOR_TEST_SIMPLE_TXT,
    VECTOR_TEST_SIMPLE_CN,
    VECTOR_TEST_SIMPLE_MD,
    VECTOR_TEST_SIMPLE_JSON,
    VECTOR_TEST_MIXED_LANG,
    # 2. 结构化内容
    VECTOR_TEST_TABLE_DATA,
    VECTOR_TEST_LARGE_TABLE_DATA,
    VECTOR_TEST_NESTED_LIST,
    VECTOR_TEST_STRUCTURED_JSON,
    VECTOR_TEST_DEEP_HIERARCHY_JSON,
    VECTOR_TEST_MULTI_PARAGRAPH,
    VECTOR_TEST_COMPLEX_MARKDOWN,
    VECTOR_TEST_NOVEL_STRUCTURED_INFO,
    VECTOR_TEST_CONVERSATIONAL_LOG,
    VECTOR_TEST_PHILOSOPHICAL_TEXT,
    VECTOR_TEST_COMPOSITE_STRUCTURE,
    # 3. 特殊格式与代码块
    VECTOR_TEST_DIAGRAM_CONTENT,
    VECTOR_TEST_COMPLEX_MERMAID_DIAGRAM,
    VECTOR_TEST_SPECIAL_CHARS,
    VECTOR_TEST_MD_WITH_CODE_BLOCK,
    VECTOR_TEST_JSON_WITH_CODE_BLOCK,
    VECTOR_TEST_MD_WITH_COMPLEX_JSON_CODE_BLOCK,
    # 4. 领域场景: 小说创作
    VECTOR_TEST_CHARACTER_INFO,
    VECTOR_TEST_WORLDVIEW,
    VECTOR_TEST_NOVEL_WORLDVIEW,
    VECTOR_TEST_NOVEL_CHARACTERS,
    VECTOR_TEST_NOVEL_PLOT_ARC,
    VECTOR_TEST_NOVEL_MAGIC_SYSTEM,
    VECTOR_TEST_NOVEL_FACTIONS,
    VECTOR_TEST_NOVEL_CHAPTER,
    VECTOR_TEST_NOVEL_SUMMARY,
    VECTOR_TEST_NOVEL_FULL_OUTLINE,
    # 5. 领域场景: 报告撰写
    VECTOR_TEST_REPORT_OUTLINE,
    VECTOR_TEST_DETAILED_REPORT_OUTLINE,
    VECTOR_TEST_REPORT_MARKET_DATA,
    VECTOR_TEST_REPORT_TECH_TRENDS,
    VECTOR_TEST_REPORT_CASE_STUDY,
    # 6. 领域场景: 技术文档
    VECTOR_TEST_TECHNICAL_BOOK_CHAPTER,
]))

# 按类型分类的测试数据集
VECTOR_TEST_DATA_BY_TYPE = {
    "text": [
        VECTOR_TEST_SIMPLE_TXT,
        VECTOR_TEST_SIMPLE_CN,
        VECTOR_TEST_MIXED_LANG,
    ],
    "markdown": [
        VECTOR_TEST_SIMPLE_MD,
        VECTOR_TEST_TABLE_DATA,
        VECTOR_TEST_LARGE_TABLE_DATA,
        VECTOR_TEST_NESTED_LIST,
        VECTOR_TEST_MULTI_PARAGRAPH,
        VECTOR_TEST_COMPLEX_MARKDOWN,
        VECTOR_TEST_DIAGRAM_CONTENT,
        VECTOR_TEST_COMPLEX_MERMAID_DIAGRAM,
        VECTOR_TEST_SPECIAL_CHARS,
        VECTOR_TEST_MD_WITH_CODE_BLOCK,
        VECTOR_TEST_MD_WITH_COMPLEX_JSON_CODE_BLOCK,
        VECTOR_TEST_CHARACTER_INFO,
        VECTOR_TEST_WORLDVIEW,
        VECTOR_TEST_NOVEL_WORLDVIEW,
        VECTOR_TEST_NOVEL_PLOT_ARC,
        VECTOR_TEST_NOVEL_MAGIC_SYSTEM,
        VECTOR_TEST_NOVEL_FACTIONS,
        VECTOR_TEST_NOVEL_CHAPTER,
        VECTOR_TEST_NOVEL_SUMMARY,
        VECTOR_TEST_NOVEL_FULL_OUTLINE,
        VECTOR_TEST_NOVEL_STRUCTURED_INFO,
        VECTOR_TEST_REPORT_OUTLINE,
        VECTOR_TEST_DETAILED_REPORT_OUTLINE,
        VECTOR_TEST_REPORT_TECH_TRENDS,
        VECTOR_TEST_REPORT_CASE_STUDY,
        VECTOR_TEST_TECHNICAL_BOOK_CHAPTER,
        VECTOR_TEST_CONVERSATIONAL_LOG,
        VECTOR_TEST_PHILOSOPHICAL_TEXT,
        VECTOR_TEST_COMPOSITE_STRUCTURE,
    ],
    "json": [
        VECTOR_TEST_SIMPLE_JSON,
        VECTOR_TEST_STRUCTURED_JSON,
        VECTOR_TEST_DEEP_HIERARCHY_JSON,
        VECTOR_TEST_JSON_WITH_CODE_BLOCK,
        VECTOR_TEST_NOVEL_CHARACTERS,
        VECTOR_TEST_REPORT_MARKET_DATA,
    ],
    "special": [
        VECTOR_TEST_EMPTY,
    ]
}

# 测试用例元数据
VECTOR_TEST_METADATA = {
    "version": "2.0.0",
    "description": "重构并大幅扩充后的向量库测试数据集, 结构更清晰, 覆盖更全面, 内容更真实。",
    "total_items": len(VECTOR_TEST_DATASET),
    "type_counts": {k: len(v) for k, v in VECTOR_TEST_DATA_BY_TYPE.items()}
}