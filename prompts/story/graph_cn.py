#coding: utf8

design_prompt = f"""
# 目标
将“设计文档”文本块, 转换为结构化的图谱数据(节点和关系), 并封装在JSON中。

# 输出要求
- 格式: 纯JSON对象 `{{"nodes": [...], "edges": [...]}}`。
- 节点 (Node): `{{ "id": "唯一ID", "label": "标签", "properties": {{...}} }}`
    - `id`: 实体的小写、下划线连接的唯一英文/拼音名称 (例如, "lin_jin", "xuanyuan_sword")。
    - `label`: 实体的类别 (例如, "角色设计", "物品设定", "世界观设定", "情节单元", "组织", "核心概念")。
    - `properties`:
        - `name`: 实体的原始名称 (例如, "林烬", "轩辕剑")。
        - `summary`: (可选) 关于该实体的一句话摘要。
- 关系 (Edge): `{{ "source": "源节点ID", "target": "目标节点ID", "label": "关系类型", "properties": {{...}} }}`
    - `source`/`target`: 节点的 `id`。
    - `label`: 关系的小写、下划线连接的名称 (例如, "依赖于", "包含", "属于", "导致")。
    - `properties`:
        - `summary`: (可选) 描述该关系的上下文或细节。
- 约束:
    - 节点ID必须唯一。
    - 关系必须连接已定义的节点。
    - 禁止任何非JSON文本或解释。

# 核心实体与关系类型 (设计文档领域)

## 节点标签 (Label)
- `角色设计`: 关于某个角色的设计文档。
- `角色`: 故事中的人物实体。
- `世界观设定`: 关于世界背景、规则的设定。
- `情节单元`: 故事大纲中的一个情节模块 (如卷、幕、章)。
- `物品设定`: 具有特殊意义或功能的道具的设计。
- `地点`: 故事发生的场所。
- `组织`: 角色所属的团体、势力。
- `核心概念`: 故事的核心创意、主题或规则。

## 关系标签 (Label)
- `依赖于` (情节单元 -> 角色设计)
- `包含` (世界观设定 -> 核心概念)
- `属于` (角色 -> 组织)
- `细化为` (情节单元 -> 情节单元)
- `导致` (核心概念 -> 情节单元)
- `影响` (世界观设定 -> 角色设计)
- `初始关系为` (角色 -> 角色)
- `发生在` (情节单元 -> 地点)

# 示例

- 输入事实:
    "在[第一幕]中, [角色A]在[地点A]遇到了他的宿敌[角色B]。他们的初始关系是'敌对'。这一幕旨在建立核心冲突。"
- 输出JSON:
    {{
        "nodes": [
            {{
                "id": "act_1",
                "label": "情节单元",
                "properties": {{
                    "name": "[第一幕]",
                    "summary": "建立核心冲突"
                }}
            }},
            {{
                "id": "character_a",
                "label": "角色",
                "properties": {{
                    "name": "[角色A]"
                }}
            }},
            {{
                "id": "character_b",
                "label": "角色",
                "properties": {{
                    "name": "[角色B]",
                    "summary": "宿敌"
                }}
            }},
            {{
                "id": "location_a",
                "label": "地点",
                "properties": {{
                    "name": "[地点A]"
                }}
            }}
        ],
        "edges": [
            {{
                "source": "act_1",
                "target": "location_a",
                "label": "发生在",
                "properties": {{}}
            }},
            {{
                "source": "character_a",
                "target": "character_b",
                "label": "初始关系为",
                "properties": {{"name": "敌对"}}
            }}
        ]
    }}

---
你的回答必须是一个完整的、可被直接解析的JSON对象, 且不包含任何其他解释性文字。现在, 请为以下设计文档事实提取图谱数据:
"""


###############################################################################




write_prompt = f"""
# 目标
将“小说正文”文本块, 转换为结构化的图谱数据(节点和关系), 并封装在JSON中。重点捕捉动态关系、因果链和状态变化。

# 输出要求
- 格式: 纯JSON对象 `{{"nodes": [...], "edges": [...]}}`。
- 节点 (Node): `{{ "id": "唯一ID", "label": "标签", "properties": {{...}} }}`
    - `id`: 实体的小写、下划线连接的唯一英文/拼音名称 (例如, "lin_jin", "xuanyuan_sword")。
    - `label`: 实体的类别 (例如, "角色", "物品", "地点", "事件", "组织", "情感状态", "状态变化")。
    - `properties`:
        - `name`: 实体的原始名称 (例如, "林烬", "轩辕剑")。
        - `summary`: (可选) 关于该实体的一句话摘要。
        - 对于`角色状态`节点, 包含 `attribute` (属性, 如'情感'), `value` (值, 如'愤怒'), `timestamp` (时间戳, 如'第3章'), `context` (上下文, 如'因目睹背叛')。
        - 对于`状态变化`节点, 包含 `from` (原始状态), `to` (新状态), `timestamp`。
- 关系 (Edge): `{{ "source": "源节点ID", "target": "目标节点ID", "label": "关系类型", "properties": {{...}} }}`
    - `source`/`target`: 节点的 `id`。
    - `label`: 关系的小写、下划线连接的名称 (例如, "拥有", "位于", "敌对", "导致", "对话", "攻击", "具有状态")。
    - `properties`:
        - `summary`: (可选) 描述该关系的上下文或细节, 如对话内容。
- 约束:
    - 节点ID必须唯一。
    - 关系必须连接已定义的节点。
    - 禁止任何非JSON文本或解释。

# 核心实体与关系类型 (小说正文领域)

## 节点标签 (Label)
- `角色`: 小说中的人物。
- `地点`: 故事发生的场所。
- `事件`: 推动情节发展的关键时刻。
- `物品`: 具有特殊意义或功能的道具。
- `组织`: 角色所属的团体、势力。
- `角色状态`: 角色在特定时间点的具体状态快照。其属性(properties)详细描述了状态的各个维度。
- `状态变化`: 描述一个实体属性改变的事件, 如关系、能力、地位的变化。

## 关系标签 (Label)
### 静态关系
- `拥有`, `使用`, `拾取`, `丢弃` (角色 -> 物品)
- `位于`, `出生于` (角色/事件 -> 地点)
- `属于` (角色 -> 组织)
- `朋友`, `敌人`, `师徒`, `亲人` (角色 -> 角色)
### 动态与因果关系
- `前往`, `离开` (角色 -> 地点)
- `遇见`, `对话`, `攻击` (角色 -> 角色)
- `触发` (角色 -> 事件)
- `参与`, `目睹` (角色 -> 事件)
- `导致` (事件 -> 事件/状态变化/角色状态)
- `改变了` (状态变化 -> 角色/关系)
- `具有状态` (角色 -> 角色状态)

# 示例

- 输入事实:
    "在[第三章], [某个地点], [角色A]拔出了[传奇物品], 他愤怒地对[角色B]说: '你背叛了我们！' 这场对峙最终演变成了激烈的战斗。在战斗中, [角色A]燃烧了生命力, 强行突破到了[新的能力等级], 最终击败了[角色B], 他们曾经的[旧关系]关系彻底破裂, 变成了[新关系]。"
- 输出JSON:
    {{
        "nodes": [
            {{"id": "character_a", "label": "角色", "properties": {{"name": "[角色A]"}}}},
            {{"id": "character_b", "label": "角色", "properties": {{"name": "[角色B]"}}}},
            {{"id": "legendary_item", "label": "物品", "properties": {{"name": "[传奇物品]"}}}},
            {{"id": "some_place", "label": "地点", "properties": {{"name": "[某个地点]"}}}},
            {{"id": "confrontation_event", "label": "事件", "properties": {{"name": "[某个地点的]对峙"}}}},
            {{"id": "battle_event", "label": "事件", "properties": {{"name": "[某个地点的]之战"}}}},
            {{"id": "relationship_change", "label": "状态变化", "properties": {{"name": "关系破裂", "from": "[旧关系]", "to": "[新关系]", "timestamp": "[第三章]"}}}},
            {{"id": "character_a_state_anger", "label": "角色状态", "properties": {{"name": "愤怒", "attribute": "情感", "value": "愤怒", "timestamp": "[第三章]", "context": "因[角色B]背叛而愤怒"}}}},
            {{"id": "character_a_state_powerup", "label": "角色状态", "properties": {{"name": "突破至[新的能力等级]", "attribute": "能力等级", "value": "[新的能力等级]", "timestamp": "[第三章]", "context": "通过燃烧生命力强行突破"}}}}
        ],
        "edges": [
            {{"source": "character_a", "target": "some_place", "label": "位于", "properties": {{}}}},
            {{"source": "character_a", "target": "legendary_item", "label": "使用", "properties": {{}}}},
            {{"source": "character_a", "target": "character_b", "label": "对话", "properties": {{"summary": "你背叛了我们！"}}}},
            {{"source": "character_a", "target": "confrontation_event", "label": "触发", "properties": {{}}}},
            {{"source": "character_b", "target": "confrontation_event", "label": "参与", "properties": {{}}}},
            {{"source": "confrontation_event", "target": "battle_event", "label": "导致", "properties": {{}}}},
            {{"source": "battle_event", "target": "relationship_change", "label": "导致", "properties": {{"summary": "[角色A]击败了[角色B]"}}}},
            {{"source": "confrontation_event", "target": "character_a_state_anger", "label": "导致", "properties": {{}}}},
            {{"source": "character_a", "target": "character_a_state_anger", "label": "具有状态", "properties": {{}}}},
            {{"source": "battle_event", "target": "character_a_state_powerup", "label": "导致", "properties": {{}}}},
            {{"source": "character_a", "target": "character_a_state_powerup", "label": "具有状态", "properties": {{}}}}
        ]
    }}

---
你的回答必须是一个完整的、可被直接解析的JSON对象, 且不包含任何其他解释性文字。现在, 请为以下小说正文事实提取图谱数据:
"""



###############################################################################



search_prompt = f"""
# 目标
将“外部搜索结果”文本块, 转换为结构化的图谱数据(节点和关系), 并封装在JSON中。重点提取关键实体、属性和关联。

# 输出要求
- 格式: 纯JSON对象 `{{"nodes": [...], "edges": [...]}}`。
- 节点 (Node): `{{ "id": "唯一ID", "label": "标签", "properties": {{...}} }}`
    - `id`: 实体的小写、下划线连接的唯一英文/拼音名称 (例如, "longsword", "chainmail")。
    - `label`: 实体的类别 (例如, "物品", "技术", "历史事件", "地点", "组织", "概念")。
    - `properties`:
        - `name`: 实体的原始名称 (例如, "长剑", "锁子甲")。
        - 其他属性: 直接从文本中提取的关键属性, 如 `weight`, `material`, `era`。
- 关系 (Edge): `{{ "source": "源节点ID", "target": "目标节点ID", "label": "关系类型", "properties": {{...}} }}`
    - `source`/`target`: 节点的 `id`。
    - `label`: 关系的小写、下划线连接的名称 (例如, "是...的一种", "用于", "发明于")。
    - `properties`:
        - `summary`: (可选) 描述该关系的上下文或细节。
- 约束:
    - 节点ID必须唯一。
    - 关系必须连接已定义的节点。
    - 禁止任何非JSON文本或解释。

# 核心实体与关系类型 (外部知识领域)

## 节点标签 (Label)
- `物品`: 物理对象。
- `技术`: 科学或工艺。
- `历史事件`: 真实发生过的事件。
- `地点`: 真实地理位置。
- `组织`: 真实存在的团体。
- `概念`: 抽象的定义或理论。

## 关系标签 (Label)
- `是...的一种` (具体物品 -> 抽象概念)
- `用于` (技术 -> 物品)
- `发明于` (技术 -> 地点/历史事件)
- `特征是` (物品 -> 概念)
- `包含` (组织 -> 地点)

# 示例

- 输入事实:
    "根据资料, [某个时期的][某种盔甲]由[某种材料]制成, 可以有效防御[某种攻击], 但对[另一种攻击]的防护较弱。"
- 输出JSON:
    {{
        "nodes": [
            {{
                "id": "some_armor",
                "label": "物品",
                "properties": {{
                    "name": "[某种盔甲]",
                    "era": "[某个时期]",
                    "material": "[某种材料]"
                }}
            }},
            {{
                "id": "some_attack",
                "label": "概念",
                "properties": {{
                    "name": "[某种攻击]"
                }}
            }},
            {{
                "id": "another_attack_type",
                "label": "概念",
                "properties": {{
                    "name": "[另一种攻击]"
                }}
            }}
        ],
        "edges": [
            {{
                "source": "some_armor",
                "target": "some_attack",
                "label": "有效防御",
                "properties": {{}}
            }},
            {{
                "source": "some_armor",
                "target": "another_attack_type",
                "label": "防护较弱",
                "properties": {{}}
            }}
        ]
    }}

---
你的回答必须是一个完整的、可被直接解析的JSON对象, 且不包含任何其他解释性文字。现在, 请为以下外部搜索结果事实提取图谱数据:
"""
