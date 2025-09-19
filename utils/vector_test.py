import os
import sys
import numpy as np
import asyncio
from pathlib import Path
import json
from loguru import logger

from llama_index.core import Document, Settings
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.vector_stores.types import VectorStore
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vector import clear_vector_index_cache, default_file_metadata, get_vector_query_engine, get_vector_store, index_query, index_query_batch, vector_add, vector_add_from_dir


def _realistic_file_metadata(file_path_str: str) -> dict:
    """
    从特定格式的文件名中解析元数据，模拟真实项目场景。
    文件名格式: {task_id}_{type}_{description}.{ext}
    例如: 1.2_design_characters.json, 3.1.1_write_chapter1.md, 4.1_search_ancient_runes.txt
    """
    file_path = Path(file_path_str)
    parts = file_path.stem.split('_', 2) # Split at most twice
    
    metadata = default_file_metadata(file_path_str) # Start with default metadata
    
    if len(parts) >= 2:
        task_id_str = parts[0]
        content_type = parts[1]
        
        metadata["task_id"] = task_id_str
        metadata["type"] = content_type
        
        # Extract chapter/arc information from task_id if applicable
        id_parts = task_id_str.split('.')
        if len(id_parts) == 3: # e.g., 3.1.1 -> chapter 1 of arc 1
            try:
                metadata["arc_num"] = int(id_parts[1])
                metadata["chapter_num"] = int(id_parts[2])
            except ValueError:
                pass # Not a numeric chapter/arc
        elif len(id_parts) == 2: # e.g., 1.1 -> design for arc 1, or chapter 1 if it's a top-level chapter
            try:
                metadata["arc_num"] = int(id_parts[0])
                # This could be a chapter or a design sub-task. Let's keep it simple for now.
            except ValueError:
                pass

        # Assign status based on type
        if content_type in ['design', 'write', 'summary']:
            metadata['status'] = 'active'
        elif content_type == 'search':
            metadata['status'] = 'search_result'
        else:
            metadata['status'] = 'archived' # Default for others
            
    return metadata


async def _test_embedding_model():
    """专门测试嵌入模型的功能和正确性。"""
    logger.info("--- 3. 测试嵌入模型 (Embedding Model) ---")
    embed_model = Settings.embed_model

    # 1. 测试不同文本是否产生不同向量
    logger.info("--- 3.1. 测试不同文本的向量差异性 ---")
    text1 = "这是一个关于人工智能的句子。"
    text2 = "这是一个关于自然语言处理的句子。"
    
    try:
        embedding1_list = await embed_model.aget_text_embedding(text1)
        embedding2_list = await embed_model.aget_text_embedding(text2)
        embedding1 = np.array(embedding1_list)
        embedding2 = np.array(embedding2_list)

        logger.debug(f"文本1的向量 (前5维): {embedding1[:5]}")
        logger.debug(f"文本2的向量 (前5维): {embedding2[:5]}")

        # 检查向量是否全为零
        assert np.any(embedding1 != 0), "嵌入向量1不应为全零向量，这表明嵌入模型可能未正确工作。"
        assert np.any(embedding2 != 0), "嵌入向量2不应为全零向量，这表明嵌入模型可能未正确工作。"
        logger.info("向量非零检查通过。")

        # 检查向量是否相同
        are_equal = np.array_equal(embedding1, embedding2)
        assert not are_equal, "不同文本不应产生完全相同的嵌入向量。如果相同，说明嵌入模型存在严重问题（向量碰撞）。"
        logger.info("不同文本的向量不同，检查通过。")

        # 检查向量相似度
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        assert norm1 > 0 and norm2 > 0, "向量模长不能为零。"
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        logger.info(f"两个不同但相关句子的余弦相似度: {similarity:.4f}")
        assert 0.5 < similarity < 0.999, "相关句子的相似度应在合理范围内 (大于0.5，小于1)。"
        logger.info("相关句子相似度检查通过。")

    except Exception as e:
        logger.error(f"获取嵌入向量时出错: {e}", exc_info=True)
        assert False, "嵌入模型调用失败，请检查API密钥、网络连接或模型配置。"

    # 2. 测试相同文本是否产生相同向量
    logger.info("--- 3.2. 测试相同文本的向量一致性 ---")
    try:
        embedding1_again_list = await embed_model.aget_text_embedding(text1)
        embedding1_again = np.array(embedding1_again_list)
        # 某些嵌入模型可能存在微小的非确定性，因此我们检查余弦相似度而不是完全相等
        similarity_same = np.dot(embedding1, embedding1_again) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding1_again))
        logger.info(f"相同文本两次嵌入的余弦相似度: {similarity_same:.6f}")
        assert similarity_same > 0.999, f"相同文本的嵌入向量应该非常相似 (相似度 > 0.999)，但实际为 {similarity_same:.6f}"
        logger.info("相同文本的向量具有高相似度，检查通过。")
    except Exception as e:
        logger.error(f"测试相同文本向量时出错: {e}", exc_info=True)
        assert False, "相同文本向量一致性测试失败。"

    # 3. 测试批量嵌入
    logger.info("--- 3.3. 测试批量嵌入 ---")
    try:
        texts_batch = [text1, text2, "第三个完全不同的句子。"]
        embeddings_batch_list = await embed_model.aget_text_embedding_batch(texts_batch)
        embeddings_batch = [np.array(e) for e in embeddings_batch_list]
        assert len(embeddings_batch) == 3, f"批量嵌入应返回3个向量，但返回了 {len(embeddings_batch)} 个。"
        logger.info("批量嵌入返回了正确数量的向量。")
        # 比较批量嵌入的第一个结果与单个嵌入结果的相似度
        similarity_batch = np.dot(embeddings_batch[0], embedding1) / (np.linalg.norm(embeddings_batch[0]) * np.linalg.norm(embedding1))
        logger.info(f"批量嵌入与单个嵌入结果的余弦相似度: {similarity_batch:.6f}")
        assert similarity_batch > 0.999, f"批量嵌入的第一个结果应与单个嵌入结果非常相似 (相似度 > 0.999)，但实际为 {similarity_batch:.6f}"
        logger.info("批量嵌入的第一个结果与单个嵌入结果具有高相似度，检查通过。")
    except Exception as e:
        logger.error(f"批量嵌入测试失败: {e}", exc_info=True)
        assert False, "批量嵌入测试失败。"
    logger.success("--- 嵌入模型测试通过 ---")

async def _test_reranker():
    """专门测试重排服务的功能和正确性。"""
    logger.info("--- 测试重排服务 (Reranker) ---")

    query = "哪部作品是关于一个男孩发现自己是巫师的故事？"
    documents = [
        "《沙丘》是一部关于星际政治和巨型沙虫的史诗科幻小说。", # low relevance
        "《哈利·波特与魔法石》讲述了一个名叫哈利·波特的年轻男孩，他发现自己是一个巫师，并被霍格沃茨魔法学校录取。", # high relevance
        "《魔戒》讲述了霍比特人佛罗多·巴金斯摧毁至尊魔戒的旅程。", # medium relevance
        "《神经漫游者》是一部赛博朋克小说，探讨了人工智能和虚拟现实。", # low relevance
        "一个男孩在魔法学校学习的故事，他最好的朋友是一个红发男孩和一个聪明的女孩。", # high relevance, but less specific
    ]

    reranker = SiliconFlowRerank(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        top_n=3,
    )

    nodes = [NodeWithScore(node=Document(text=d), score=1.0) for d in documents]
    query_bundle = QueryBundle(query_str=query)

    try:
        reranked_nodes = await reranker.apostprocess_nodes(nodes, query_bundle=query_bundle)
        
        assert len(reranked_nodes) <= 3, f"重排后应返回最多 3 个节点, 但返回了 {len(reranked_nodes)} 个。"
        logger.info(f"重排后返回 {len(reranked_nodes)} 个节点，数量正确。")
        
        assert len(reranked_nodes) > 0, "Reranker 返回了空列表，服务可能未正常工作。"

        reranked_texts = [node.get_content() for node in reranked_nodes]
        reranked_scores = [node.score for node in reranked_nodes]
        logger.info(f"重排后的文档顺序及分数: {list(zip(reranked_texts, reranked_scores))}") # type: ignore

        assert "哈利·波特" in reranked_texts[0], "最相关的文档没有排在第一位。"
        logger.info("最相关的文档排序正确。")

        for i in range(len(reranked_scores) - 1):
            assert reranked_scores[i] >= reranked_scores[i+1], f"重排后分数没有递减: {reranked_scores}"
        logger.info("重排后分数递减，检查通过。")

    except Exception as e:
        logger.error(f"重排服务测试失败: {e}", exc_info=True)
        assert False, "重排服务测试失败，请检查API或配置。"

    logger.success("--- 重排服务测试通过 ---")

def _prepare_legacy_test_data(input_dir: str):
    """准备所有用于基础测试的输入文件。"""
    logger.info(f"--- 2. 准备多样化的测试文件 ---")
    # 短文本
    (Path(input_dir) / "doc1.md").write_text("# 角色：龙傲天\n龙傲天是一名来自异世界的穿越者。", encoding='utf-8')
    (Path(input_dir) / "doc2.txt").write_text("世界树是宇宙的中心，连接着九大王国。", encoding='utf-8')
    # 表格和简单列表
    (Path(input_dir) / "doc3.md").write_text(
        "# 势力成员表\n\n| 姓名 | 门派 | 职位 |\n|---|---|---|\n| 萧炎 | 炎盟 | 盟主 |\n| 林动 | 武境 | 武祖 |\n\n## 功法清单\n- 焚决\n- 大荒芜经",
        encoding='utf-8'
    )
    # JSON
    (Path(input_dir) / "doc4.json").write_text(
        json.dumps({"character": "药尘", "alias": "药老", "occupation": "炼药师", "specialty": "异火"}, ensure_ascii=False),
        encoding='utf-8'
    )
    # 空文件
    (Path(input_dir) / "empty.txt").write_text("", encoding='utf-8')
    # 长文本段落
    (Path(input_dir) / "long_text.md").write_text(
        "# 设定：九天世界\n\n九天世界是一个广阔无垠的修炼宇宙，由九重天界层叠构成。每一重天界都拥有独特的法则和能量体系，居住着形态各异的生灵。从最低的第一重天到至高的第九重天，灵气浓度呈指数级增长，修炼环境也愈发严苛。传说中，第九重天之上，是触及永恒的彼岸。世界的中心是“建木”，一棵贯穿九天、连接万界的通天神树，其枝叶延伸至无数个下位面，是宇宙能量流转的枢纽。武道、仙道、魔道、妖道等千百种修炼体系在此并存，共同谱写着一曲波澜壮阔的史诗。无数天骄人杰为了争夺有限的资源、追求更高的境界，展开了永无休止的争斗与探索。",
        encoding='utf-8'
    )
    # 包含Mermaid图
    (Path(input_dir) / "diagram.md").write_text(
        '# 关系图：主角团\n\n```mermaid\ngraph TD\n    A[龙傲天] -->|师徒| B(风清扬)\n    A -->|宿敌| C(叶良辰)\n    A -->|挚友| D(赵日天)\n    C -->|同门| E(魔尊重楼)\n    B -->|曾属于| F(华山剑派)\n```\n\n上图展示了主角龙傲天与主要角色的关系网络。',
        encoding='utf-8'
    )
    # 复杂嵌套列表
    (Path(input_dir) / "complex_list.md").write_text(
        "# 物品清单\n\n- **神兵利器**\n  1. 赤霄剑: 龙傲天的佩剑，削铁如泥。\n  2. 诛仙四剑: 上古遗留的杀伐至宝，分为四柄。\n     - 诛仙剑\n     - 戮仙剑\n     - 陷仙剑\n     - 绝仙剑\n- **灵丹妙药**\n  - 九转还魂丹: 可活死人，肉白骨。\n  - 菩提子: 辅助悟道，提升心境。",
        encoding='utf-8'
    )
    # 复合设计文档，模拟真实场景
    (Path(input_dir) / "composite_design_doc.md").write_text(
        """# 卷一：东海风云 - 章节设计

本卷主要围绕主角龙傲天初入江湖，在东海区域结识盟友、遭遇宿敌，并最终揭开“苍龙七宿”秘密一角的序幕。

> **创作笔记**: 本卷的重点是快节奏的奇遇和人物关系的建立，为后续更宏大的世界观铺垫。

![东海地图](./images/donghai_map.png)

## 章节大纲

### 流程图：龙傲天成长路径
```mermaid
graph LR
    A[初入江湖] --> B{遭遇危机}
    B --> C{获得奇遇}
    C --> D[实力提升]
    D --> A
```

| 章节 | 标题 | 核心事件 | 出场角色 | 关键场景/物品 | 备注 |
|---|---|---|---|---|---|
| 1.1 | 孤舟少年 | 龙傲天乘孤舟抵达临海镇，初遇赵日天。 | - **龙傲天** (主角)<br>- 赵日天 (挚友) | 临海镇码头、海鲜酒楼 | 奠定本卷轻松诙谐的基调。 |
| 1.2 | 不打不相识 | 龙傲天与赵日天因误会大打出手，结为兄弟。 | - 龙傲天<br>- 赵日天 | 镇外乱石岗 | 展示龙傲天的剑法和赵日天的拳法。 |
| 1.3 | 黑风寨之危 | 黑风寨山贼袭扰临海镇，掳走镇长之女。 | - 龙傲天<br>- 赵日天<br>- 黑风寨主 (反派) | 临海镇、黑风寨 | 引入第一个小冲突，主角团首次合作。 |
| 1.4 | 夜探黑风寨 | 龙傲天与赵日天潜入黑风寨，发现其与北冥魔殿有关。 | - 龙傲天<br>- 赵日天 | 黑风寨地牢 | 获得关键物品：**北冥令牌**。 |
| 1.5 | 决战黑风寨 | 主角团与黑风寨决战，救出人质，叶良辰首次登场。 | - 龙傲天<br>- 赵日天<br>- **叶良辰** (宿敌) | 黑风寨聚义厅 | 叶良辰以压倒性实力击败黑风寨主，带走令牌，与龙傲天结下梁子。 |

## 核心设定：苍龙七宿

“苍龙七宿”是流传于东海之上的古老传说，与七件上古神器及星辰之力有关。

- **设定细节**:
  - **东方七宿**: 角、亢、氐、房、心、尾、箕。
  - **对应神器**: 每宿对应一件神器，如“角宿”对应“苍龙角”。
  - **力量体系**:
    ```json
    {
      "system_name": "星宿之力",
      "activation": "集齐七件神器，于特定时辰在特定地点（东海之眼）举行仪式。",
      "effect": "可号令四海，引动星辰之力，拥有毁天灭地的威能。"
    }
    ```
- **剧情关联**: 北冥魔殿和主角团都在寻找这七件神器。

### 关键情节线索
- **北冥令牌**: 叶良辰从黑风寨夺走的令牌，是寻找北冥魔殿分舵的关键。
- **龙傲天的身世**: 主角的身世之谜，可能与某个隐世家族有关。
- **赵日天的背景**: 挚友赵日天看似憨厚，但其拳法路数不凡，背后或有故事。
""",
        encoding='utf-8'
    )
    # 包含特殊字符和不同语言代码块的文档
    (Path(input_dir) / "special_chars_and_code.md").write_text(
        """# 特殊内容测试

这是一段包含各种特殊字符的文本： `!@#$%^&*()_+-=[]{};':"\\|,.<>/?~`

## Python 代码示例

下面是一个 Python 函数，用于计算斐波那契数列。

```python
def fibonacci(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
```""",
        encoding='utf-8'
    )
    logger.info(f"测试文件已写入目录: {input_dir}")


def _prepare_realistic_test_data(input_dir: str): # Renamed from _prepare_realistic_test_data to avoid confusion with the previous one
    """准备更复杂、更真实的测试文件，模拟小说和报告项目。"""
    logger.info(f"--- 2. 准备真实的测试文件 ---")
    
    # --- 小说项目 ---
    novel_dir = Path(input_dir) / "novel_project"
    novel_dir.mkdir(exist_ok=True)

    # 设计文档 (Design)
    (novel_dir / "1.1_design_worldview.md").write_text(
        """# 世界观设定：九霄大陆

九霄大陆是一个广阔无垠的修炼宇宙，由九重天界层叠构成。每一重天界都拥有独特的法则和能量体系，居住着形态各异的生灵。

## 能量体系：灵力
- **来源**: 天地间的游离能量，通过吐纳吸收。
- **等级**: 炼气、筑基、金丹、元婴、化神。每个大境界分为初期、中期、后期、圆满四个小境界。
- **特性**: 金丹期修士可在体外凝聚护体罡气，元婴期可神魂出窍，化神期则能初步掌控空间法则，进行短距离瞬移。

## 地理
- **中央神州**: 大陆中心，灵气最浓郁，顶尖宗门林立。
- **东海**: 东方尽头，海域广阔，遍布无数岛屿，盛产各种天材地宝，但也暗藏凶险。主角龙傲天故事的起点。
- **北境魔域**: 极北之地，环境恶劣，魔道修士的聚集地。叶良辰所属的北冥魔殿便坐落于此。
- **南疆巫蛊**: 南方丛林密布，巫蛊之术盛行，神秘莫测。

## 势力
- **青云宗**: 正道领袖之一，位于中央神州，是龙傲天最初的宗门。
- **北冥魔殿**: 魔道巨擘，行事诡秘，与正道为敌。
- **天机阁**: 中立组织，知晓天下事，贩卖情报为生。
""", encoding='utf-8'
    )

    (novel_dir / "1.2_design_characters.json").write_text(
        json.dumps({
            "characters": [
                {
                    "name": "龙傲天",
                    "description": "本书主角，性格坚毅，重情重义。从地球穿越而来，身怀神秘的“鸿蒙道体”，修炼速度远超常人。",
                    "initial_ability": "穿越时灵魂与鸿蒙道体融合，使其对所有元素灵力具有极高的亲和力，修炼无瓶颈。初始修为在炼气期圆满。",
                    "goal": "寻找回到地球的方法，并保护身边的人。"
                },
                {
                    "name": "叶良辰",
                    "description": "主要宿敌，北冥魔殿的少主。性格冷酷，为达目的不择手段，但对其师妹柳如烟十分温柔。",
                    "ability": "修炼魔功“吞天魔功”，可吞噬他人修为化为己用。当前修为在筑基中期。",
                    "goal": "证明自己比所谓的正道天骄更强，夺取天下资源。"
                }
            ]
        }, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    (novel_dir / "1.3_design_plot_arc1.md").write_text(
        """# 第一卷：东海风云 - 情节大纲

- **核心冲突**: 龙傲天为保护临海镇，与北冥魔殿的势力发生冲突，并与叶良辰结下梁子。
- **主要情节**:
  1. **初入东海**: 龙傲天离开新手村，抵达东海区域的临海镇，结识了赵日天。
  2. **黑风寨之乱**: 临海镇附近的黑风寨烧杀抢掠，龙傲天与赵日天决定为民除害。
  3. **初遇宿敌**: 在剿灭黑风寨时，发现山寨背后有北冥魔殿的影子。叶良辰出现，以雷霆手段夺走一件关键物品“海图残卷”，并重伤赵日天。龙傲天与其立下三年之约。
  4. **寻药疗伤**: 为救治赵日天，龙傲天深入“无尽之海”寻找传说中的“龙涎草”。
  5. **卷末高潮**: 龙傲天在海底遗迹中不仅找到了龙涎草，还意外获得上古传承“御水决”，修为突破至筑基期。归来后，治好赵日天，并利用新学功法击退了北冥魔殿的又一次侵袭。
""", encoding='utf-8'
    )
    (novel_dir / "1.4_design_magic_system.md").write_text(
        """# 设定：九霄大陆 - 功法与神通体系

## 功法分类
- **基础功法**: 炼气期修士修炼，提升灵力亲和与基础修为。
- **进阶功法**: 筑基期以上修士修炼，分为攻击、防御、辅助等类型。
- **传承功法**: 稀有功法，通常来自上古遗迹或宗门秘传，威力强大，往往伴随特殊神通。

## 神通
- **定义**: 功法修炼到一定境界后领悟的特殊能力，或通过血脉觉醒、奇遇获得。
- **类型**:
  - **元素神通**: 操控金、木、水、火、土、风、雷等元素。
  - **空间神通**: 瞬移、空间禁锢、空间裂缝等。
  - **时间神通**: 减缓、加速、回溯（极难）。
  - **体修神通**: 强化肉身，如“金刚不坏体”。
  - **魂修神通**: 攻击神魂，如“摄魂术”。

## 龙傲天的特殊功法：鸿蒙道体与御水决
- **鸿蒙道体**: 赋予龙傲天对所有元素灵力极高亲和力，修炼速度快，且能兼容多种功法。
- **御水决**: 在无尽之海海底遗迹中获得，可操控水元素，并能短暂在水中呼吸、高速移动。进阶后可凝聚水龙攻击。
""", encoding='utf-8'
    )

    (novel_dir / "1.5_design_factions.md").write_text(
        """# 设定：九霄大陆 - 主要势力详解

## 正道联盟
- **青云宗**: 位于中央神州，历史悠久，以剑道闻名。宗主“青玄真人”修为深不可测。
- **天剑阁**: 位于东海之滨，擅长御剑飞行和剑阵。与青云宗关系密切。
- **万象宗**: 位于南疆，以符箓、阵法和炼丹术见长。

## 魔道势力
- **北冥魔殿**: 位于北境魔域，行事诡秘，以吞噬生灵精气修炼魔功。殿主“幽冥老祖”手段残忍。
- **血煞门**: 活跃于大陆边缘，以血祭和蛊术为主要手段，门人嗜血好杀。

## 中立势力
- **天机阁**: 遍布九霄大陆，不参与正魔之争，以贩卖情报、推演天机为生。阁主身份神秘。
- **散修联盟**: 由众多不愿受宗门束缚的散修组成，松散但势力庞大，在各地都有据点。
""", encoding='utf-8'
    )

    (novel_dir / "2.1_design_plot_arc2.md").write_text(
        """# 第二卷：天机风云 - 情节大纲

- **核心冲突**: 龙傲天为探寻鸿蒙道体之谜，前往中央神州，卷入天机阁与某个古老宗门的秘境争夺。
- **主要情节**:
  1. **中央神州之旅**: 龙傲天与赵日天告别临海镇，踏上前往中央神州的旅程。途中遭遇劫匪，展现实力。
  2. **天机阁求助**: 龙傲天为寻找身世线索，前往天机阁寻求帮助，得知鸿蒙道体与上古秘境“鸿蒙秘境”有关。
  3. **秘境开启**: 鸿蒙秘境即将开启，引来各方势力关注。龙傲天与天机阁合作，共同进入秘境。
  4. **秘境探险**: 在秘境中，龙傲天遭遇古老宗门的阻挠，并与叶良辰再次相遇，双方短暂交手。
  5. **卷末高潮**: 龙傲天在秘境深处获得鸿蒙道体的部分传承，修为再次突破，并发现一个惊天秘密。
""", encoding='utf-8'
    )

    # 实际章节内容 (Write)
    (novel_dir / "3.1.1_write_chapter1.md").write_text(
        """# 第一章：孤舟少年初临海

东海之滨，碧波万顷，海风轻拂。一叶扁舟随波逐流，缓缓靠向临海镇的码头。舟上，少年龙傲天一袭青衫，面容清秀却眼神深邃，仿佛蕴藏着无尽的星辰。他并非此界之人，而是从遥远的地球穿越而来，身怀神秘的“鸿蒙道体”。

“终于到了……”龙傲天轻声自语，感受着空气中充沛的灵气，与地球的末法时代截然不同。临海镇人声鼎沸，渔民的吆喝声、商贩的叫卖声此起彼伏，一派繁华景象。他跳下船，好奇地打量着四周，对这个全新的世界充满了探索欲。

然而，这份好奇很快被打破。几个膀大腰圆的地痞流氓拦住了他的去路，为首的疤脸大汉狞笑着：“小子，初来乍到不懂规矩？码头费交了吗？”龙傲天眉头微皱，正欲出手，一道爽朗的声音传来：“疤脸，又欺负新人？！”

一个身材魁梧、面带憨厚的少年大步走来，正是赵日天。他一拳轰出，将疤脸大汉打得倒飞出去，然后拍了拍龙傲天的肩膀，笑道：“兄弟，没事吧？这帮狗腿子，三天不打上房揭瓦！”
""", encoding='utf-8'
    )

    (novel_dir / "3.1.2_write_chapter2.md").write_text(
        """# 第二章：不打不相识，黑风寨初显

赵日天与龙傲天在海鲜酒楼把酒言欢，两人相谈甚欢，很快便引为知己。赵日天是临海镇的本地人，性情豪爽，对龙傲天这个“外来者”毫无芥蒂，反而对其神秘的来历充满了好奇。龙傲天也从赵日天口中得知了许多关于东海修真界的基本常识，以及临海镇附近的一些传闻。

“最近黑风寨那帮山贼越来越猖獗了，镇长家的千金都被他们掳走了！”赵日天愤愤不平地说道。龙傲天闻言，眼神微凝。他虽然初来乍到，但前世的侠义之心并未泯灭。

“黑风寨？在何处？”龙傲天问道。

“就在镇子北边的黑风山，那帮家伙有筑基期修士坐镇，寻常人根本奈何不了他们。”赵日天叹了口气。

龙傲天沉思片刻，决定出手。他向赵日天询问了黑风寨的详细情况，并提出两人联手。赵日天虽然有些惊讶于龙傲天的胆识，但也被其气概所感染，欣然同意。两人约定次日清晨，一同前往黑风山。
""", encoding='utf-8'
    )

    # 更多搜索结果 (Search)
    (novel_dir / "4.1_search_ancient_runes.txt").write_text(
        """关于上古符文和阵法的研究资料：
- **符文起源**: 符文是天地法则的具象化表现，最早由上古仙人从天地大道中领悟并刻画。
- **符文作用**: 刻画在器物上可增强其威能，刻画在阵基上可构成强大阵法。
- **阵法分类**:
  - **聚灵阵**: 汇聚天地灵气，加速修炼。
  - **防御阵**: 抵御外敌攻击。
  - **攻击阵**: 释放强大攻击。
  - **幻阵**: 迷惑敌人心智。
- **古老符文**: 许多上古符文已经失传，或只有少数古老宗门掌握。
""", encoding='utf-8'
    )

    # 更多摘要 (Summary)
    (novel_dir / "5.1_summary_arc1.md").write_text(
        """# 摘要: 第一卷：东海风云

## 摘要
- 龙傲天穿越至九霄大陆东海之滨的临海镇，结识赵日天。
- 两人联手剿灭黑风寨，龙傲天初遇宿敌叶良辰，并与其结下三年之约。
- 龙傲天为救赵日天，深入无尽之海，获得上古传承“御水决”，修为突破至筑基期。

## 主题与氛围
- 冒险与成长，友情与宿命。
- 轻松诙谐与紧张刺激并存。

## 场景时间线
- [初临临海镇] 龙傲天初入东海，结识赵日天 [激励事件]
    - 原因: 龙傲天穿越，赵日天解围。
    - 结果:
        - 总体影响: 龙傲天对新世界有了初步了解，获得第一个盟友。
        - 状态变更: [龙傲天:心态:谨慎->好奇], [龙傲天:目标:适应新世界->行侠仗义]
- [黑风寨之乱] 龙傲天与赵日天联手剿灭黑风寨 [上升行动]
    - 原因: 黑风寨掳走镇长千金，赵日天愤愤不平。
    - 结果:
        - 总体影响: 龙傲天展现实力，与赵日天友情加深。
        - 状态变更: [龙傲天:修为:炼气期圆满->筑基初期]
- [初遇宿敌] 叶良辰出现，夺走海图残卷，重伤赵日天 [高潮]
    - 原因: 黑风寨背后是北冥魔殿，叶良辰奉命行事。
    - 结果:
        - 总体影响: 龙傲天与叶良辰结下梁子，赵日天重伤。
        - 状态变更: [龙傲天:目标:行侠仗义->为友复仇], [关系:龙傲天-叶良辰:陌生->宿敌]
- [无尽之海寻药] 龙傲天深入无尽之海寻找龙涎草 [下降行动]
    - 原因: 赵日天重伤垂危。
    - 结果:
        - 总体影响: 龙傲天获得“御水决”传承，修为突破。
        - 状态变更: [龙傲天:修为:筑基初期->筑基中期], [龙傲天:能力:无水系神通->掌握御水决]

## 伏笔与悬念
- 悬念: 鸿蒙道体的真正来历和作用。
- 伏笔: 叶良辰夺走的海图残卷指向何处。
- 伏笔: 赵日天拳法路数不凡，背景可能不简单。

## 角色关系与冲突分析
graph TD
    龙傲天 -- 挚友 --> 赵日天
    龙傲天 -- 宿敌 --> 叶良辰
    叶良辰 -- 效忠 --> 北冥魔殿
    龙傲天 -- 敌对 --> 北冥魔殿

- 龙傲天 & 赵日天:
    - 关系变化/现状: 从陌生到生死之交。
    - 核心矛盾/性质: 无矛盾，互相信任扶持。
    - 关键事件: 初临临海镇，黑风寨之乱，无尽之海寻药。
- 龙傲天 vs 叶良辰:
    - 关系变化/现状: 从陌生到宿敌。
    - 核心矛盾/性质: 正魔对立，个人恩怨（赵日天受伤），对海图残卷的争夺。
    - 关键事件: 初遇宿敌。

## 故事地图
graph TD
    A[地球] --> B(临海镇)
    B --> C(黑风寨)
    B --> D(无尽之海)
    B --> F(中央神州)
    D --> E(海底遗迹)
    F --> G(天机阁)
    F --> H(鸿蒙秘境)

- 临海镇: 龙傲天初临之地，结识赵日天，黑风寨之乱的受害者。
- 黑风寨: 北冥魔殿在东海的据点，龙傲天与赵日天首次联手，初遇叶良辰。
- 无尽之海: 龙傲天为救赵日天深入，获得御水决传承。
- 海底遗迹: 龙傲天获得御水决传承之地。
- 中央神州: 龙傲天第二卷的冒险起点，天机阁所在地。
- 天机阁: 龙傲天寻求身世线索的地方。
- 鸿蒙秘境: 与龙傲天鸿蒙道体相关的上古秘境。

## 世界观与设定
- 九霄大陆: 广阔的修真世界，灵气充沛。
- 鸿蒙道体: 龙傲天特殊体质，修炼无瓶颈。
- 北冥魔殿: 魔道势力，在东海有据点。
- 天机阁: 中立势力，贩卖情报。

## 关键物品与概念
- 鸿蒙道体: 龙傲天特殊体质，修炼无瓶颈。
- 御水决: 龙傲天在海底遗迹中获得的功法。
- 龙涎草: 救治赵日天的灵药。
- 海图残卷: 叶良辰夺走的神秘物品，与北冥魔殿有关。
- 鸿蒙秘境: 与鸿蒙道体相关的上古秘境。
""", encoding='utf-8'
    )
    
    # 搜索文档 (Search)
    (novel_dir / "2.1_search_shipbuilding.txt").write_text(
        """关于中世纪欧洲帆船技术的研究资料：
- **船体结构**: 早期多采用“克林克”搭接法，船板部分重叠。后期“卡维尔”平接法普及，使得船体更大更坚固。龙骨是船的脊梁，提供了主要的结构强度。
- **帆装**: 三角帆（Lateen sail）的引入是革命性的，它使得船只能够逆风航行，极大地提高了航行效率和范围。相比之下，横帆在顺风时效率更高。大型船只通常混合使用两种帆。
- **导航**: 主要依靠星盘、十字测天仪和罗盘。对海岸线和季风的经验知识至关重要。
- **代表船型**: 柯克船（Cog）、卡瑞克船（Carrack）、卡拉维尔帆船（Caravel）。卡拉维尔帆船因其轻便、快速和使用了三角帆，在地理大发现时代扮演了重要角色。
""", encoding='utf-8'
    )

    # 摘要文档 (Summary)
    (novel_dir / "3.1.1_summary_chapter1.md").write_text(
        """## 第一章：孤舟少年
- **核心事件**: 主角龙傲天乘着一叶扁舟，从与世隔绝的“新手村”来到繁华的东海临海镇。
- **人物动态**: 龙傲天对外界充满好奇，但行事谨慎。在码头，他因不熟悉当地规矩，与地痞发生小冲突，被路过的赵日天解围。
- **关键信息**: 揭示了龙傲天初入尘世、实力不显但性格沉稳的特点。引出了重要配角赵日天。
""", encoding='utf-8')

    # --- 报告项目 ---
    report_dir = Path(input_dir) / "report_project"
    report_dir.mkdir(exist_ok=True)

    (report_dir / "report_1_design_outline.md").write_text(
        """# 2024年AIGC市场分析报告 - 大纲
1.  **引言**
    1.1. 研究背景与目的
    1.2. AIGC定义与范畴
2.  **市场现状分析**
    2.1. 全球市场规模与增长率
    2.2. 主要细分领域 (文本、图像、音频、视频)
    2.3. 产业链结构 (底层模型、中间层、应用层)
3.  **技术趋势洞察**
    3.1. 多模态模型的发展
    3.2. Agent智能体技术的兴起
    3.3. 开源与闭源模型生态对比
4.  **商业应用与挑战**
    4.1. 成功商业案例分析
    4.2. 面临的挑战 (成本、安全、伦理)
5.  **未来展望**
    5.1. 市场趋势预测
    5.2. 结论与建议
""", encoding='utf-8'
    )

    (report_dir / "report_2.1_search_market_data.json").write_text(
        json.dumps({
            "source": "Market Insights Inc.",
            "report_date": "2024-07-15",
            "data": {
                "global_market_size_usd_billion": {
                    "2023": 15.7,
                    "2024_est": 25.2
                },
                "growth_rate_yoy": "60.5%",
                "segment_share": {
                    "text": "45%",
                    "image": "35%",
                    "video": "15%",
                    "audio": "5%"
                }
            }
        }, indent=2),
        encoding='utf-8'
    )
    logger.info(f"真实的测试文件已写入目录: {input_dir}")


async def _test_data_ingestion(vector_store: VectorStore, input_dir: str, test_dir: str):
    """测试从目录和单个内容添加向量，包括各种边缘情况。"""
    # 4. 测试从目录添加入库
    logger.info("--- 4. 测试 vector_add_from_dir (常规) ---")
    vector_add_from_dir(vector_store, input_dir, default_file_metadata)

    # 5. 测试 vector_add (各种场景)
    logger.info("--- 5. 测试 vector_add (各种场景) ---")
    logger.info("--- 5.1. 首次添加 ---")
    vector_add(
        vector_store,
        "虚空之石是一个神秘物品。",
        {"type": "item", "source": "manual_add_1"},
        doc_id="item_void_stone"
    )

    logger.info("--- 5.2. 更新文档 ---")
    vector_add(
        vector_store,
        "虚空之石是一个极其稀有的神秘物品，据说蕴含着宇宙初开的力量。",
        {"type": "item", "source": "manual_add_2"},
        doc_id="item_void_stone"
    )

    logger.info("--- 5.3. 添加 JSON 内容 ---")
    json_content = json.dumps({"event": "双帝之战", "protagonist": ["萧炎", "魂天帝"]}, ensure_ascii=False)
    vector_add(
        vector_store,
        content=json_content,
        metadata={"type": "event", "source": "manual_json"},
        content_format="json",
        doc_id="event_doudi"
    )

    logger.info("--- 5.4. 添加空内容 (应跳过) ---")
    added_empty = vector_add(
        vector_store,
        content="  ",
        metadata={"type": "empty"},
        doc_id="empty_content"
    )
    assert not added_empty

    logger.info("--- 5.5. 添加包含错误信息的内容 (应跳过) ---")
    added_error = vector_add(
        vector_store,
        content="这是一个包含错误信息的报告: 生成报告时出错。",
        metadata={"type": "error"},
        doc_id="error_content"
    )
    assert not added_error
    logger.info("包含错误信息的内容未被添加，验证通过。")

    logger.info("--- 5.6. 添加无法解析出节点的内容 (应跳过) ---")
    added_no_nodes = vector_add(
        vector_store,
        content="---\n\n---\n",  # 仅包含 Markdown 分割线
        metadata={"type": "no_nodes"},
        doc_id="no_nodes_content"
    )
    assert not added_no_nodes
    logger.info("无法解析出节点的内容未被添加，验证通过。")

    # 6. 测试从无效目录添加
    logger.info("--- 6. 测试 vector_add_from_dir (空目录或仅含无效文件) ---")
    empty_input_dir = os.path.join(test_dir, "empty_input_data")
    os.makedirs(empty_input_dir, exist_ok=True)
    (Path(empty_input_dir) / "unsupported.log").write_text("some log data", encoding='utf-8')
    (Path(empty_input_dir) / "another_empty.txt").write_text("   ", encoding='utf-8')
    added_from_empty = vector_add_from_dir(vector_store, empty_input_dir)
    assert not added_from_empty
    logger.info("从仅包含无效文件的目录添加，返回False，验证通过。")


async def _test_node_deletion(vector_store: VectorStore):
    """测试节点的显式删除功能。"""
    logger.info("--- 7. 测试显式删除 ---")
    doc_id_to_delete = "to_be_deleted"
    content_to_delete = "这是一个唯一的、即将被删除的节点XYZ123。"
    vector_add(
        vector_store,
        content_to_delete,
        {"type": "disposable", "source": "delete_test"},
        doc_id=doc_id_to_delete
    )
    await asyncio.sleep(2)

    clear_vector_index_cache(vector_store)
    
    filters = MetadataFilters(filters=[MetadataFilter(key="ref_doc_id", value=doc_id_to_delete)])
    query_engine_for_check = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=1, rerank_top_n=0)
    response_before = await query_engine_for_check.aquery("any")
    retrieved_nodes_before = response_before.source_nodes

    assert retrieved_nodes_before and content_to_delete in retrieved_nodes_before[0].get_content()
    logger.info("删除前节点存在，验证通过。")

    vector_store.delete(ref_doc_id=doc_id_to_delete)
    logger.info("已调用删除方法。")

    clear_vector_index_cache(vector_store)

    query_engine_after_delete = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=1, rerank_top_n=0)
    response_after = await query_engine_after_delete.aquery("any")
    retrieved_nodes_after = response_after.source_nodes
    assert not retrieved_nodes_after
    logger.success("--- 节点删除测试通过 ---")


async def _test_node_update(vector_store: VectorStore):
    """测试节点的更新操作（通过覆盖doc_id）。"""
    logger.info("--- 8. 测试更新操作 ---")
    doc_id_to_update = "to_be_updated"
    content_v1 = "这是文档的初始版本 V1，用于测试更新功能。"
    content_v2 = "这是文档更新后的版本 V2，旧内容应被覆盖。"

    vector_add(
        vector_store,
        content_v1,
        {"type": "update_test", "version": 1},
        doc_id=doc_id_to_update
    )
    await asyncio.sleep(2)

    filters_update = MetadataFilters(filters=[MetadataFilter(key="ref_doc_id", value=doc_id_to_update)])
    query_engine_v1 = get_vector_query_engine(vector_store, filters=filters_update, similarity_top_k=1)
    response_v1 = await query_engine_v1.aquery("any")
    retrieved_v1 = response_v1.source_nodes
    assert retrieved_v1 and "V1" in retrieved_v1[0].get_content()
    logger.info("更新前，版本 V1 存在，验证通过。")

    vector_add(
        vector_store,
        content_v2,
        {"type": "update_test", "version": 2},
        doc_id=doc_id_to_update
    )
    await asyncio.sleep(2)

    query_engine_v2 = get_vector_query_engine(vector_store, filters=filters_update, similarity_top_k=1)
    response_v2 = await query_engine_v2.aquery("any")
    retrieved_v2 = response_v2.source_nodes
    assert retrieved_v2 and "V2" in retrieved_v2[0].get_content() and "V1" not in retrieved_v2[0].get_content()
    logger.success("--- 节点更新测试通过 ---")


async def _test_basic_queries(vector_store: VectorStore):
    """测试标准查询模式。"""
    logger.info("--- 9. 测试 get_vector_query_engine (标准模式) ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=2)
    logger.info(f"成功创建标准查询引擎: {type(query_engine)}")

    questions = [
        "龙傲天是谁？",
        "什么是虚空之石？", # 调整问题，使其更容易从上下文中获得确切答案
        "萧炎是什么门派的？",
        "药老是谁？",
        "双帝之战的主角是谁？",
        "九天世界的中心是什么？",
        "龙傲天和叶良辰是什么关系？",
        "诛仙四剑包括哪些？",
        "黑风寨发生了什么事？",
        "苍龙七宿是什么？",
        "龙傲天的成长路径是怎样的？",
        "北冥令牌有什么用？",
        "如何用python计算斐波那契数列？"
    ]
    results = await index_query_batch(query_engine, questions)
    logger.info(f"标准查询结果:\n{results}")
    assert any("龙傲天" in r for r in results)
    assert any("虚空之石" in r for r in results)
    assert any("炎盟" in r for r in results) # 放宽断言，因为模型可能只回答“炎盟”
    assert any("药尘" in r for r in results)
    assert any("萧炎" in r and "魂天帝" in r for r in results)
    assert any("建木" in r for r in results)
    assert any("宿敌" in r for r in results)
    assert any("戮仙剑" in r and "绝仙剑" in r for r in results)
    assert any("黑风寨" in r and "北冥魔殿" in r for r in results)
    assert any("苍龙七宿" in r and "星宿之力" in r for r in results)
    assert any("初入江湖" in r and "实力提升" in r for r in results)
    assert any("北冥魔殿分舵" in r for r in results)
    assert any("fibonacci" in r and "def" in r for r in results)
    assert not any("错误信息" in r for r in results)
    # assert not any("即将被删除" in r for r in results) # This assertion is for _test_node_deletion, not _test_basic_queries. It should be removed or moved.
    logger.success("--- 标准查询测试通过 ---")


async def _test_filtered_query(vector_store: VectorStore):
    """测试带固定元数据过滤器的查询。"""
    logger.info("--- 10. 测试 get_vector_query_engine (带固定过滤器) ---")
    filters = MetadataFilters(filters=[MetadataFilter(key="type", value="item")])
    query_engine_filtered = get_vector_query_engine(vector_store, filters=filters)
    
    results_hit = await index_query_batch(query_engine_filtered, ["介绍一下那个石头。"])
    logger.info(f"带过滤器的查询结果 (应命中):\n{results_hit}")
    assert len(results_hit) > 0 and "虚空之石" in results_hit[0]

    results_miss = await index_query_batch(query_engine_filtered, ["龙傲天是谁？"])
    logger.info(f"被过滤器阻挡的查询结果 (应未命中):\n{results_miss}")
    assert not results_miss[0]
    logger.success("--- 带固定过滤器的查询测试通过 ---")


async def _test_no_reranker_sync_query(vector_store: VectorStore):
    """测试无重排器和同步查询模式。"""
    logger.info("--- 11. 测试无重排器和同步查询 ---")
    query_engine_no_rerank = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=0)
    sync_question = "林动的功法是什么？"
    sync_response = query_engine_no_rerank.query(sync_question)
    logger.info(f"同步查询 (无重排器) 结果:\n{sync_response}")
    assert "大荒芜经" in str(sync_response)
    logger.success("--- 无重排器和同步查询测试通过 ---")


async def _test_auto_retriever_query(vector_store: VectorStore):
    """测试自动检索（AutoRetriever）模式。"""
    logger.info("--- 12. 测试 get_vector_query_engine (自动检索模式) ---")
    query_engine_auto = get_vector_query_engine(vector_store, use_auto_retriever=True, similarity_top_k=5, rerank_top_n=2)
    logger.info(f"成功创建自动检索查询引擎: {type(query_engine_auto)}")

    auto_question = "请根据类型为 'item' 的文档，介绍一下那个物品。"
    auto_results = await index_query_batch(query_engine_auto, [auto_question])
    logger.info(f"自动检索查询结果:\n{auto_results}")
    assert len(auto_results) > 0 and "虚空之石" in auto_results[0]
    logger.success("--- 自动检索查询测试通过 ---")


async def _test_empty_query(vector_store: VectorStore):
    """测试对无结果查询的处理。"""
    logger.info("--- 13. 测试空查询 ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=2)
    empty_results = await index_query_batch(query_engine, ["一个不存在的概念xyz"])
    logger.info(f"空查询结果: {empty_results}")
    assert not empty_results[0]
    logger.success("--- 空查询测试通过 ---")


async def _test_synthesis_from_multiple_nodes(vector_store: VectorStore):
    """测试需要整合多个节点信息才能回答的查询。"""
    logger.info("--- 测试：整合多节点信息 ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=3)
    question = "在九霄大陆的世界观下，主角龙傲天的初始能力是什么？"
    result = await index_query(query_engine, question)
    logger.info(f"多节点整合查询结果:\n{result}")
    assert "龙傲天" in result and "鸿蒙道体" in result and "炼气期" in result and "九霄大陆" in result
    logger.success("--- 整合多节点信息测试通过 ---")


async def _test_complex_filtering_nin_operator(vector_store: VectorStore):
    """测试 'nin' (not in) 元数据过滤器，模拟排除已处理任务的场景。"""
    logger.info("--- 测试：复杂过滤器 (nin) ---")
    # 模拟场景：正在处理任务1.4，需要查询之前的设计，但要排除任务1.3（假设它正在被修改）
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="task_id", value=["1.3"], operator="nin")
        ]
    )
    query_engine = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=5, rerank_top_n=3)
    question = "请提供关于故事背景和角色的设计信息。"
    result = await index_query(query_engine, question)
    logger.info(f"'nin' 过滤器查询结果:\n{result}")
    assert "九霄大陆" in result and "龙傲天" in result  # 应该能找到 1.1 和 1.2
    assert "东海风云" not in result and "海图残卷" not in result # 不应该找到 1.3 的内容
    logger.success("--- 复杂过滤器 (nin) 测试通过 ---")


async def _test_complex_filtering_and_operator(vector_store: VectorStore):
    """测试多个过滤条件的组合 (AND 逻辑)。"""
    logger.info("--- 测试：复杂过滤器 (AND) ---")
    # 查找所有类型为 'design' 且状态为 'active' 的文档
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="type", value="design"),
            MetadataFilter(key="status", value="active"),
        ],
        condition="and"
    )
    query_engine = get_vector_query_engine(vector_store, filters=filters, similarity_top_k=10, rerank_top_n=5)
    question = "列出所有活跃的设计文档内容。"
    result = await index_query(query_engine, question)
    logger.info(f"'AND' 过滤器查询结果:\n{result}")
    # 应该找到所有 design 文件
    assert "九霄大陆" in result
    assert "龙傲天" in result
    assert "东海风云" in result
    assert "市场分析报告" in result
    # 不应该找到 search 或 summary 的内容
    assert "中世纪欧洲帆船" not in result
    assert "孤舟少年" not in result
    logger.success("--- 复杂过滤器 (AND) 测试通过 ---")


async def _test_retrieval_from_long_document(vector_store: VectorStore):
    """测试从长文档中准确抽取出特定细节的能力。"""
    logger.info("--- 测试：从长文档中检索细节 ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=3, rerank_top_n=1)
    question = "天机阁是做什么的？"
    result = await index_query(query_engine, question)
    logger.info(f"长文档细节检索结果:\n{result}")
    assert "中立组织" in result and "贩卖情报" in result
    logger.success("--- 从长文档中检索细节测试通过 ---")


async def _test_retrieval_across_formats(vector_store: VectorStore):
    """测试需要结合 Markdown 和 JSON 文件内容才能回答的查询。"""
    logger.info("--- 测试：跨格式检索 (MD + JSON) ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=5, rerank_top_n=3)
    question = "根据2024年的市场分析报告大纲和已有的市场数据，AIGC市场的现状如何？"
    result = await index_query(query_engine, question)
    logger.info(f"跨格式检索结果:\n{result}")
    assert "市场规模" in result and "25.2" in result  # 来自 JSON
    assert "细分领域" in result and "产业链" in result # 来自 MD
    assert "文本" in result and "45%" in result # 来自 JSON
    logger.success("--- 跨格式检索 (MD + JSON) 测试通过 ---")


async def _test_realistic_novel_queries(vector_store: VectorStore):
    """测试针对复杂小说项目的真实查询场景。"""
    logger.info("--- 测试：真实小说项目查询场景 ---")
    query_engine = get_vector_query_engine(vector_store, similarity_top_k=10, rerank_top_n=5)

    # 1. 续写时的问题：关于当前情节的细节、角色状态
    logger.info("--- 1. 续写时的问题 ---")
    q1 = "龙傲天在击退北冥魔殿侵袭后，他的修为达到了什么境界？获得了哪些新的能力？"
    r1 = await index_query(query_engine, q1)
    logger.info(f"Q: {q1}\nA: {r1}")
    assert "筑基期" in r1 and "御水决" in r1

    q2 = "赵日天被重伤后，龙傲天是如何救治他的？"
    r2 = await index_query(query_engine, q2)
    logger.info(f"Q: {q2}\nA: {r2}")
    assert "龙涎草" in r2

    # 2. 下一卷/章设计时的问题：关于未来情节、世界观扩展
    logger.info("--- 2. 下一卷/章设计时的问题 ---")
    q3 = "第二卷的核心冲突和主要角色有哪些？"
    r3 = await index_query(query_engine, q3)
    logger.info(f"Q: {q3}\nA: {r3}")
    assert "天机阁" in r3 and "古老宗门" in r3 and "秘境探险" in r3

    q4 = "九霄大陆的灵力等级体系是怎样的？金丹期修士有什么特点？"
    r4 = await index_query(query_engine, q4)
    logger.info(f"Q: {q4}\nA: {r4}")
    assert "炼气、筑基、金丹、元婴、化神" in r4 and "护体罡气" in r4

    q5 = "除了青云宗和北冥魔殿，九霄大陆还有哪些主要势力？"
    r5 = await index_query(query_engine, q5)
    logger.info(f"Q: {q5}\nA: {r5}")
    assert "天机阁" in r5 and "南疆巫蛊" in r5

    # 3. 角色关系与过往历史情节
    logger.info("--- 3. 角色关系与过往历史情节 ---")
    q6 = "龙傲天和叶良辰的三年之约是如何结下的？"
    r6 = await index_query(query_engine, q6)
    logger.info(f"Q: {q6}\nA: {r6}")
    assert "黑风寨" in r6 and "海图残卷" in r6

    q7 = "叶良辰的性格特点和主要目标是什么？"
    r7 = await index_query(query_engine, q7)
    logger.info(f"Q: {q7}\nA: {r7}")
    assert "冷酷" in r7 and "吞天魔功" in r7 and "证明自己" in r7

    q8 = "龙傲天在临海镇都经历了哪些事件？"
    r8 = await index_query(query_engine, q8)
    logger.info(f"Q: {q8}\nA: {r8}")
    assert "初入东海" in r8 and "黑风寨之乱" in r8 and "初遇宿敌" in r8

    # 4. 细节查询：关于特定设定、物品
    logger.info("--- 4. 细节查询 ---")
    q9 = "龙傲天获得的'鸿蒙道体'有什么特殊之处？"
    r9 = await index_query(query_engine, q9)
    logger.info(f"Q: {q9}\nA: {r9}")
    assert "极高的亲和力" in r9 and "修炼无瓶颈" in r9

    q10 = "古代帆船的三角帆有什么作用？"
    r10 = await index_query(query_engine, q10)
    logger.info(f"Q: {q10}\nA: {r10}")
    assert "逆风航行" in r10

    q11 = "北冥魔殿在东海区域的势力布局是怎样的？"
    r11 = await index_query(query_engine, q11)
    logger.info(f"Q: {q11}\nA: {r11}")
    assert "黑风寨" in r11 and "海图残卷" in r11

    # 5. 跨文档类型整合查询 (例如，结合设计和正文)
    logger.info("--- 5. 跨文档类型整合查询 ---")
    q12 = "龙傲天在第一章中初入临海镇时，他的心情和所见所闻是怎样的？"
    r12 = await index_query(query_engine, q12)
    logger.info(f"Q: {q12}\nA: {r12}")
    assert "好奇" in r12 and "繁华" in r12 and "码头" in r12

    q13 = "请总结龙傲天在第一卷中的主要经历和成长。"
    r13 = await index_query(query_engine, q13)
    logger.info(f"Q: {q13}\nA: {r13}")
    assert "初入东海" in r13 and "黑风寨" in r13 and "筑基期" in r13

    # 6. 带有元数据过滤的查询 (例如，只查询设计文档)
    logger.info("--- 6. 带有元数据过滤的查询 ---")
    design_filters = MetadataFilters(filters=[MetadataFilter(key="type", value="design")])
    design_query_engine = get_vector_query_engine(vector_store, filters=design_filters, similarity_top_k=5)
    q14 = "请描述九霄大陆的地理构成。"
    r14 = await index_query(design_query_engine, q14)
    logger.info(f"Q: {q14}\nA: {r14}")
    assert "中央神州" in r14 and "东海" in r14 and "北境魔域" in r14

    # 7. 带有元数据过滤的查询 (例如，只查询特定章节的摘要)
    logger.info("--- 7. 带有元数据过滤的查询 (特定章节摘要) ---")
    summary_filters = MetadataFilters(filters=[MetadataFilter(key="type", value="summary"), MetadataFilter(key="chapter_num", value=1)])
    summary_query_engine = get_vector_query_engine(vector_store, filters=summary_filters, similarity_top_k=5)
    q15 = "第一章的摘要是什么？"
    r15 = await index_query(summary_query_engine, q15)
    logger.info(f"Q: {q15}\nA: {r15}")
    assert "龙傲天" in r15 and "临海镇" in r15 and "赵日天" in r15

    logger.success("--- 真实小说项目查询场景测试通过 ---")


if __name__ == '__main__':
    import asyncio
    import shutil
    from pathlib import Path
    import json
    from utils.log import init_logger
    from utils.file import project_root
    import nest_asyncio

    init_logger("vector_test")

    nest_asyncio.apply()

    import logging
    logging.getLogger("litellm").setLevel(logging.WARNING)

    test_dir = project_root / ".test" / "vector_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    # --- 为旧测试用例创建单独的目录 ---
    legacy_test_dir = test_dir / "legacy"
    legacy_db_path = str(legacy_test_dir / "chroma_db")
    legacy_input_dir = str(legacy_test_dir / "input_data")
    os.makedirs(legacy_input_dir, exist_ok=True)

    # --- 为新测试用例创建单独的目录 ---
    realistic_test_dir = test_dir / "realistic"
    realistic_db_path = str(realistic_test_dir / "chroma_db")
    realistic_input_dir = str(realistic_test_dir / "input_data")
    os.makedirs(realistic_input_dir, exist_ok=True)

    async def main():
        # --- 运行旧的、基础的测试用例 ---
        logger.info("--- 开始执行基础向量库测试 ---")
        _prepare_legacy_test_data(legacy_input_dir)
        
        await _test_embedding_model()
        await _test_reranker()

        legacy_vector_store = get_vector_store(db_path=legacy_db_path, collection_name="legacy_collection") # Changed collection name
        await _test_data_ingestion(legacy_vector_store, legacy_input_dir, str(legacy_test_dir))
        await _test_node_deletion(legacy_vector_store)
        await _test_node_update(legacy_vector_store)
        await _test_basic_queries(legacy_vector_store) # Renamed
        await _test_filtered_query(legacy_vector_store)
        await _test_no_reranker_sync_query(legacy_vector_store)
        await _test_auto_retriever_query(legacy_vector_store)
        await _test_empty_query(legacy_vector_store)
        logger.success("--- 所有基础向量库测试通过 ---")

        # --- 运行新的、复杂的测试用例 ---
        logger.info("\n\n--- 开始执行复杂与真实场景测试 ---")
        _prepare_realistic_test_data(realistic_input_dir)
        
        realistic_vector_store = get_vector_store(db_path=realistic_db_path, collection_name="realistic_collection")
        
        # 使用新的元数据提取函数进行入库
        vector_add_from_dir(realistic_vector_store, realistic_input_dir, _realistic_file_metadata)
        await _test_synthesis_from_multiple_nodes(realistic_vector_store)
        await _test_retrieval_across_formats(realistic_vector_store)
        await _test_realistic_novel_queries(realistic_vector_store) # Added new test function
        logger.success("--- 所有复杂与真实场景测试通过 ---")

    try:
        asyncio.run(main())
        logger.success("所有 vector.py 测试用例通过！")
    finally:
        # 清理
        if test_dir.exists():
            shutil.rmtree(test_dir)
        logger.info(f"测试目录已删除: {test_dir}")
