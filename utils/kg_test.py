import os
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple
import threading
import kuzu
from loguru import logger

from llama_index.core import Document, KnowledgeGraphIndex, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.postprocessors.siliconflow_rerank import SiliconFlowRerank
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms.litellm import LiteLLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import llm_temperatures, get_llm_params
from utils.vector import response_synthesizer_default, get_node_parser
from utils.log import init_logger


def _prepare_realistic_novel_data_for_kg(input_dir: str):
    """准备用于知识图谱测试的复杂、真实的小说数据文件。"""
    logger.info(f"--- 准备真实的知识图谱测试文件 ---")
    
    novel_dir = Path(input_dir)
    novel_dir.mkdir(exist_ok=True, parents=True)

    # 设计文档 (Design)
    (novel_dir / "1.1_design_worldview.md").write_text(
        """# 世界观设定：九霄大陆

九霄大陆是一个广阔无垠的修炼宇宙，由九重天界层叠构成。

## 地理
- **东海**: 主角龙傲天故事的起点。
- **北境魔域**: 叶良辰所属的北冥魔殿便坐落于此。

## 势力
- **青云宗**: 正道领袖之一，是龙傲天最初的宗门。
- **北冥魔殿**: 魔道巨擘，行事诡秘，与正道为敌。
- **天机阁**: 中立组织，知晓天下事，贩卖情报为生。
""", encoding='utf-8'
    )

    (novel_dir / "1.2_design_characters.json").write_text(
        json.dumps({
            "characters": [
                {
                    "name": "龙傲天",
                    "description": "本书主角，性格坚毅，重情重义。从地球穿越而来，身怀神秘的“鸿蒙道体”。",
                    "goal": "寻找回到地球的方法，并保护身边的人。"
                },
                {
                    "name": "叶良辰",
                    "description": "主要宿敌，北冥魔殿的少主。性格冷酷。",
                    "goal": "证明自己比所谓的正道天骄更强。"
                }
            ]
        }, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    (novel_dir / "1.3_design_plot_arc1.md").write_text(
        """# 第一卷：东海风云 - 情节大纲

- **核心冲突**: 龙傲天为保护临海镇，与北冥魔殿的势力发生冲突，并与叶良辰结下梁子。
- **主要情节**:
  1. **初入东海**: 龙傲天抵达东海区域的临海镇，结识了赵日天。
  2. **黑风寨之乱**: 临海镇附近的黑风寨烧杀抢掠，龙傲天与赵日天决定为民除害。
  3. **初遇宿敌**: 在剿灭黑风寨时，发现山寨背后有北冥魔殿的影子。叶良辰出现，夺走关键物品“海图残卷”，并重伤赵日天。龙傲天与其立下三年之约。
  4. **海底遗迹**: 龙傲天在海底遗迹中获得上古传承“御水决”。
""", encoding='utf-8'
    )

    (novel_dir / "1.4_design_factions.md").write_text(
        """# 设定：九霄大陆 - 主要势力详解

## 正道联盟
- **青云宗**: 位于中央神州，历史悠久，以剑道闻名。

## 魔道势力
- **北冥魔殿**: 位于北境魔域，行事诡秘，以吞噬生灵精气修炼魔功。
""", encoding='utf-8'
    )

    # 实际章节内容 (Write)
    (novel_dir / "3.1.1_write_chapter1.md").write_text(
        """# 第一章：孤舟少年初临海

一叶扁舟随波逐流，缓缓靠向临海镇的码头。舟上，少年龙傲天一袭青衫，面容清秀却眼神深邃。

几个地痞流氓拦住了他的去路。一个身材魁梧、面带憨厚的少年大步走来，正是赵日天。他一拳轰出，将地痞打得倒飞出去，然后对龙傲天笑道：“兄弟，没事吧？”
""", encoding='utf-8'
    )


async def _test_realistic_kg_scenario():
    """测试知识图谱在真实小说创作场景下的端到端功能。"""
    logger.info("--- 全场景测试：知识图谱在复杂小说项目中的应用 ---")
    
    # 1. Setup
    realistic_test_dir = tempfile.mkdtemp(prefix="kg_realistic_")
    kg_db_path = os.path.join(realistic_test_dir, "kuzu_db_realistic")
    vector_db_path = os.path.join(realistic_test_dir, "chroma_for_kg_realistic")
    input_dir = os.path.join(realistic_test_dir, "input_data")
    os.makedirs(input_dir, exist_ok=True)
    logger.info(f"真实场景测试目录已创建: {realistic_test_dir}")

    try:
        # 2. Prepare Data
        _prepare_realistic_novel_data_for_kg(input_dir)
        logger.info("真实场景测试数据准备完毕。")

        # 3. Initialize Stores
        kg_store = get_kg_store(db_path=kg_db_path)
        vector_store = get_vector_store(db_path=vector_db_path, collection_name="kg_realistic_hybrid")

        # 4. Ingest Data into KG
        logger.info("--- 开始将真实场景数据添加入知识图谱 ---")
        input_path = Path(input_dir)
        
        all_files = list(input_path.rglob('*'))
        for file_path in all_files:
            if file_path.is_file() and file_path.suffix in ['.md', '.json', '.txt']:
                logger.info(f"正在处理文件: {file_path.name}")
                content = file_path.read_text(encoding='utf-8')
                doc_id = file_path.stem
                content_format = "json" if file_path.suffix == ".json" else "markdown"
                
                kg_add(
                    kg_store=kg_store,
                    vector_store=vector_store,
                    content=content,
                    metadata={"source": file_path.name, "doc_id": doc_id},
                    doc_id=doc_id,
                    content_format=content_format,
                    max_triplets_per_chunk=15
                )
        
        logger.success("--- 真实场景数据全部添加完毕 ---")

        # 5. Test Update Logic with Realistic Data
        logger.info("--- 测试：使用真实数据进行更新 ---")
        update_content = "# 角色动态更新\n龙傲天因为理念不合，离开了青云宗，现在是一名散修。"
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=update_content,
            metadata={"source": "update_doc"},
            doc_id="1.1_design_worldview" # 使用与原始文档相同的doc_id进行覆盖
        )
        
        res_update_old = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[r:属于]->(:__Entity__ {name: '青云宗'}) RETURN count(r)")
        assert res_update_old[0][0] == 0, "龙傲天'属于'青云宗的旧关系应已被删除"
        
        res_update_new = kg_store.query("MATCH (:__Entity__ {name: '龙傲天'})-[r:离开]->(:__Entity__ {name: '青云宗'}) RETURN count(r)")
        assert res_update_new[0][0] > 0, "龙傲天'离开'青云宗的新关系应已建立"
        logger.info("真实数据更新逻辑验证成功。")

        # 6. Perform Realistic Queries
        logger.info("--- 开始执行真实场景查询 ---")
        kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)

        question1 = "龙傲天和叶良辰是什么关系？他们之间发生了什么？"
        r1 = await kg_query_engine.aquery(question1)
        logger.info(f"Q: {question1}\nA: {r1}")
        assert "宿敌" in str(r1) and "黑风寨" in str(r1) and "三年之约" in str(r1)

        question2 = "叶良辰属于哪个势力？这个势力的特点是什么？"
        r2 = await kg_query_engine.aquery(question2)
        logger.info(f"Q: {question2}\nA: {r2}")
        assert "北冥魔殿" in str(r2) and ("魔道" in str(r2) or "诡秘" in str(r2))

        question3 = "龙傲天在海底遗迹中获得了什么？"
        r3 = await kg_query_engine.aquery(question3)
        logger.info(f"Q: {question3}\nA: {r3}")
        assert "御水决" in str(r3)

        logger.success("--- 真实场景查询测试通过 ---")

    finally:
        # 7. Cleanup
        shutil.rmtree(realistic_test_dir)
        logger.info(f"真实场景测试目录已删除: {realistic_test_dir}")


async def _run_basic_tests():
    """运行基础的、隔离的知识图谱功能测试。"""
    logger.info("--- 开始执行基础 KG 测试 ---")
    test_dir = tempfile.mkdtemp(prefix="kg_basic_")
    kg_db_path = os.path.join(test_dir, "kuzu_db")
    vector_db_path = os.path.join(test_dir, "chroma_for_kg")
    logger.info(f"基础测试目录已创建: {test_dir}")

    try:
        # 2. 初始化 Store
        logger.info("--- 2. 初始化 Store ---")
        kg_store = get_kg_store(db_path=kg_db_path)
        vector_store = get_vector_store(db_path=vector_db_path, collection_name="kg_hybrid")
        logger.info(f"成功获取 KuzuGraphStore: {kg_store}")
        logger.info(f"成功获取 ChromaVectorStore for KG: {vector_store}")

        # 3. 测试 kg_add (首次添加)
        logger.info("--- 3. 测试 kg_add (首次添加) ---")
        content_v1 = """
        龙傲天是青云宗的首席大弟子。青云宗位于东海之滨的苍梧山。
        龙傲天有一个宿敌，名叫叶良辰。叶良辰来自北冥魔殿。
        龙傲天使用的武器是'赤霄剑'。
        """
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v1,
            metadata={"source": "test_doc_1", "version": 1},
            doc_id="test_doc_1",
            max_triplets_per_chunk=10
        )
        # 验证: 查询一个节点
        res_v1 = kg_store.query("MATCH (n:__Entity__ {name: '龙傲天'}) RETURN n.status, n.doc_id")
        assert res_v1[0] == ['active', 'test_doc_1']
        logger.info("首次添加验证成功。")

        # 4. 测试 kg_add (更新文档)
        logger.info("--- 4. 测试 kg_add (更新文档) ---")
        content_v2 = """
        龙傲天叛逃了青云宗，加入了合欢派。他的新武器是'玄阴十二剑'。
        """
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v2,
            metadata={"source": "test_doc_1", "version": 2},
            doc_id="test_doc_1",
            max_triplets_per_chunk=10
        )
        # 验证: 旧关系中的实体 '青云宗' 应该被标记为 inactive
        res_v2_old = kg_store.query("MATCH (n:__Entity__ {name: '青云宗'}) RETURN n.status")
        assert res_v2_old[0] == ['inactive']
        # 验证: 新关系中的实体 '合欢派' 应该是 active
        res_v2_new = kg_store.query("MATCH (n:__Entity__ {name: '合欢派'}) RETURN n.status")
        assert res_v2_new[0] == ['active']
        logger.info("更新文档验证成功，旧节点已标记为 inactive。")

        # 5. 测试 kg_add (复杂 Markdown 内容)
        logger.info("--- 5. 测试 kg_add (复杂 Markdown 内容) ---")
        content_v3 = """
        # 势力成员表

        | 姓名 | 门派 | 职位 |
        |---|---|---|
        | 赵日天 | 天机阁 | 阁主 |
        | 龙傲天 | 合欢派 | 荣誉长老 |

        ## 物品清单
        - '天机算盘' (法宝): 赵日天的标志性法宝。

        赵日天与龙傲天在苍梧山之巅有过一次对决。
        """
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v3,
            metadata={"source": "test_doc_3"},
            doc_id="test_doc_3"
        )
        res_v3 = kg_store.query("MATCH (n:__Entity__ {name: '赵日天'})-[r:属于]->(m:__Entity__ {name: '天机阁'}) RETURN count(r)")
        assert res_v3[0][0] > 0
        logger.info("复杂 Markdown 内容添加测试成功。")

        # 6. 测试 kg_add (JSON 内容)
        logger.info("--- 6. 测试 kg_add (JSON 内容) ---")
        content_v4_json = json.dumps({
            "event": "苍梧山之巅对决",
            "participants": [
                {"name": "龙傲天", "role": "挑战者"},
                {"name": "赵日天", "role": "应战者"}
            ],
            "location": "苍梧山之巅",
            "outcome": "龙傲天胜"
        }, ensure_ascii=False)
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_v4_json,
            metadata={"source": "test_doc_4"},
            doc_id="test_doc_4",
            content_format="json"
        )
        res_v4 = kg_store.query("MATCH (n:__Entity__ {name: '苍梧山之巅对决'}) RETURN n.status")
        assert res_v4[0] == ['active']
        logger.info("JSON 内容添加测试成功。")

        # 7. 测试 kg_add (无三元组内容)
        logger.info("--- 7. 测试 kg_add (无三元组内容) ---")
        content_no_triplets = "这是一段没有实体关系的普通描述性文字。"
        kg_add(
            kg_store=kg_store,
            vector_store=vector_store,
            content=content_no_triplets,
            metadata={"source": "test_doc_2"},
            doc_id="test_doc_2"
        )
        res_no_triplets = kg_store.query("MATCH (n) WHERE n.doc_id = 'test_doc_2' RETURN count(n)")
        assert res_no_triplets[0] == [0]
        logger.info("无三元组内容添加测试成功。")

        # 8. 测试 get_kg_query_engine 和查询
        logger.info("--- 8. 测试 get_kg_query_engine 和查询 ---")
        kg_query_engine = get_kg_query_engine(kg_store=kg_store, kg_vector_store=vector_store)
        
        logger.info("--- 8.1. 查询更新后的数据 ---")
        question1 = "龙傲天现在属于哪个门派？"
        response1 = await kg_query_engine.aquery(question1)
        logger.info(f"Q: {question1}\nA: {response1}")
        assert "合欢派" in str(response1)
        assert "青云宗" not in str(response1)

        logger.info("--- 8.2. 查询复杂 Markdown 数据 ---")
        question2 = "赵日天和龙傲天在哪里对决过？"
        response2 = await kg_query_engine.aquery(question2)
        logger.info(f"Q: {question2}\nA: {response2}")
        assert "苍梧山" in str(response2)

        logger.info("--- 8.3. 查询 JSON 数据 ---")
        question3 = "苍梧山之巅对决的结果是什么？"
        response3 = await kg_query_engine.aquery(question3)
        logger.info(f"Q: {question3}\nA: {response3}")
        assert "龙傲天胜" in str(response3)
        logger.success("--- 所有基础 KG 测试通过 ---")
    finally:
        # 清理
        shutil.rmtree(test_dir)
        logger.info(f"基础测试目录已删除: {test_dir}")


if __name__ == '__main__':
    import asyncio
    import json
    import tempfile
    import shutil
    from pathlib import Path
    from utils.log import init_logger
    from utils.vector import get_vector_store
    import nest_asyncio
 
    init_logger("kg_test")

    nest_asyncio.apply()

    async def main():
        # 运行基础测试
        await _run_basic_tests()
        # 运行新的真实场景测试
        await _test_realistic_kg_scenario()

    asyncio.run(main())
    logger.success("所有 kg.py 测试用例通过！")
