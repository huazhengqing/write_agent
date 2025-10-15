<template>
  <div class="create-book-view">
    <div class="page-header">
      <h1>创建新书项目</h1>
      <div class="header-buttons">
      <el-button
        type="success"
        @click="handleGenerateIdea"
        :loading="isGeneratingIdea"
      >AI 生成创意</el-button>
      <el-button
        type="primary"
        @click="handleCreateBook"
        :loading="isCreating"
        style="margin-left: 12px;"
      >创建新书项目</el-button>
      </div>

    </div>
    <div class="form-container">
      <el-form :model="newBookForm" label-position="top" ref="newBookFormRef">
        <!-- 第一行: 书名/项目名, 类别, 语言 -->
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="书名/项目名" prop="name" :rules="{ required: true, message: '书名不能为空', trigger: 'blur' }">
              <el-input v-model="newBookForm.name" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="类别" prop="category">
              <el-select v-model="newBookForm.category" placeholder="请选择项目类别" style="width: 100%;">
                <el-option label="故事/小说 (Story)" value="story"></el-option>
                <el-option label="书籍 (Book)" value="book"></el-option>
                <el-option label="报告 (Report)" value="report"></el-option>
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="语言" prop="language">
              <el-select v-model="newBookForm.language" placeholder="请选择语言" style="width: 100%;">
                <el-option label="中文 (cn)" value="cn"></el-option>
                <el-option label="英文 (en)" value="en"></el-option>
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <!-- 第二行: 预估总字数, 每日字数目标 -->
        <el-row :gutter="20">
          <el-col :span="12">
             <el-form-item label="预估总字数" prop="length">
              <el-input v-model="newBookForm.length" placeholder="例如: 100万字"/>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="每日字数目标" prop="day_wordcount_goal">
              <el-input-number v-model="newBookForm.day_wordcount_goal" :min="0" :step="1000" style="width: 100%;" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="核心目标" prop="goal" :rules="{ required: true, message: '核心目标不能为空', trigger: 'blur' }">
          <el-input v-model="newBookForm.goal" type="textarea" :rows="3" />
        </el-form-item>
        <el-form-item label="具体指令 (Instructions)">
          <el-input v-model="newBookForm.instructions" type="textarea" autosize />
        </el-form-item>
        <el-form-item label="输入简报 (Input Brief)">
            <el-input v-model="newBookForm.input_brief" type="textarea" autosize />
        </el-form-item>
        <el-form-item label="约束条件 (Constraints)">
            <el-input v-model="newBookForm.constraints" type="textarea" autosize />
        </el-form-item>
        <el-form-item label="验收标准 (Acceptance Criteria)">
            <el-input v-model="newBookForm.acceptance_criteria" type="textarea" autosize />
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue';
import { useRouter } from 'vue-router';
import { useBookStore } from '@/stores/bookStore';
import { ElMessage, ElNotification } from 'element-plus';
import type { FormInstance } from 'element-plus';
import type { BookCreate } from '@/api/books';

const router = useRouter();
const bookStore = useBookStore();

const newBookFormRef = ref<FormInstance>();
const newBookForm = reactive<BookCreate>({
  name: '我靠美食修炼，竟成了国运之神',
  goal: '在一个美食能觉醒‘食灵’、影响国运的现代都市，一个背负家族衰落秘密的青年厨师，意外激活了能‘解析万物’的神秘系统。他从一道家传菜开始，通过烹饪极致美味，唤醒沉睡的食灵，积累美食气运，兑换超凡能力。他不仅要重振家族荣耀，还要在各国美食世家、神秘组织和国家力量的博弈中，揭开导致美食文明断代的历史真相，最终以美食之道，守护国运，登顶世界之巅。',
  category: 'story',
  language: 'cn',
  length: '100万字',
  day_wordcount_goal: 20000,
  instructions: `
### 1. 产品定位与目标读者
- **目标平台**: 番茄小说、起点中文网。兼顾免费阅读的快节奏和付费阅读的深度设定。
- **核心读者**: 18-35岁男性读者，对系统流、都市异能、美食文、国运元素有阅读偏好。他们追求新奇设定、强烈的爽点反馈和有深度的情节布局。

### 2. 故事风格与基调
- **整体风格**: 现代都市幻想，节奏明快，爽点密集。
- **基调**: 以轻松、热血的成长为主线，中期引入悬疑和权谋元素增加故事张力。主角性格杀伐果断，智商在线，行为逻辑符合“精致利己主义者”的成长过程。

### 3. 核心爽点设计
- **主爽点 (系统成长)**:
  - **解析万物**: 系统能解析食材、菜谱、对手技能，甚至古代遗迹，提供最优解，带来信息差碾压的快感。
  - **气运兑换**: 烹饪美食获得“美食气运”，兑换稀有食材、传说厨具、强大食灵、甚至临时国运加持。
- **辅爽点 (情节驱动)**:
  - **美食对决**: 不仅是厨艺比拼，更是“食灵”的战斗和背后势力的博弈，场面宏大。
  - **打脸逆袭**: 主角从一个落魄厨师，一步步打脸看不起他的竞争对手、美食世家和海外挑战者。
  - **探索解谜**: 揭开家族衰败、美食文明断代的历史谜团，获得古代传承和宝藏。
  - **国运之争**: 以美食影响国运，参与国际级的美食竞赛，为国争光，获得国民崇拜和现实世界的影响力。
`,
  input_brief: `
### 1. 主角设定 (Character)
- **姓名**: 楚曜
- **背景**: 曾经是名震一方的美食世家“楚家”的嫡系传人，但家族在一夜之间离奇衰败，父母失踪，他只能在城市角落开一家小餐馆勉强度日，并守护着家族最后的秘密。
- **成长弧光**:
  - **初始缺陷 (Lie)**: “家族的衰败是我的原罪，我只能苟且偷生，无法重现辉煌。”
  - **外在欲望 (Want)**: 赚钱，找到父母，重振家族。
  - **内在需求 (Need)**: 找到自我身份认同，不再背负家族的沉重枷锁，而是作为“楚曜”自己，开创一条前无古人的美食之道，并承担起守护国运的责任。
- **金手指 (System)**:
  - **名称**: “道”系统 (内部代号，初期表现为解析能力)
  - **核心机制**:
    1. **解析**: 可解析一切与“食”相关事物。解析食材，可知其最佳烹饪方式；解析菜谱，可优化流程；解析对手，可知其厨艺弱点。
    2. **演化**: 随着主角烹饪的菜品等级和蕴含气运的提升，系统会解锁新功能，如“气运熔炉”（兑换）、“食灵空间”（培养）、“时空秘境”（获取特殊食材）。
  - **限制**: 解析和演化需要消耗精神力和“美食气运”，过度使用会导致虚弱。高级物品的兑换有前置条件和成功率。

### 2. 世界观核心设定 (World Building)
- **核心概念**: 美食是连接物质世界与精神能量的桥梁。极致的美味可以诞生拥有自主意识的能量体——**食灵**。
- **独特法则**:
  1. **食灵共生**: 厨师可以通过血脉或特殊仪式与“食灵”签订契约，获得超凡厨艺和战斗力。强大的食灵甚至可以影响一方水土的气候。
  2. **美食气运**: 食物不仅提供能量，还蕴含“气运”。普通食物滋养个人，而蕴含历史文化、国民情感的“国菜”则与国运相连。烹饪或品尝国菜，可以增强或削弱国运。
  3. **文明断代**: 历史上曾存在一个辉煌的美食文明，厨师能移山填海，食灵可媲美神明。但一场未知的灾难导致了文明断代，大部分强大的食灵和菜谱失传，主角的家族秘密与此相关。
- **背景谜团**:
  - 楚家为何一夜衰败？父母失踪是否与守护的秘密有关？
  - 历史上的美食文明断代是天灾还是人祸？
  - 海外神秘组织为何也在寻找失落的食灵和菜谱？
`,
  constraints: `
### 1. 竞品分析与差异化
- **差异化**:
  - **世界观**: 引入“食灵”和“美食文明断代史”，相比传统美食文，增加了养成和探索元素，格局更宏大。
  - **金手指**: “解析”能力比简单的“兑换”更具操作感和智慧感，为主角的“智斗”情节提供支撑。
  - **核心冲突**: 从个人恩怨、家族复兴上升到国运之争和文明传承，目标更远大。

### 2. 市场风险与规避策略
- **风险1 (设定复杂)**: 世界观和系统设定较多，可能劝退部分只想看快节奏爽文的读者。
  - **规避**: 不在早期堆砌设定。通过“神秘老人”的口和具体情节，逐步、自然地揭示世界观。系统功能逐级解锁，保持新鲜感。
- **风险2 (金手指平衡)**: “解析”能力过强可能导致主角无敌，失去成长感。
  - **规避**: 设定精神力消耗和冷却时间。高级目标的解析需要前置条件（如特定道具、自身厨艺等级）。强调解析只是提供“最优解”，执行仍需主角自身努力和技巧。
- **风险3 (读者毒点)**:
  - **圣母/降智**: 主角楚曜设定为有家族仇恨背景的现实主义者，行事以自身利益和复仇为优先，不圣母，不无脑。
  - **感情线**: 感情线为辅，可以有红颜知己，但绝不拖沓，不影响主线节奏。女性角色应独立、有魅力，是伙伴而非附庸。
`,
  acceptance_criteria: `
1. **开篇留存**: 三章内必须完成主角背景交代、金手指激活、核心悬念（家族之谜、国运之争）的铺设，并至少完成一次“打脸-装逼”的完整爽点循环。读者评论区应出现“养肥”、“追了”等正面反馈。
2. **角色塑造**: 主角楚曜的“背负仇恨”和“渴望崛起”的形象必须在开篇通过行动（而非旁白）鲜明地建立起来。
3. **爽点验证**: “解析”能力的核心爽点必须在第二章得到清晰展示，并证明其在解决冲突中的有效性和强大之处。
4. **世界观呈现**: “食灵”或“美食气运”的核心概念必须在第三章通过具体情节（而非设定轰炸）向读者揭示，并引发读者对后续世界的好奇。
5. **钩子强度**: 第三章结尾必须留下一个强有力的钩子，例如“饕餮议会”的第一个敌人已经出现，或者主角接到了一个不可能完成的新手任务。
`,
});

const isGeneratingIdea = ref(false);
const isCreating = ref(false);

const handleGenerateIdea = async () => {
  isGeneratingIdea.value = true;
  try {
    const idea = await bookStore.generateNewIdea();
    newBookForm.name = idea.name;
    newBookForm.goal = idea.goal;
    newBookForm.instructions = idea.instructions;
    newBookForm.input_brief = idea.input_brief;
    newBookForm.constraints = idea.constraints;
    newBookForm.acceptance_criteria = idea.acceptance_criteria;
    ElMessage.success('AI 创意已生成并填充！');
  } catch (error) {
    ElMessage.error('生成创意失败，请稍后重试。');
  } finally {
    isGeneratingIdea.value = false;
  }
};

const handleCreateBook = async () => {
  if (!newBookFormRef.value) return;
  await newBookFormRef.value.validate(async (valid) => {
    if (valid) {
      isCreating.value = true;
      try {
        const newBook = await bookStore.createNewBook(newBookForm);
        ElNotification({
          title: '成功',
          message: `项目《${newBook.name}》已创建！现在将跳转到项目管理页面。`,
          type: 'success',
        });
        // 创建成功后跳转到项目管理页, 并通过查询参数告知新创建的项目ID
        router.push({ path: '/dashboard', query: { newBookId: newBook.run_id } });
      } catch (error) {
        ElMessage.error('创建项目失败！');
      } finally {
        isCreating.value = false;
      }
    }
  });
};
</script>

<style scoped>
.create-book-view {
  padding: 20px;
}
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.page-header h1 {
  font-size: 24px;
  color: #303133;
  margin: 0;
}
.header-buttons {
  margin-left: auto;
}
</style>