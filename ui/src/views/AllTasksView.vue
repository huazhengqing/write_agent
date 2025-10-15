<template>
    <div class="all-tasks-view" v-loading="isLoading">
    <!-- 顶部操作栏 -->
    <div class="page-header action-bar">
      <el-button type="primary" @click="fetchAllTasks(true)" :loading="isManualRefreshing">刷新所有任务</el-button>
    </div>

    <div class="main-container">
      <!-- Left Pane: Task Forest -->
        <div class="left-pane">
        <div v-if="booksWithTasks.length > 0">
          <div v-for="book in booksWithTasks" :key="book.run_id" class="project-block">            
            <el-tree
              v-if="getTaskTree(book.run_id) && getTaskTree(book.run_id).length > 0"
              :data="getTaskTree(book.run_id)"
              :props="treeProps"
              node-key="task_id"
              :default-expanded-keys="expandedKeys[book.run_id]"
              :expand-on-click-node="false"
              @node-click="(task) => handleNodeClick(task, book.run_id)"
              :current-node-key="selectedTaskInfo?.task.task_id"
              @node-expand="(data) => handleNodeExpandCollapse(data, book.run_id, true)"
              @node-collapse="(data) => handleNodeExpandCollapse(data, book.run_id, false)"
              highlight-current
            >
              <template #default="{ node, data }">                
                <span class="custom-tree-node" :class="{ 'is-running': getTaskDisplayStatus(data, book) === 'running' }">
                  <span class="task-id" :title="data.task_id">[{{ data.task_id }}]</span>
                  <span v-if="data.hierarchical_position" class="task-info" :title="data.hierarchical_position">[{{ data.hierarchical_position }}]</span>
                  <span class="task-info" :title="data.task_type">[{{ data.task_type }}]</span>
                  <span v-if="data.length" class="task-info" :title="String(data.length)">[{{ data.length }}]</span>
                  <span class="task-goal-text" :title="node.level === 1 ? book.run_id : data.goal">
                    {{ node.level === 1 ? book.run_id : data.goal }}
                  </span>
                  <!-- 根任务额外显示今日字数 -->
                  <el-tag v-if="node.level === 1 && book.word_count_today !== undefined" :type="book.word_count_today > 0 ? 'success' : 'info'" size="small" class="word-count-tag">
                    今日: {{ book.word_count_today }} 字
                  </el-tag>
                </span>
              </template>
            </el-tree>
            <el-empty v-else description="该项目暂无任务" :image-size="50" />
          </div>
        </div>
        <el-empty v-else-if="!isLoading" description="暂无任何项目" />
      </div>

      <!-- Right Pane: Task Details and Actions -->
      <div class="right-pane">
        <div v-if="selectedTaskInfo" class="details-content">
          <div class="details-header">
            <h3>任务详情</h3>
            <div class="action-buttons">
              <el-popconfirm title="确定要删除此任务及其所有子任务吗？" @confirm="handleDeleteTask">
                <template #reference><el-button type="danger">删除</el-button></template>
              </el-popconfirm>
              <el-button type="primary" @click="handleUpdateTask" :loading="isUpdating">保存</el-button>
              <el-button type="success" @click="handleRunTask" :loading="runningTasks[selectedTaskInfo.task.task_id]" :disabled="getTaskDisplayStatus(selectedTaskInfo.task, findBook(selectedTaskInfo.run_id)!) === 'running' || isSelectedProjectRunning">运行</el-button>
            </div>
          </div>
          <el-form :model="editForm" label-position="top" class="task-edit-form">
            <el-row :gutter="20">
              <el-col :span="8">
                <el-form-item label="Task ID (只读)"><el-input :value="editForm.task_id" readonly /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="Parent ID (只读)"><el-input :value="editForm.parent_id" readonly /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="长度 (Length)"><el-input v-model="editForm.length" /></el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="8">
                <el-form-item label="任务类型 (Task Type)"><el-input v-model="editForm.task_type" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="状态 (Status)"><el-input v-model="editForm.status" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="层级位置 (Hierarchical Position)"><el-input v-model="editForm.hierarchical_position" /></el-form-item>
              </el-col>
            </el-row>
            <el-form-item label="任务目标 (Goal)">
                <el-input v-model="editForm.goal" type="textarea" autosize />
            </el-form-item>
            <el-form-item label="详细指令 (Instructions)">
                <el-input v-model="editForm.instructions" type="textarea" autosize />
            </el-form-item>
            <el-form-item label="输入简报 (Input Brief)">
                <el-input v-model="editForm.input_brief" type="textarea" autosize />
            </el-form-item>
            <el-form-item label="约束条件 (Constraints)">
                <el-input v-model="editForm.constraints" type="textarea" autosize />
            </el-form-item>
            <el-form-item label="验收标准 (Acceptance Criteria)">
                <el-input v-model="editForm.acceptance_criteria" type="textarea" autosize />
            </el-form-item>
            <el-form-item label="路由专家 (Expert)">
              <el-input v-model="editForm.expert" />
            </el-form-item>
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="原子任务判断 (Atom)">
                  <el-input v-model="editForm.atom" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-form-item label="原子任务判断推理 (Atom Reasoning)"><el-input v-model="editForm.atom_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="任务分解结果 (Plan)"><el-input v-model="editForm.plan" type="textarea" autosize /></el-form-item>
            <el-form-item label="任务分解推理 (Plan Reasoning)"><el-input v-model="editForm.plan_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="设计方案 (Design)"><el-input v-model="editForm.design" type="textarea" autosize /></el-form-item>
            <el-form-item label="设计方案推理 (Design Reasoning)"><el-input v-model="editForm.design_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="搜索结果 (Search)"><el-input v-model="editForm.search" type="textarea" autosize /></el-form-item>
            <el-form-item label="搜索结果推理 (Search Reasoning)"><el-input v-model="editForm.search_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="结构划分结果 (Hierarchy)"><el-input v-model="editForm.hierarchy" type="textarea" autosize /></el-form-item>
            <el-form-item label="结构划分推理 (Hierarchy Reasoning)"><el-input v-model="editForm.hierarchy_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="正文 (Write)"><el-input v-model="editForm.write" type="textarea" autosize /></el-form-item>
            <el-form-item label="正文推理 (Write Reasoning)"><el-input v-model="editForm.write_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="正文摘要 (Summary)"><el-input v-model="editForm.summary" type="textarea" autosize /></el-form-item>
            <el-form-item label="正文摘要推理 (Summary Reasoning)"><el-input v-model="editForm.summary_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="全书设计方案 (Book Level Design)"><el-input v-model="editForm.book_level_design" type="textarea" autosize /></el-form-item>
            <el-form-item label="全局状态 (Global State)"><el-input v-model="editForm.global_state" type="textarea" autosize /></el-form-item>
            <el-form-item label="正文评审结果 (Write Review)"><el-input v-model="editForm.write_review" type="textarea" autosize /></el-form-item>
            <el-form-item label="正文评审推理 (Write Review Reasoning)"><el-input v-model="editForm.write_review_reasoning" type="textarea" autosize /></el-form-item>
            <el-form-item label="翻译结果 (Translation)"><el-input v-model="editForm.translation" type="textarea" autosize /></el-form-item>
            <el-form-item label="翻译推理 (Translation Reasoning)"><el-input v-model="editForm.translation_reasoning" type="textarea" autosize /></el-form-item>
            
            <h4>上下文与知识图谱</h4>
            <el-form-item label="设计上下文 (Context Design)"><el-input v-model="editForm.context_design" type="textarea" autosize /></el-form-item>
            <el-form-item label="摘要上下文 (Context Summary)"><el-input v-model="editForm.context_summary" type="textarea" autosize /></el-form-item>
            <el-form-item label="搜索上下文 (Context Search)"><el-input v-model="editForm.context_search" type="textarea" autosize /></el-form-item>
            <el-form-item label="设计知识图谱 (KG Design)"><el-input v-model="editForm.kg_design" type="textarea" autosize /></el-form-item>
            <el-form-item label="写作知识图谱 (KG Write)"><el-input v-model="editForm.kg_write" type="textarea" autosize /></el-form-item>

            <h4>检索词</h4>
            <el-form-item label="设计检索词 (Inquiry Design)"><el-input v-model="editForm.inquiry_design" type="textarea" autosize /></el-form-item>
            <el-form-item label="摘要检索词 (Inquiry Summary)"><el-input v-model="editForm.inquiry_summary" type="textarea" autosize /></el-form-item>
            <el-form-item label="搜索检索词 (Inquiry Search)"><el-input v-model="editForm.inquiry_search" type="textarea" autosize /></el-form-item>

            <el-form-item label="结果 (Results, 只读)">
                <div class="readonly-details">
                    <pre>{{ JSON.stringify(selectedTaskInfo.task.results, null, 2) }}</pre>
                </div>
            </el-form-item>
          </el-form>
        </div>
        <el-empty v-else description="请从左侧选择一个任务进行操作" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive, watch, onUnmounted, computed } from 'vue';
import { useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useBookStore } from '@/stores/bookStore';
import { getTasksForBook, runTask, updateTask, deleteTask as apiDeleteTask, type Task, type TaskUpdate } from '@/api/tasks';
import { type BookMeta } from '@/api/books';
import { ElMessage } from 'element-plus';

interface BookWithTasks extends BookMeta {
  tasks: Task[]; // 扁平的任务列表
}

const router = useRouter();
const bookStore = useBookStore();
// 直接使用 bookStore 的 books 响应式引用
const { books } = storeToRefs(bookStore); 
const isLoading = ref(true);
const isManualRefreshing = ref(false);
const isUpdating = ref(false);
const booksWithTasks = ref<BookWithTasks[]>([]);

const selectedTaskInfo = ref<{ task: Task; run_id: string } | null>(null);
const runningTasks = reactive<Record<string, boolean>>({});
const expandedKeys = reactive<Record<string, string[]>>({});

const editForm = reactive<Partial<Task>>({});

let pollInterval: number | undefined;

const treeProps = {
  children: 'subtasks',
  label: 'goal',
};

watch(selectedTaskInfo, (newInfo) => {
    if (newInfo) {
        // 清空旧数据并用新任务数据填充
        Object.keys(editForm).forEach(key => delete editForm[key as keyof Task]);
        const taskData = JSON.parse(JSON.stringify(newInfo.task));
        // 确保 task_id 字段存在且正确，因为 el-tree 的 node-key 和模板都依赖它
        taskData.task_id = taskData.id;
        Object.assign(editForm, taskData);
    } else {
        Object.keys(editForm).forEach(key => delete editForm[key as keyof Task]);
    }
});

const isSelectedProjectRunning = computed(() => {
  if (!selectedTaskInfo.value) {
    return false;
  }
  const project = booksWithTasks.value.find(b => b.run_id === selectedTaskInfo.value.run_id);
  return project?.status === 'running' ?? false;
});

const hasAnyProjectRunning = computed(() => {
    return booksWithTasks.value.some(book => book.status === 'running');
});

const getTaskDisplayStatus = (task: Task, book: BookMeta): string => {
  // 如果项目正在运行，并且任务不是一个终态，我们乐观地认为它可能在运行。
  // 这是一个简化处理，因为后端现在只提供项目级别的运行状态。
  if (book.status === 'running' && !['completed', 'failed', 'cancelled', 'paused'].includes(task.status)) {
    return 'running';
  }
  // 否则，返回任务自身的数据库状态。
  return task.status;
};

const findBook = (runId: string) => booksWithTasks.value.find(b => b.run_id === runId);

// 辅助函数：构建任务树
function buildTaskTree(tasks: Task[]): Task[] {
    const taskMap = new Map<string, Task>();
    const rootTasks: Task[] = [];

    if (!tasks) return [];

    // 初始化所有任务，并为其添加 subtasks 数组
    tasks.forEach(task => {
        // 确保 task_id 存在，因为 el-tree 的 node-key 和模板都依赖它
        const taskWithId = { ...task, task_id: task.id, subtasks: [] };
        taskMap.set(task.task_id, taskWithId);
    });

    // 构建父子关系
    tasks.forEach(task => {
        const taskNode = taskMap.get(task.task_id)!;
        if (task.parent_id && taskMap.has(task.parent_id)) {
            const parent = taskMap.get(task.parent_id);
            parent?.subtasks.push(taskNode);
        } else {
            rootTasks.push(taskNode);
        }
    });
    
    // 自然排序
    const naturalSort = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
    const sortTasks = (taskList: Task[]) => {
        taskList.sort((a, b) => naturalSort(a.task_id, b.task_id));
        taskList.forEach(t => {
            if (t.subtasks && t.subtasks.length > 0) {
                sortTasks(t.subtasks);
            }
        });
    };
    sortTasks(rootTasks);

    return rootTasks;
}

const getTaskTree = (runId: string) => {
    const book = booksWithTasks.value.find(b => b.run_id === runId);
    return book ? buildTaskTree(book.tasks) : [];
};

const fetchAllTasks = async (isManual = false) => {
  if (isManual) isManualRefreshing.value = true;
  isLoading.value = true;
  try {
    // 关键加固：每次刷新都先获取最新的全量项目列表
    await bookStore.fetchAllBooks(true);
    const taskPromises = books.value.map(book =>
      getTasksForBook(book.run_id)
        .then(response => ({ ...book, tasks: response.data }))
        .catch(e => ({ ...book, tasks: [] }))
    );
    booksWithTasks.value = await Promise.all(taskPromises);
    if (isManual) ElMessage.success('所有任务已刷新！');
  } catch (error) {
    console.error("刷新所有任务失败:", error);
    if (isManual) ElMessage.error("刷新任务失败！");
  } finally {
    isLoading.value = false;
    if (isManual) isManualRefreshing.value = false;
  }
};

onMounted(() => {
  fetchAllTasks(true); // 初始加载
});

onUnmounted(() => {
    clearInterval(pollInterval);
});

const goBack = () => router.push('/');

const statusTagType = (status: string) => {
  switch (status) {
    case 'completed': return 'success';
    case 'running': return 'primary';
    case 'failed': return 'danger';
    case 'pending': return 'info';
    default: return 'warning';
  }
};

const handleNodeClick = (task: Task, run_id: string) => {
    selectedTaskInfo.value = { task, run_id };
};

const handleNodeExpandCollapse = (task: Task, run_id: string, isExpand: boolean) => {
    if (!expandedKeys[run_id]) {
        expandedKeys[run_id] = [];
    }
    const keyIndex = expandedKeys[run_id].indexOf(task.task_id);

    if (isExpand) {
        if (keyIndex === -1) {
            expandedKeys[run_id].push(task.task_id);
        }
    } else {
        if (keyIndex > -1) {
            expandedKeys[run_id].splice(keyIndex, 1);
        }
    }
};

const handleUpdateTask = async () => {
    if (!selectedTaskInfo.value) return;
    isUpdating.value = true;
    const { task, run_id } = selectedTaskInfo.value;
    try {
        await updateTask(run_id, task.task_id, editForm as TaskUpdate);
        ElMessage.success('任务更新成功！');

    } catch (error) {
        ElMessage.error('任务更新失败！');
    } finally {
        isUpdating.value = false;
    }
};

const handleRunTask = async () => {
    if (!selectedTaskInfo.value) return;
    const { task, run_id } = selectedTaskInfo.value;
    runningTasks[task.task_id] = true;
    try {
        await runTask(run_id, task.task_id);
        ElMessage.success(`任务 '${task.goal}' 已开始在后台执行。`);
        await fetchAllTasks(true); // 立即刷新以看到状态变化
    } catch (error: any) {
        ElMessage.error(error.response?.data?.detail || `任务 '${task.goal}' 启动失败！`);
    } finally {
        runningTasks[task.task_id] = false;
    }
};

const handleDeleteTask = async () => {
    if (!selectedTaskInfo.value) return;
    const { task, run_id } = selectedTaskInfo.value;
    try {
        await apiDeleteTask(run_id, task.task_id);
        ElMessage.success(`任务 '${task.goal}'及其子任务已删除。`);
        selectedTaskInfo.value = null; // 清除选中
        await fetchAllTasks(true);
    } catch (error) {
        ElMessage.error(`任务删除失败！`);
    }
};

</script>
<style scoped>
.all-tasks-view { padding: 20px; display: flex; flex-direction: column; height: calc(100vh - 40px); box-sizing: border-box; }
.page-header { margin-bottom: 20px; flex-shrink: 0; height: 32px; /* 保持布局稳定性 */ }
.action-bar {
  text-align: left;
}
.main-container { display: flex; flex-grow: 1; overflow: hidden; gap: 20px; }
.left-pane { width: 40%; border: 1px solid #dcdfe6; border-radius: 4px; padding: 15px; overflow-y: auto; }
.right-pane { width: 60%; border: 1px solid #dcdfe6; border-radius: 4px; padding: 15px; overflow-y: auto; }
.project-block { margin-bottom: 20px; }
.project-header { display: flex; align-items: center; margin-bottom: 10px; }
.book-name { font-weight: bold; font-size: 16px; margin: 0 10px; }
.custom-tree-node { display: flex; align-items: center; width: 100%; overflow: hidden; }
.project-root-node {
  font-size: 14px;
  font-weight: bold;
}
.running-tag {
  height: 20px;
  line-height: 20px;
  margin-right: 5px;
}
.word-count-tag {
  margin-left: 10px;
  height: 20px;
}
.task-goal-text {
  margin-left: 5px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 1; /* 允许目标文本收缩 */
}
.task-tag {
  flex-shrink: 0; /* 不允许标签收缩 */
}
.task-id {
  margin-left: 8px;
  font-weight: bold;
  flex-shrink: 0;
}
.task-info {
  margin-left: 5px;
  color: #909399;
  font-size: 12px;
  flex-shrink: 0;
}
.el-tag { margin-right: 8px; }
.details-content { display: flex; flex-direction: column; height: 100%; }
.details-content .el-form { flex-grow: 1; }
.task-edit-form {
  overflow-y: auto;
  /* 增加一些内边距以避免滚动条遮挡内容 */
  padding-right: 10px;
}
.action-buttons {
  flex-shrink: 0;
}
.action-buttons .el-button {
  margin-left: 10px;
}
.details-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  flex-shrink: 0;
}
.readonly-details pre {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 4px;
    white-space: pre-wrap;
    font-size: 12px;
    max-height: 200px;
    overflow-y: auto;
    word-break: break-all;
}

/* 新增：为正在运行的任务行添加背景色 */
.custom-tree-node.is-running {
  background-color: #ecf5ff; /* element-plus primary-light-9 */
  color: #409eff; /* element-plus primary */
}
/* 新增：调整高亮选中行的样式，使其优先级更高 */
.el-tree--highlight-current .el-tree-node.is-current > .el-tree-node__content .custom-tree-node {
  background-color: #d9ecff; /* 更深一点的蓝色，用于区分选中和运行 */
}

/* 覆盖 element-plus 的默认样式，确保 task-id 不会因为 flex 布局被挤压 */
.el-tree-node__content > .custom-tree-node {
  display: flex;
  align-items: center;
  width: 100%;
}
</style>