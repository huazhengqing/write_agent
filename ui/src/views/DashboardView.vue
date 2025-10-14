<template>
  <div class="dashboard" v-loading="isLoading">
    <el-card class="box-card create-book-card">
      <template #header>
        <div class="card-header">
          <span>➕ 创建新项目</span>
          <el-button
            class="button"
            text
            @click="handleGenerateIdea"
            :loading="isGeneratingIdea"
          >
            🤖 AI 生成创意
          </el-button>
        </div>
      </template>
      <el-form :model="newBookForm" label-position="top" ref="newBookFormRef">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="书名/项目名" prop="name" :rules="{ required: true, message: '书名不能为空', trigger: 'blur' }">
              <el-input v-model="newBookForm.name" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
             <el-form-item label="预估总字数" prop="length">
              <el-input v-model="newBookForm.length" placeholder="例如: 100万字"/>
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="核心目标" prop="goal" :rules="{ required: true, message: '核心目标不能为空', trigger: 'blur' }">
          <el-input v-model="newBookForm.goal" type="textarea" :rows="3" />
        </el-form-item>
        <el-form-item label="具体指令 (Instructions)">
          <el-input v-model="newBookForm.instructions" type="textarea" :rows="5" />
        </el-form-item>
        <el-row :gutter="20">
            <el-col :span="12">
                <el-form-item label="输入简报 (Input Brief)">
                    <el-input v-model="newBookForm.input_brief" type="textarea" autosize />
                </el-form-item>
            </el-col>
            <el-col :span="12">
                <el-form-item label="约束条件 (Constraints)">
                    <el-input v-model="newBookForm.constraints" type="textarea" autosize />
                </el-form-item>
            </el-col>
            <el-col :span="12">
                <el-form-item label="验收标准 (Acceptance Criteria)">
                    <el-input v-model="newBookForm.acceptance_criteria" type="textarea" autosize />
                </el-form-item>
            </el-col>
        </el-row>
        <el-form-item>
          <el-button type="primary" @click="handleCreateBook" :loading="isCreating">创建项目</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-divider>
      <el-icon><notebook /></el-icon>
      <span style="margin: 0 10px;">项目列表</span>
    </el-divider>

    <div v-if="books.length > 0">
      <el-card v-for="book in books" :key="book.run_id" class="box-card book-item-card">
        <template #header>
          <div class="card-header">
            <span class="book-title">{{ book.name }} <small>(ID: {{ book.run_id }})</small></span>
            <div>
              <el-button type="primary" @click="goToTasks">查看任务</el-button>
              <el-button @click="handleSyncBook(book.run_id)">同步项目</el-button>
              <el-popconfirm
                title="确定要删除这个项目吗？所有相关数据将无法恢复。"
                @confirm="handleDeleteBook(book.run_id)"
              >
                <template #reference>
                  <el-button type="danger">删除</el-button>
                </template>
              </el-popconfirm>
            </div>
          </div>
        </template>
        
        <el-collapse v-model="activeCollapse[book.run_id]" @change="handleCollapseChange(book, $event)">
          <el-collapse-item title="编辑项目详情" :name="book.run_id">
            <el-form v-if="editingBooks[book.run_id]" :model="editingBooks[book.run_id]" label-position="top">
                <el-form-item label="核心目标">
                    <el-input v-model="editingBooks[book.run_id].goal" type="textarea" :rows="3" />
                </el-form-item>
                <!-- 渲染所有可编辑字段 -->
                <el-row :gutter="20">
                    <el-col v-for="key in editableKeys" :key="key" :span="12">
                         <el-form-item :label="key">
                            <el-input v-model="editingBooks[book.run_id][key]" type="textarea" autosize />
                        </el-form-item>
                    </el-col>
                </el-row>
                <el-form-item>
                    <el-button type="primary" @click="handleUpdateBook(editingBooks[book.run_id])" :loading="updatingState[book.run_id]">保存修改</el-button>
                </el-form-item>
            </el-form>
          </el-collapse-item>
        </el-collapse>
      </el-card>
    </div>
    <el-empty v-else description="暂无项目，请在上方创建一个新项目。"></el-empty>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, reactive } from 'vue';
import { useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useBookStore } from '@/stores/bookStore';
import { ElMessage, ElNotification } from 'element-plus';
import type { FormInstance } from 'element-plus'
import { Notebook } from '@element-plus/icons-vue'
import type { BookCreate, BookMeta } from '@/api/books';

const router = useRouter();
const bookStore = useBookStore();
const { books, isLoading } = storeToRefs(bookStore);

const newBookFormRef = ref<FormInstance>();
const newBookForm = reactive<BookCreate>({
  name: '',
  goal: '',
  instructions: '',
  length: '',
  input_brief: '',
  constraints: '',
  acceptance_criteria: '',
});

const isGeneratingIdea = ref(false);
const isCreating = ref(false);
const activeCollapse = reactive<Record<string, string[]>>({});
const updatingState = reactive<Record<string, boolean>>({});
const editingBooks = reactive<Record<string, BookMeta>>({});

// 定义哪些 BookMeta 字段是可编辑的文本域
const editableKeys: (keyof BookMeta)[] = [
    'instructions', 'input_brief', 'constraints', 'acceptance_criteria', 'length',
    'title', 'synopsis', 'style', 'book_level_design', 'global_state_summary'
];

onMounted(() => {
  bookStore.fetchAllBooks();
});

const handleCollapseChange = (book: BookMeta, activeNames: any) => {
    const isActive = activeNames.includes(book.run_id);
    if (isActive && !editingBooks[book.run_id]) {
        // 深拷贝一份数据用于编辑，避免直接修改 store
        editingBooks[book.run_id] = JSON.parse(JSON.stringify(book));
    }
};

const handleGenerateIdea = async () => {
  isGeneratingIdea.value = true;
  try {
    const idea = await bookStore.generateNewIdea();
    newBookForm.name = idea.name;
    newBookForm.goal = idea.goal;
    newBookForm.instructions = idea.instructions;
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
          message: `项目《${newBook.name}》已创建！`,
          type: 'success',
        });
        // 重置表单
        newBookFormRef.value?.resetFields();
      } catch (error) {
        ElMessage.error('创建项目失败！');
      } finally {
        isCreating.value = false;
      }
    }
  });
};

const goToTasks = () => {
  router.push(`/tasks`);
};

const handleSyncBook = async (runId: string) => {
  try {
    const res = await bookStore.syncBookById(runId);
    ElMessage.success(res.data.message || '项目同步成功！');
  } catch (error) {
    ElMessage.error('项目同步失败！');
  }
};

const handleDeleteBook = async (runId: string) => {
  try {
    await bookStore.deleteBookById(runId);
    ElMessage.success('项目已删除。');
  } catch (error) {
    ElMessage.error('删除项目失败！');
  }
};

const handleUpdateBook = async (book: BookMeta) => {
    updatingState[book.run_id] = true;
    try {
        await bookStore.updateBookById(book.run_id, book);
        ElMessage.success(`项目《${book.name}》已更新！`);
        // 可选：更新后自动折叠
        if (activeCollapse[book.run_id]) {
            activeCollapse[book.run_id] = [];
        }
    } catch (error) {
        ElMessage.error('更新失败！');
    } finally {
        updatingState[book.run_id] = false;
    }
};

</script>

<style scoped>
.dashboard {
  padding: 20px;
}
.create-book-card, .book-item-card {
  margin-bottom: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.book-title {
  font-weight: bold;
  font-size: 1.1em;
}
.book-title small {
    font-size: 0.8em;
    color: #909399;
    margin-left: 8px;
}
.el-divider span {
    font-size: 1.2em;
    color: #606266;
}
</style>