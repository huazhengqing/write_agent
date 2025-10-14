<template>
  <div class="dashboard" v-loading="isComponentLoading">
    <div v-if="!initError">
    <div v-if="!isComponentLoading && books.length > 0">
        <el-card v-for="book in books" :key="book.run_id" class="box-card book-item-card">
            <template #header>
            <div class="card-header">
                <span class="book-title">{{ book.name }} <small>(ID: {{ book.run_id }})</small></span>
                <div>
                <el-button @click="handleSyncBook(book.run_id)" :loading="syncingState[book.run_id]">同步项目</el-button>
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
            
            <el-collapse v-model="activeCollapse[book.run_id]" @change="handleCollapseChange(book, $event)" accordion>
            <el-collapse-item title="编辑项目详情" :name="book.run_id">
                <el-form v-if="editingBooks[book.run_id]" :model="editingBooks[book.run_id]" label-position="top">

                    <!-- 第一行: run_id 和 name -->
                    <el-row :gutter="20">
                        <el-col :span="12">
                            <el-form-item label="run_id (只读)">
                                <el-input :value="book.run_id" disabled />
                            </el-form-item>
                        </el-col>
                        <el-col :span="12">
                            <el-form-item label="name">
                                <el-input v-model="editingBooks[book.run_id].name" />
                            </el-form-item>
                        </el-col>
                    </el-row>

                    <!-- 其他双列布局字段 -->
                    <el-row :gutter="20">
                        <el-col v-for="key in twoColumnKeys" :key="key" :span="12">
                            <el-form-item :label="key">
                                <el-input-number v-if="key === 'day_wordcount_goal'" v-model="editingBooks[book.run_id][key]" :min="0" controls-position="right" style="width: 100%;" placeholder="未设置" />
                                <el-select v-else-if="key === 'category'" v-model="editingBooks[book.run_id][key]" style="width: 100%;"><el-option label="故事/小说 (story)" value="story" /><el-option label="报告 (report)" value="report" /><el-option label="书籍 (book)" value="book" /></el-select>
                                <el-select v-else-if="key === 'language'" v-model="editingBooks[book.run_id][key]" style="width: 100%;"><el-option label="中文 (cn)" value="cn" /><el-option label="英文 (en)" value="en" /></el-select>
                                <el-input v-else v-model="editingBooks[book.run_id][key]" />
                            </el-form-item>
                        </el-col>
                    </el-row>

                    <!-- 核心目标 -->
                    <el-form-item label="核心目标">
                        <el-input v-model="editingBooks[book.run_id].goal" type="textarea" :rows="3" />
                    </el-form-item>

                    <!-- 单列布局 -->
                    <el-form-item v-for="key in oneColumnKeys" :key="key" :label="key">
                        <el-input v-model="editingBooks[book.run_id][key]" type="textarea" autosize />
                    </el-form-item>

                    <el-form-item>
                        <el-button type="primary" @click="handleUpdateBook(editingBooks[book.run_id])" :loading="updatingState[book.run_id]">保存修改</el-button>
                    </el-form-item>
                </el-form>
            </el-collapse-item>
            </el-collapse>
        </el-card>
    </div>
    <el-empty v-else-if="!isComponentLoading && books.length === 0" description="暂无项目，请从侧边栏“创建项目”开始。"></el-empty>
    </div>
    <div v-else>
       <el-alert title="页面加载失败" :description="initError" type="error" show-icon :closable="false" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, reactive } from 'vue';
import { useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useBookStore } from '@/stores/bookStore';
import { ElMessage } from 'element-plus';
import { Notebook } from '@element-plus/icons-vue'
import type { BookMeta } from '@/api/books';

const isComponentLoading = ref(true);
const initError = ref<string | null>(null);

const router = useRouter();
let bookStore: ReturnType<typeof useBookStore>;
let books = ref<BookMeta[]>([]);
let isLoading = ref(true);

try {
  bookStore = useBookStore();
  // 从 store 中解构出响应式状态
  const storeRefs = storeToRefs(bookStore);
  books = storeRefs.books;
  isLoading = storeRefs.isLoading;
} catch (e: any) {
  initError.value = `初始化数据存储失败: ${e.message}. 请检查 Pinia store 配置。`;
  isComponentLoading.value = false;
  console.error(initError.value, e);
}

const activeCollapse = reactive<Record<string, string>>({});
const updatingState = reactive<Record<string, boolean>>({});
const syncingState = reactive<Record<string, boolean>>({});
const editingBooks = reactive<Record<string, BookMeta>>({});

// 双列显示的字段
const twoColumnKeys: (keyof BookMeta)[] = [
    'category', 'language', 'length', 'day_wordcount_goal'
];

// 单列显示的字段
const oneColumnKeys: (keyof BookMeta)[] = [
    'instructions',
    'input_brief',
    'constraints',
    'acceptance_criteria',
    'title',
    'synopsis',
    'style',
    'book_level_design',
    'global_state_summary'
];

onMounted(async () => {
  if (bookStore) {
    try {
      await bookStore.fetchAllBooks();
    } catch (e: any) {
      ElMessage.error('获取项目列表失败，请稍后重试。');
      console.error('Failed to fetch all books:', e);
    }
  }
  isComponentLoading.value = false;
});

const handleCollapseChange = (book: BookMeta, activeName: string | number) => {
    const isActive = activeName === book.run_id;
    if (isActive && !editingBooks[book.run_id]) {
        // 深拷贝一份数据用于编辑，避免直接修改 store
        editingBooks[book.run_id] = JSON.parse(JSON.stringify(book));
    } else if (!isActive && editingBooks[book.run_id]) {
        // 当折叠项关闭时，从 editingBooks 中移除，以释放内存
        delete editingBooks[book.run_id];
    }
};

const handleSyncBook = async (runId: string) => {
  syncingState[runId] = true;
  try {
    const res = await bookStore.syncBookById(runId);
    ElMessage.success(res.data.message || '项目同步成功！');
  } catch (error) {
    ElMessage.error('项目同步失败！');
  } finally {
    syncingState[runId] = false;
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
        activeCollapse[book.run_id] = '';
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