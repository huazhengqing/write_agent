<template>
  <div class="dashboard" v-loading="isComponentLoading">
    <div v-if="!initError">
    <div v-if="!isComponentLoading && books.length > 0">
        <el-card v-for="book in books" :key="book.run_id" class="box-card book-item-card" :body-style="{ padding: activeCollapse[book.run_id] ? '20px' : '0px' }">
            <template #header>
            <div class="card-header" @click="handleCollapseChange(book)">
                <span class="book-title">
                  <el-icon class="collapse-arrow" :class="{ 'is-active': activeCollapse[book.run_id] }"><ArrowRight /></el-icon>
                  <el-tag :type="book.status === 'running' ? 'primary' : 'info'" effect="dark" size="small" class="running-tag">
                    {{ book.status === 'running' ? '运行中' : '未运行' }}
                  </el-tag>
                  {{ book.name }}
                  <el-tag v-if="book.word_count_today !== undefined" :type="book.word_count_today > 0 ? 'success' : 'info'" size="small" class="word-count-tag">
                    今日: {{ book.word_count_today }} 字
                  </el-tag>
                </span>
                <div>
                <el-popconfirm
                    title="确定要删除这个项目吗？所有相关数据将无法恢复。"
                    @confirm="handleDeleteBook(book.run_id)"
                >
                    <template #reference>
                    <el-button type="danger" @click.stop style="margin-right: 10px;">删除</el-button>
                    </template>
                </el-popconfirm>
                <el-button type="primary" @click.stop="handleUpdateBook(editingBooks[book.run_id])" :loading="updatingState[book.run_id]">保存修改</el-button>
                <el-button type="success" @click.stop="handleSyncBook(book.run_id)" :loading="syncingState[book.run_id]">同步项目</el-button>
                </div>
            </div>
            </template>
            
            <div v-if="activeCollapse[book.run_id]">
                <el-form v-if="editingBooks[book.run_id]" :model="editingBooks[book.run_id]" label-position="top" class="edit-form">

                    <!-- 第一行: run_id, name, category -->
                    <el-row :gutter="20">
                        <el-col :span="8">
                            <el-form-item label="run_id (只读)">
                                <el-input :value="book.run_id" disabled />
                            </el-form-item>
                        </el-col>
                        <el-col :span="8">
                            <el-form-item label="name">
                                <el-input v-model="editingBooks[book.run_id].name" />
                            </el-form-item>
                        </el-col>
                        <el-col :span="8">
                            <el-form-item label="category">
                                <el-select v-model="editingBooks[book.run_id].category" style="width: 100%;"><el-option label="故事/小说 (story)" value="story" /><el-option label="报告 (report)" value="report" /><el-option label="书籍 (book)" value="book" /></el-select>
                            </el-form-item>
                        </el-col>
                    </el-row>

                    <!-- 第二行: language, length, day_wordcount_goal -->
                    <el-row :gutter="20">
                        <el-col :span="8"><el-form-item label="language"><el-select v-model="editingBooks[book.run_id].language" style="width: 100%;"><el-option label="中文 (cn)" value="cn" /><el-option label="英文 (en)" value="en" /></el-select></el-form-item></el-col>
                        <el-col :span="8"><el-form-item label="length"><el-input v-model="editingBooks[book.run_id].length" /></el-form-item></el-col>
                        <el-col :span="8"><el-form-item label="day_wordcount_goal"><el-input-number v-model="editingBooks[book.run_id].day_wordcount_goal" :min="0" controls-position="right" style="width: 100%;" placeholder="未设置" /></el-form-item></el-col>
                    </el-row>

                    <!-- 核心目标 -->
                    <el-form-item label="核心目标">
                        <el-input v-model="editingBooks[book.run_id].goal" type="textarea" :rows="3" />
                    </el-form-item>

                    <!-- 单列布局 -->
                    <el-form-item v-for="key in oneColumnKeys" :key="key" :label="key">
                        <el-input v-model="editingBooks[book.run_id][key]" type="textarea" autosize />
                    </el-form-item>

                </el-form>
            </div>
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
import { onMounted, onUnmounted, ref, reactive, computed } from 'vue';
import { useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useBookStore } from '@/stores/bookStore'; 
import { ElMessage } from 'element-plus';
import type { BookMeta } from '@/api/books';

const isComponentLoading = ref(true);
const initError = ref<string | null>(null);

const router = useRouter();

// 修复：使用 const 和 storeToRefs 直接从 store 获取响应式状态，避免重新赋值。
const bookStore = useBookStore();
const { books, isLoading } = storeToRefs(bookStore);

try {
  // 可以在这里保留 try-catch 以捕获 useBookStore() 可能出现的罕见初始化错误
  // 但 storeToRefs 的操作是安全的，不需要放在这里。
} catch (e: any) {
  initError.value = `初始化数据存储失败: ${e.message}. 请检查 Pinia store 配置。`;
  isComponentLoading.value = false;
  console.error(initError.value, e);
}
const activeCollapse = reactive<Record<string, boolean>>({});
const updatingState = reactive<Record<string, boolean>>({});
const syncingState = reactive<Record<string, boolean>>({});


const editingBooks = reactive<Record<string, BookMeta>>({});

// 双列显示的字段
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

const handleCollapseChange = (book: BookMeta) => {
    const isActive = !activeCollapse[book.run_id];
    activeCollapse[book.run_id] = isActive;

    if (isActive && !editingBooks[book.run_id]) {
        // 深拷贝一份数据用于编辑，避免直接修改 store
        editingBooks[book.run_id] = JSON.parse(JSON.stringify(book));
    }
};

const handleSyncBook = async (runId: string) => {
  syncingState[runId] = true;
  try {
    const res = await bookStore.syncBookById(runId);
    ElMessage.success(res.data.message || '项目同步成功！正在等待后台启动...');
    // 同步成功后，全局轮询会自动检测到状态变化并开始工作
    await bookStore.updateBooksStatus(); // 立即手动更新一次状态，以便全局轮询能够尽快检测到
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
        const updatedBook = await bookStore.updateBookById(book.run_id, book);
        ElMessage.success(`项目《${updatedBook.name}》已更新！`);
        // 将 store 中最新的数据同步回 editingBooks，防止再次展开时看到旧数据
        // 必须从 store.books 中找到最新的 book 数据，因为它包含了 word_count_today
        const latestBookFromStore = books.value.find(b => b.run_id === book.run_id);
        if (latestBookFromStore) editingBooks[book.run_id] = JSON.parse(JSON.stringify(latestBookFromStore));
        // 更新后自动折叠
        activeCollapse[book.run_id] = false;
    } catch (error) {
        ElMessage.error('更新失败！');
    } finally {
        updatingState[book.run_id] = false;
    }
};

onMounted(async () => {
  try {
    await bookStore.fetchAllBooks();
    // 检查路由参数，看是否是从创建页面跳转而来
    const newBookId = router.currentRoute.value.query.newBookId as string;
    if (newBookId) {
      const newBook = books.value.find(b => b.run_id === newBookId);
      if (newBook) {
        handleCollapseChange(newBook);
      }
    }
  } catch (e: any) {
    ElMessage.error('获取项目列表失败，请稍后重试。');
    console.error('Failed to fetch all books:', e);
  }
  isComponentLoading.value = false;
});

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
  cursor: pointer;
}
.book-title {
  font-weight: bold;
  font-size: 1.1em;
  display: flex;
  align-items: center;
}
.running-tag {
  margin-right: 8px;
  height: 20px;
}
.word-count-tag {
  margin-left: 10px;
  height: 20px;
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
.collapse-arrow {
  margin-right: 8px;
  transition: transform 0.3s;
}
.collapse-arrow.is-active {
  transform: rotate(90deg);
}
.edit-form {}
</style>