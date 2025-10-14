// src/stores/bookStore.ts
import { defineStore } from 'pinia';
import { ref } from 'vue';
import { getAllBooks, createBook, deleteBook, updateBook, syncBook, generateIdea, type BookMeta, type BookCreate, type IdeaOutput } from '@/api/books';

export const useBookStore = defineStore('books', () => {
  const books = ref<BookMeta[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  async function fetchAllBooks() {
    isLoading.value = true;
    error.value = null;
    try {
      const response = await getAllBooks();
      books.value = response.data;
      return response.data;
    } catch (err) {
      error.value = '获取书籍列表失败';
      console.error(error.value, err);
      throw error;
    } finally {
      isLoading.value = false;
    }
  }

  async function createNewBook(bookData: BookCreate) {
    isLoading.value = true;
    error.value = null;
    try {
      const response = await createBook(bookData);
      // 优化：直接将新书添加到列表，而不是重新获取所有
      books.value.push(response.data);
      return response.data;
    } catch (err) {
      error.value = '创建书籍失败';
      console.error(error.value, err);
      throw error;
    } finally {
      isLoading.value = false;
    }
  }

  async function deleteBookById(runId: string) {
    isLoading.value = true;
    error.value = null;
    try {
      await deleteBook(runId);
      // 优化：直接从列表中移除，而不是重新获取所有
      const index = books.value.findIndex(b => b.run_id === runId);
      if (index !== -1) books.value.splice(index, 1);
    } catch (err) {
      error.value = `删除书籍 ${runId} 失败`;
      console.error(error.value, err);
      throw err;
    } finally {
      isLoading.value = false;
    }
  }

  async function updateBookById(runId: string, bookData: BookMeta) {
    isLoading.value = true;
    error.value = null;
    try {
      // 创建一个不包含 run_id 的副本用于请求体
      const dataToUpdate = { ...bookData };
      // delete dataToUpdate.run_id; // API的updateBook需要完整的BookMeta，如果后端不需要run_id在body中，则取消此行注释

      const response = await updateBook(runId, dataToUpdate);
      // 更新成功后刷新列表，以获取最新数据
      const index = books.value.findIndex(b => b.run_id === runId);
      if (index !== -1) {
        books.value[index] = response.data;
      } else {
        await fetchAllBooks();
      }
      return response.data;
    } catch (err) {
      error.value = `更新书籍 ${runId} 失败`;
      console.error(error.value, err);
      throw err;
    } finally {
      isLoading.value = false;
    }
  }

  async function syncBookById(runId: string) {
    isLoading.value = true;
    error.value = null;
    try {
      return await syncBook(runId);
    } catch (err) {
      error.value = `同步书籍 ${runId} 失败`;
      console.error(error.value, err);
      throw err;
    } finally {
      isLoading.value = false;
    }
  }

  async function generateNewIdea(): Promise<IdeaOutput> {
    // 对于这种一次性操作，可以在组件级别处理加载和错误状态
    const response = await generateIdea();
    return response.data;
  }

  return { books, isLoading, error, fetchAllBooks, createNewBook, deleteBookById, updateBookById, syncBookById, generateNewIdea };
});
