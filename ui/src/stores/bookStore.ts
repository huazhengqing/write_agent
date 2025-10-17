// src/stores/bookStore.ts
import { defineStore } from 'pinia';
import { ref, reactive } from 'vue';
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
      // 修复: 将获取到的书籍数据赋值给 store 中的 books ref
      books.value = response.data;
      return books.value;
    } catch (err: any) {
      const message = err.response?.data?.detail || '获取书籍列表失败';
      error.value = message;
      console.error(error.value, err);
      throw new Error(message);
    } finally {
      isLoading.value = false;
    }
  }

  async function createNewBook(bookData: BookCreate) {
    try {
      const response = await createBook(bookData);
      books.value.push(response.data);
      return response.data;
    } catch (err) {
      console.error('创建书籍失败', err);
      throw err;
    }
  }

  async function deleteBookById(runId: string) {
    try {
      await deleteBook(runId);
      const index = books.value.findIndex(b => b.run_id === runId);
      if (index !== -1) books.value.splice(index, 1);
    } catch (err) {
      console.error(`删除书籍 ${runId} 失败`, err);
      throw err;
    }
  }

  async function updateBookById(runId: string, bookData: BookMeta) {
    try {
      const dataToUpdate = { ...bookData };
      const response = await updateBook(runId, dataToUpdate);
      const index = books.value.findIndex(b => b.run_id === runId);
      if (index !== -1) {
        // 后端返回了完整的更新后的 book 对象, 直接替换即可
        books.value[index] = response.data;
      } else {
        await fetchAllBooks();
      }
      return response.data;
    } catch (err) {
      console.error(`更新书籍 ${runId} 失败`, err);
      throw err;
    } finally {
    }
  }

  async function syncBookById(runId: string) {
    try {
      return await syncBook(runId);
    } catch (err) {
      console.error(error.value, err);
      throw err;
    }
  }

  async function generateNewIdea(): Promise<IdeaOutput> {
    const response = await generateIdea();
    return response.data;
  }

  async function updateBooksStatus() {
    if (books.value.length === 0) return;
    try {
      await fetchAllBooks();
    } catch (err) {
      console.error('更新书籍运行状态失败', err);
    }
  }

  return { books, isLoading, error, fetchAllBooks, createNewBook, deleteBookById, updateBookById, syncBookById, generateNewIdea, updateBooksStatus };
});
