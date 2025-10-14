// src/api/books.ts
import apiClient from './index';

// 对应后端的 BookMeta 模型
export interface BookMeta {
  run_id: string;
  name: string;
  goal?: string | null;
  category?: string | null;
  language?: string | null;
  instructions?: string | null;
  input_brief?: string | null;
  constraints?: string | null;
  acceptance_criteria?: string | null;
  length?: string | null;
  day_wordcount_goal?: number | null;
  title?: string | null;
  synopsis?: string | null;
  style?: string | null;
  book_level_design?: string | null;
  global_state_summary?: string | null;
}

// 对应后端的 BookCreate 模型
export interface BookCreate {
  name: string;
  goal: string;
  category?: string;
  language?: string;
  instructions?: string;
  input_brief?: string;
  constraints?: string;
  acceptance_criteria?: string;
  length?: string;
}

// 对应后端的 IdeaOutput 模型
export interface IdeaOutput {
  name: string;
  goal: string;
  instructions: string;
}

export const getAllBooks = () => apiClient.get<BookMeta[]>('/books');
export const getBook = (runId: string) => apiClient.get<BookMeta>(`/books/${runId}`);
export const createBook = (bookData: BookCreate) => apiClient.post<BookMeta>('/books', bookData);
export const updateBook = (runId: string, bookData: BookMeta) => apiClient.put<BookMeta>(`/books/${runId}`, bookData);
export const deleteBook = (runId: string) => apiClient.delete(`/books/${runId}`);
export const syncBook = (runId: string) => apiClient.post<{ message: string }>(`/books/${runId}/sync`);
export const generateIdea = () => apiClient.post<IdeaOutput>('/books/generate-idea');
