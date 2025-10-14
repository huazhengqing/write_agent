// src/api/tasks.ts
import apiClient from './index';

// 任务状态枚举
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';

// 对应后端的 models.Task 模型
export interface Task {
  task_id: string;
  run_id: string;
  parent_id: string | null;
  hierarchical_position: string;
  task_type: string;
  status: TaskStatus;
  goal: string;
  length: string | null;
  instructions: string | null;
  input_brief: string | null;
  constraints: string | null;
  acceptance_criteria: string | null;
  reasoning: string | null;
  expert: string | null;
  results: Record<string, any>;
  subtasks: Task[]; // 这个字段在前端构建树形结构时添加
}

// 对应后端的 TaskUpdate 模型
export interface TaskUpdate {
  parent_id?: string;
  hierarchical_position?: string;
  task_type?: string;
  status?: TaskStatus;
  goal?: string;
  length?: string;
  instructions?: string;
  input_brief?: string;
  constraints?: string;
  acceptance_criteria?: string;
  reasoning?: string;
  expert?: string;
  results?: Record<string, any>;
}

export interface TaskRunResponse {
    message: string;
    run_id: string;
    task_id: string;
    status_url: string;
}

export const getTasksForBook = (runId: string) => apiClient.get<Task[]>(`/books/${runId}/tasks`);
export const getTask = (runId: string, taskId: string) => apiClient.get<Task>(`/tasks/${taskId}`, { params: { run_id: runId } });
export const updateTask = (runId: string, taskId: string, taskUpdate: TaskUpdate) => apiClient.put<Task>(`/tasks/${taskId}`, taskUpdate, { params: { run_id: runId } });
export const deleteTask = (runId: string, taskId: string) => apiClient.delete(`/tasks/${taskId}`, { params: { run_id: runId } });
export const runTask = (runId: string, taskId: string) => apiClient.post<TaskRunResponse>(`/tasks/${taskId}/run`, {}, { params: { run_id: runId } });
