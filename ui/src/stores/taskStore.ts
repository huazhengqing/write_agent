// src/stores/taskStore.ts
import { defineStore } from 'pinia';
import { ref } from 'vue';
import { getTasksForBook, updateTask as apiUpdateTask, deleteTask as apiDeleteTask, runTask as apiRunTask, type TaskUpdate, type TaskRunResponse } from '@/api/tasks';
import { getBook, type BookMeta } from '@/api/books';
import type { Task } from '@/api/tasks';

/**
 * 自然排序函数，用于正确排序 '1.2', '1.10' 这样的字符串
 * @param a 字符串a
 * @param b 字符串b
 */
function naturalSort(a: string, b: string): number {
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

function buildTaskTree(tasks: Task[]): Task[] {
    const taskMap = new Map<string, Task>();
    const rootTasks: Task[] = [];

    // 初始化所有任务，并为其添加 subtasks 数组
    tasks.forEach(task => {
        task.subtasks = [];
        taskMap.set(task.task_id, task);
    });

    // 构建父子关系
    tasks.forEach(task => {
        if (task.parent_id && taskMap.has(task.parent_id)) {
            const parent = taskMap.get(task.parent_id);
            parent?.subtasks.push(task);
        } else {
            rootTasks.push(task);
        }
    });
    
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

export const useTaskStore = defineStore('tasks', () => {
    const tasks = ref<Task[]>([]);
    const currentBook = ref<BookMeta | null>(null);
    const isLoading = ref(false);
    const error = ref<string | null>(null);
    const currentRunId = ref<string | null>(null);

    async function fetchTasksForBook(runId: string) {
        isLoading.value = true;
        currentRunId.value = runId;
        error.value = null;
        try {
            const [bookResponse, tasksResponse] = await Promise.all([
                getBook(runId),
                getTasksForBook(runId)
            ]);
            currentBook.value = bookResponse.data;
            tasks.value = buildTaskTree(tasksResponse.data);
        } catch (err) {
            error.value = `获取项目 ${runId} 的任务失败`;
            console.error(error.value, err);
            tasks.value = [];
            currentBook.value = null;
            throw err;
        } finally {
            isLoading.value = false;
        }
    }

    // 辅助函数：在树中查找任务
    function findTaskInTree(taskId: string, taskList: Task[]): Task | null {
        for (const task of taskList) {
            if (task.task_id === taskId) return task;
            if (task.subtasks) {
                const found = findTaskInTree(taskId, task.subtasks);
                if (found) return found;
            }
        }
        return null;
    }

    async function updateTask(taskId: string, taskUpdate: TaskUpdate) {
        if (!currentRunId.value) return;
        isLoading.value = true;
        error.value = null;
        try {
            const response = await apiUpdateTask(currentRunId.value, taskId, taskUpdate);
            const updatedTaskData = response.data;
            // 在本地更新任务数据，UI会自动响应
            const taskToUpdate = findTaskInTree(taskId, tasks.value);
            if (taskToUpdate) {
                Object.assign(taskToUpdate, updatedTaskData);
            }
        } catch (err) {
            error.value = `更新任务 ${taskId} 失败`;
            console.error(error.value, err);
            throw err;
        } finally {
            isLoading.value = false;
        }
    }

    async function runTask(taskId: string): Promise<TaskRunResponse | undefined> {
        if (!currentRunId.value) return;
        // 运行任务通常是异步的，可以只改变状态，不阻塞整体 isLoading
        const task = findTaskInTree(taskId, tasks.value);
        if (task) task.status = 'running';
        error.value = null;
        try {
            const response = await apiRunTask(currentRunId.value, taskId);
            return response.data;
        } catch (err) {
            error.value = `运行任务 ${taskId} 失败`;
            console.error(error.value, err);
            if (task) task.status = 'failed'; // 运行失败，更新状态
            throw err;
        }
    }

    // 辅助函数：在树中查找并删除任务
    function findAndRemoveTask(taskId: string, taskList: Task[]): boolean {
        const index = taskList.findIndex(t => t.task_id === taskId);
        if (index !== -1) {
            taskList.splice(index, 1);
            return true;
        }
        for (const task of taskList) {
            if (task.subtasks && findAndRemoveTask(taskId, task.subtasks)) {
                return true;
            }
        }
        return false;
    }

    async function deleteTask(taskId: string) {
        if (!currentRunId.value) return;
        isLoading.value = true;
        error.value = null;
        try {
            await apiDeleteTask(currentRunId.value, taskId);
            // 优化：直接从本地树中移除，而不是重新获取所有
            findAndRemoveTask(taskId, tasks.value);
        } catch (err) {
            error.value = `删除任务 ${taskId} 失败`;
            console.error(error.value, err);
            throw err;
        } finally {
            isLoading.value = false;
        }
    }

    return { tasks, currentBook, isLoading, error, currentRunId, fetchTasksForBook, updateTask, runTask, deleteTask };
});
