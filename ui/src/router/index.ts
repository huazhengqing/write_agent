// src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router';
import CreateBookView from '@/views/CreateBookView.vue'
import DashboardView from '@/views/DashboardView.vue';
import AllTasksView from '@/views/AllTasksView.vue';

const routes = [
  {
    path: '/create-book', // 新的“创建项目”页面路由
    name: 'create-book',
    component: CreateBookView
  },
  {
    path: '/dashboard', // 将路径修改为 /dashboard, 与 App.vue 中的菜单项匹配
    name: 'dashboard',
    component: DashboardView,
  },
  {
    path: '/', // 添加根路径重定向
    redirect: '/dashboard',
  },
  {
    path: '/tasks',
    name: 'tasks',
    component: AllTasksView,
  },
];

const router = createRouter({ history: createWebHistory(), routes });
export default router;