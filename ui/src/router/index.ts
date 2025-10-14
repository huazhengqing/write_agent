// src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router';
import DashboardView from '@/views/DashboardView.vue';
import AllTasksView from '@/views/AllTasksView.vue';

const routes = [
  {
    path: '/',
    name: 'dashboard',
    component: DashboardView,
  },
  {
    path: '/tasks',
    name: 'tasks',
    component: AllTasksView,
  },
];

const router = createRouter({ history: createWebHistory(), routes });
export default router;