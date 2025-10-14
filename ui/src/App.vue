<template>
  <el-container class="main-layout">
    <el-aside :width="isCollapsed ? '64px' : '200px'" class="sidebar-transition">
      <el-menu
        :default-active="activeMenu"
        class="el-menu-vertical-demo"
        router
        :collapse="isCollapsed"
        :collapse-transition="false"
      >
        <div class="logo-container">
          <h3 v-show="!isCollapsed">AI 写作智能体</h3>
          <el-icon class="collapse-icon" @click="isCollapsed = !isCollapsed">
            <component :is="isCollapsed ? 'Expand' : 'Fold'" />
          </el-icon>
        </div>
        <el-menu-item index="/create-book">
          <el-icon><Edit /></el-icon>
          <span>创建项目</span>
        </el-menu-item>
        <el-menu-item index="/dashboard">
          <el-icon><Folder /></el-icon>
          <span>项目管理</span>
        </el-menu-item>
        <!-- 任务中心现在是一个独立的顶级页面 -->
        <el-menu-item index="/tasks">
          <el-icon><Tickets /></el-icon>
          <span>任务中心</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-main>
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useRoute } from 'vue-router';
import { Folder, Tickets, Fold, Expand } from '@element-plus/icons-vue';
import { Edit } from '@element-plus/icons-vue';

const route = useRoute();
const isCollapsed = ref(false);

const activeMenu = computed(() => {
  // 如果路由是任务详情页，也高亮任务中心
  if (route.path.startsWith('/tasks/')) {
    return '/tasks';
  }
  // 如果是项目详情页，也高亮项目管理
  if (route.path.startsWith('/books/')) {
    return '/dashboard';
  }
  return route.path;
});
</script>

<style>
.main-layout {
  height: 100vh;
}
.el-aside {
  background-color: #fff;
  border-right: 1px solid #e6e6e6;
}
.sidebar-transition {
  transition: width 0.3s ease;
}
.el-menu {
  border-right: none !important;
}
.logo-container {
  padding: 20px;
  text-align: center;
  font-size: 16px;
  font-weight: bold;
  color: #303133;
}
.logo-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.collapse-icon {
  cursor: pointer;
  font-size: 20px;
  color: #606266;
}
.el-menu--collapse .logo-container {
  justify-content: center;
}
</style>
