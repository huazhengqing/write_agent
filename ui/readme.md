# Vue 3 + TypeScript + Vite

This template should help get you started developing with Vue 3 and TypeScript in Vite. The template uses Vue 3 `<script setup>` SFCs, check out the [script setup docs](https://v3.vuejs.org/api/sfc-script-setup.html#sfc-script-setup) to learn more.

Learn more about the recommended Project Setup and IDE Support in the [Vue Docs TypeScript Guide](https://vuejs.org/guide/typescript/overview.html#project-setup).



Vue 3 + Vite + Element Plus
框架 (Framework): Vue 3
优势: 渐进式框架，学习曲线平缓，文档优秀。其响应式系统非常直观，处理复杂的状态变化（如任务状态更新）心智负担小。Composition API 使得组织复杂逻辑（如任务轮询）更为清晰。
为什么适合本项目: 对于任务树和详情面板这种联动更新的场景，Vue 的数据驱动视图理念能极大地简化开发。
构建工具 (Build Tool): Vite
优势: 极速的开发服务器启动和热模块更新 (HMR)，开发体验远超传统工具。由 Vue 作者开发，与 Vue 生态无缝集成。
UI 组件库 (UI Library): Element Plus
优势: 一套为 Vue 3 设计的高质量企业级组件库。提供了所有你需要的基础组件，如表单、按钮、弹窗、表格等。
关键组件: 它包含了开箱即用的 Tree (树形控件) 组件，完美契合我们展示层级任务的需求。还有 Loading、Notification 等组件可以很好地反馈异步操作的状态。
状态管理 (State Management): Pinia
优势: Vue 官方推荐的新一代状态管理库，轻量、易用，对 TypeScript 支持极佳，且能与 Vue Devtools 完美集成。
为什么需要: 用于管理全局状态，例如当前选中的书籍 run_id，以及所有任务的状态，方便在不同组件间共享和响应。
HTTP 请求库 (HTTP Client): Axios
优势: 成熟稳定，支持 Promise，可以方便地封装 API 请求和处理拦截器（例如，统一处理 API 错误）。