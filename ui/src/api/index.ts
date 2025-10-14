// src/api/index.ts
import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8000/api', // 您的 FastAPI 后端地址
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;
