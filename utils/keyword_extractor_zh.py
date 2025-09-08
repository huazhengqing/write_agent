#coding: utf8
import re
import os
import math
import jieba
import hashlib
import torch
import threading
import stopwordsiso
import multiprocessing
import diskcache as dc
from typing import List
from loguru import logger
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from markdown import markdown
from collections import defaultdict


"""
# KeywordExtractorZh
- 中文关键词提取器, 基于 KeyBERT 和 jieba 库
- 支持从文本和 Markdown 中提取关键词
- 实现文本分块处理、预处理和缓存功能
- 使用 BAAI/bge-small-zh 模型进行中文语义理解

# model
中文: 
shibing624/text2vec-base-chinese（专为中文优化的通用模型）
BAAI/bge-small-zh（中文语义理解能力强, 适合长文本）
多语言: 
paraphrase-multilingual-MiniLM-L12-v2（轻量, 支持 100 + 语言）
xlm-r-bert-base-nli-stsb-mean-tokens（支持语言更多, 精度较高）
"""


###############################################################################


class KeywordExtractorZh:
    def __init__(self):
        self.model = None
        self.chunk_size = 700  # 中文字符数 (约对应 450-500 tokens)
        self.chunk_overlap = 100  # 中文字符重叠数
        self._model_lock = threading.Lock()

        threads = min(multiprocessing.cpu_count(), 4)
        jieba.enable_parallel(threads)

        self.base_stop_words = set(stopwordsiso.stopwords("zh"))

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cache_dir = os.path.join(project_root, ".cache", "keyword_extractor_zh")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = dc.Cache(cache_dir, size_limit=1024 * 1024 * 300)

    def __del__(self):
        try:
            jieba.disable_parallel()
        except:
            pass

    def _ensure_model_initialized(self):
        if not self.model:
            with self._model_lock:
                if not self.model:
                    # 构建本地模型的期望路径
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    local_model_path = os.path.join(project_root, "models", "bge-small-zh")
                    
                    # 检查是否有可用的 GPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"正在加载关键词提取模型: {local_model_path} 到设备: {device}")
                    
                    st_model = SentenceTransformer(local_model_path, device=device)
                    self.model = KeyBERT(model=st_model)

    def extract_from_text(self, text: str, top_k: int = 30) -> List[str]:
        """
        从小说正文中提取关键词
        包含层级结构（全书、卷、幕、章、场景、节拍、段落）
        KeyBERT 批处理: 
            - 输入: `docs` 参数传文本列表 `[text1, text2, ...]`
            - 返回: 嵌套列表, 每个子列表对应输入文本的关键词 `[(kw, score), ...]`
        """
        if not text or not text.strip():
            return []

        self._ensure_model_initialized()

        text_hash = hashlib.blake2b(text.encode('utf-8'), digest_size=32).hexdigest()
        cache_key = text_hash
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            chunks = self.split_long_text(text)
            all_keywords_with_scores = defaultdict(float)
            keyword_chunk_count = defaultdict(int)

            processed_chunks = []
            for chunk in chunks:
                processed_text = self.preprocess_chunk(chunk)
                if processed_text:
                    processed_chunks.append(processed_text)

            if not processed_chunks:
                return []

            # 优化KeyBERT参数, 提升中文语义理解
            batch_results = self.model.extract_keywords(
                processed_chunks,
                keyphrase_ngram_range=(1, 3),  # 扩展到3-gram, 捕获更多中文词组
                stop_words="chinese",
                use_mmr=True,
                diversity=0.7,  # 提高多样性
                top_n=min(top_k * 2, 50),  # 提取更多候选词用于后续筛选
                batch_size=16  # 减少批次大小, 提升处理稳定性
            )

            for idx, keywords_with_scores in enumerate(batch_results):
                for kw, score in keywords_with_scores:
                    all_keywords_with_scores[kw] += score
                    keyword_chunk_count[kw] += 1

            # 优化关键词权重计算算法 - 多维度评分机制
            chunk_count = len(processed_chunks)
            weighted_keywords = {}
            for kw, score in all_keywords_with_scores.items():
                # 频次因子: 平衡出现频率和过度普遍性
                freq_ratio = keyword_chunk_count[kw] / chunk_count
                if freq_ratio > 0.8:  # 过于频繁的词降权
                    freq_factor = 0.7 + 0.3 * (1 - freq_ratio)
                else:
                    freq_factor = math.sqrt(freq_ratio) * 1.2  # 适度出现的词提权
                
                # 长度因子: 偏好2-4字的关键词
                length_factor = 1.0
                kw_len = len(kw)
                if 2 <= kw_len <= 4:
                    length_factor = 1.3
                elif kw_len == 1:
                    length_factor = 0.6
                elif kw_len > 6:
                    length_factor = 0.8
                
                # 语义多样性因子: 避免过于相似的关键词
                diversity_factor = 1.0
                for existing_kw in weighted_keywords.keys():
                    # 简单的字符重叠检测
                    common_chars = set(kw) & set(existing_kw)
                    if len(common_chars) / max(len(kw), len(existing_kw)) > 0.6:
                        diversity_factor *= 0.9
                
                # 综合权重计算
                final_score = score * freq_factor * length_factor * diversity_factor
                weighted_keywords[kw] = final_score

            # 按加权分数排序
            sorted_keywords = sorted(weighted_keywords.items(), key=lambda x: x[1], reverse=True)
            final_keywords = [kw for kw, _ in sorted_keywords][:top_k]

            self.cache.set(cache_key, final_keywords)
            logger.info(f"extract_from_text() final_keywords={final_keywords}")
            return final_keywords
        except Exception as e:
            logger.error(f"Unexpected error during keyword extraction: {e}")
            return []

    def extract_from_markdown(self, markdown_content: str, top_k: int = 30) -> list[str]:
        """
        从Markdown格式的内容(小说大纲、设计方案)中提取关键词
        """
        if not markdown_content or not markdown_content.strip():
            return []

        self._ensure_model_initialized()
        
        text_content = self.clean_markdown(markdown_content)
        return self.extract_from_text(text_content, top_k)

    def clean_markdown(self, markdown_text: str) -> str:
        html = markdown(markdown_text)
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_long_text(self, text):
        # 改进的句子感知分块, 考虑段落结构
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ''
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # 对每个段落进行句子分割
            sentences = re.split(r'([。！？；;!?])', paragraph)
            
            # 合并句子时保留分隔符
            for i in range(0, len(sentences)-1, 2):
                sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                
                # 检查是否需要新建块
                estimated_length = len(current_chunk) + len(sentence) + 2  # +2 for potential separators
                
                if estimated_length > self.chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        # 智能重叠: 保留关键上下文
                        overlap_text = self._get_overlap_context(current_chunk)
                        current_chunk = overlap_text + sentence
                    else:  # 处理超长单句
                        if len(sentence) > self.chunk_size:
                            # 超长句子强制分割
                            chunks.append(sentence[:self.chunk_size])
                            current_chunk = sentence[self.chunk_size - self.chunk_overlap:]
                        else:
                            current_chunk = sentence
                else:
                    current_chunk += ('\n' if current_chunk and not current_chunk.endswith('\n') else '') + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _get_overlap_context(self, text):
        """获取重叠上下文, 优先保留完整句子"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # 尝试从末尾找到完整句子
        overlap_start = len(text) - self.chunk_overlap
        sentence_markers = ['。', '！', '？', '；', ';', '!', '?']
        
        # 向前查找句子边界
        for i in range(overlap_start, len(text)):
            if text[i] in sentence_markers:
                return text[i+1:].lstrip()
        
        # 如果没找到句子边界, 返回字符重叠
        return text[-self.chunk_overlap:].lstrip()


    def preprocess_chunk(self, chunk):
        # 清理特殊字符, 但保留关键标点
        chunk = re.sub(r'[\u3000-\u303F\uff00-\uffef\u2018-\u201f]', ' ', chunk)
        # 保留基本标点符号的语义信息
        chunk = re.sub(r'[, ,]', ' COMMA ', chunk)
        chunk = re.sub(r'[。！？]', ' PERIOD ', chunk)
        
        # 使用jieba分词
        words = jieba.lcut(chunk, cut_all=False)
        
        # 改进的过滤策略
        filtered = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            # 保留标点符号标记用于语义理解
            if w in ['COMMA', 'PERIOD']:
                filtered.append(w)
            # 过滤停用词和单字符（除非是重要单字）
            elif w not in self.base_stop_words and (len(w) > 1 or w in ['我', '你', '他', '她', '它']):
                filtered.append(w)
        
        # 移除连续的标点符号标记
        result = []
        prev_punct = False
        for w in filtered:
            if w in ['COMMA', 'PERIOD']:
                if not prev_punct:
                    result.append(w)
                prev_punct = True
            else:
                result.append(w)
                prev_punct = False
        
        return " ".join(result)


###############################################################################
