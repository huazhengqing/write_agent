#coding: utf8
import math
from collections import defaultdict
from typing import List
import stopwordsiso
from loguru import logger
from keybert import KeyBERT
from markdown import markdown
import diskcache as dc
import os
import hashlib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import multiprocessing
import threading


"""
# KeywordExtractorEn
- 英文关键词提取器，基于 KeyBERT 和 NLTK 库
- 支持从文本和 Markdown 中提取关键词
- 实现文本分块处理、预处理和缓存功能
- 使用 all-MiniLM-L6-v2 模型进行英文语义理解

# model
英文：
all-MiniLM-L6-v2（轻量高效，适合大多数场景）
all-mpnet-base-v2（精度更高，但速度稍慢）
多语言：
paraphrase-multilingual-MiniLM-L12-v2（轻量，支持 100 + 语言）
xlm-r-bert-base-nli-stsb-mean-tokens（支持语言更多，精度较高）
"""


###############################################################################


class KeywordExtractorEn:
    def __init__(self):
        self.model = None
        self.chunk_size = 1500  # 英文字符数 (约对应 450-500 tokens)
        self.chunk_overlap = 200  # 英文字符重叠数
        self._model_lock = threading.Lock()

        self._punkt_downloaded = False
        self._download_lock = threading.Lock()

        self.base_stop_words = set(stopwordsiso.stopwords("en"))

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cache_dir = os.path.join(project_root, ".cache", "keyword_extractor_zh")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = dc.Cache(cache_dir, size_limit=1024 * 1024 * 300)

    def _ensure_model_initialized(self):
        if not self.model:
            with self._model_lock:
                if not self.model:
                    # self.model = KeyBERT(model="all-MiniLM-L6-v2", nr_processes=os.cpu_count()//2)
                    self.model = KeyBERT(model="./models/all-MiniLM-L6-v2", nr_processes=os.cpu_count()//2)

    def _ensure_punkt_downloaded(self):
        if not self._punkt_downloaded:
            with self._download_lock:
                if not self._punkt_downloaded:
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True)
                    self._punkt_downloaded = True

    def extract_from_text(self, text: str, top_k: int = 30) -> List[str]:
        """
        从小说正文中提取关键词
        包含层级结构（全书、卷、幕、章、场景、节拍、段落）
        KeyBERT 批处理：
            - 输入：`docs` 参数传文本列表 `[text1, text2, ...]`
            - 返回：嵌套列表，每个子列表对应输入文本的关键词 `[(kw, score), ...]`
        """
        if not text or not text.strip():
            return []
        self._ensure_model_initialized()
        self._ensure_punkt_downloaded()

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

            # 优化KeyBERT参数，提升英文语义理解
            batch_results = self.model.extract_keywords(
                processed_chunks,
                keyphrase_ngram_range=(1, 4),  # 扩展到4-gram，适合英文词组
                stop_words="english",
                use_mmr=True,
                diversity=0.75,  # 较高的多样性
                top_n=min(top_k * 3, 60),  # 提取更多候选词用于后续筛选
                batch_size=16  # 减少批次大小，提升处理稳定性
            )

            for idx, keywords_with_scores in enumerate(batch_results):
                for kw, score in keywords_with_scores:
                    all_keywords_with_scores[kw] += score
                    keyword_chunk_count[kw] += 1

            # 优化关键词权重计算算法 - 多维度评分机制
            chunk_count = len(processed_chunks)
            weighted_keywords = {}
            for kw, score in all_keywords_with_scores.items():
                # 频次因子：平衡出现频率和过度普遍性
                freq_ratio = keyword_chunk_count[kw] / chunk_count
                if freq_ratio > 0.8:  # 过于频繁的词降权
                    freq_factor = 0.7 + 0.3 * (1 - freq_ratio)
                else:
                    freq_factor = math.sqrt(freq_ratio) * 1.2  # 适度出现的词提权
                
                # 长度因子：偏好中等长度的英文词组
                length_factor = 1.0
                words = kw.split()
                word_count = len(words)
                if word_count == 1:
                    word_len = len(kw)
                    if 3 <= word_len <= 8:  # 单词适中长度
                        length_factor = 1.2
                    elif word_len < 3:  # 过短单词
                        length_factor = 0.7
                    elif word_len > 12:  # 过长单词
                        length_factor = 0.8
                elif 2 <= word_count <= 3:  # 首选词组长度
                    length_factor = 1.4
                elif word_count > 4:  # 过长词组
                    length_factor = 0.6
                
                # 语义多样性因子：避免过于相似的关键词
                diversity_factor = 1.0
                kw_lower = kw.lower()
                for existing_kw in weighted_keywords.keys():
                    existing_lower = existing_kw.lower()
                    # 检查子串包含和单词重叠
                    if (kw_lower in existing_lower or existing_lower in kw_lower or
                        len(set(kw_lower.split()) & set(existing_lower.split())) / 
                        max(len(kw_lower.split()), len(existing_lower.split())) > 0.5):
                        diversity_factor *= 0.85
                
                # 英文大小写一致性加分
                case_factor = 1.0
                if kw.istitle() or (word_count > 1 and all(w.istitle() for w in words)):  # 正规大小写
                    case_factor = 1.1
                
                # 综合权重计算
                final_score = score * freq_factor * length_factor * diversity_factor * case_factor
                weighted_keywords[kw] = final_score

            # 按加权分数排序
            sorted_keywords = sorted(weighted_keywords.items(), key=lambda x: x[1], reverse=True)
            final_keywords = [kw for kw, _ in sorted_keywords][:top_k]

            self.cache.set(cache_key, final_keywords)
            logger.info(f"extract_from_text() final_keywords={final_keywords}")
            return final_keywords
        except ValueError as e:
            logger.error(f"Invalid value during keyword extraction: {e}")
            return []
        except RuntimeError as e:
            logger.error(f"Runtime error during model inference: {e}")
            return []
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
        self._ensure_punkt_downloaded()
        text_content = self.clean_markdown(markdown_content)
        return self.extract_from_text(text_content, top_k)

    def clean_markdown(self, markdown_text: str) -> str:
        html = markdown(markdown_text)
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_long_text(self, text):
        # 改进的段落和句子感知分块
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ''
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # 使用NLTK进行句子分割
            sentences = sent_tokenize(paragraph.strip())
            
            for sentence in sentences:
                # 计算包含当前句子的预估长度
                estimated_length = len(current_chunk) + len(sentence) + 2  # +2 for separators
                
                if estimated_length > self.chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        # 智能重叠：保留关键上下文
                        overlap_text = self._get_overlap_context(current_chunk)
                        current_chunk = overlap_text + ' ' + sentence
                    else:  # 处理超长单句
                        if len(sentence) > self.chunk_size:
                            # 超长句子按单词边界分割
                            words = sentence.split()
                            chunk_words = []
                            current_length = 0
                            
                            for word in words:
                                if current_length + len(word) + 1 > self.chunk_size:
                                    if chunk_words:
                                        chunks.append(' '.join(chunk_words))
                                        # 保留重叠单词
                                        overlap_count = min(self.chunk_overlap // 10, len(chunk_words) // 3)
                                        chunk_words = chunk_words[-overlap_count:] + [word]
                                        current_length = sum(len(w) + 1 for w in chunk_words)
                                    else:
                                        chunks.append(word)
                                        chunk_words = []
                                        current_length = 0
                                else:
                                    chunk_words.append(word)
                                    current_length += len(word) + 1
                            
                            if chunk_words:
                                current_chunk = ' '.join(chunk_words)
                            else:
                                current_chunk = ''
                        else:
                            current_chunk = sentence
                else:
                    current_chunk += ('\n' if current_chunk and not current_chunk.endswith('\n') else '') + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _get_overlap_context(self, text):
        """获取重叠上下文，优先保留完整句子"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # 尝试从末尾找到完整句子
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            # 如果只有一个句子，按单词返回末尾部分
            words = text.split()
            overlap_words = min(self.chunk_overlap // 10, len(words) // 3)
            return ' '.join(words[-overlap_words:]) if overlap_words > 0 else ''
        
        # 从末尾选择适合的句子数量
        overlap_text = ''
        for i in range(len(sentences) - 1, -1, -1):
            candidate = ' '.join(sentences[i:])
            if len(candidate) <= self.chunk_overlap:
                overlap_text = candidate
            else:
                break
        
        return overlap_text if overlap_text else text[-self.chunk_overlap:].lstrip()

    def preprocess_chunk(self, chunk):
        # 保留基本标点符号的语义信息
        chunk = re.sub(r'[,]', ' COMMA ', chunk)
        chunk = re.sub(r'[.!?]', ' PERIOD ', chunk)
        chunk = re.sub(r'[;:]', ' SEMICOLON ', chunk)
        
        # 提取单词，包括连字符单词
        words = re.findall(r'\b[\w-]{2,}\b|COMMA|PERIOD|SEMICOLON', chunk, flags=re.IGNORECASE)
        
        # 改进的过滤策略
        filtered = []
        prev_punct = False
        
        for w in words:
            w_lower = w.lower()
            
            # 保留标点符号标记用于语义理解
            if w in ['COMMA', 'PERIOD', 'SEMICOLON']:
                if not prev_punct:  # 避免连续标点
                    filtered.append(w)
                prev_punct = True
            # 过滤停用词和过短单词
            elif w_lower not in self.base_stop_words and len(w) >= 2:
                # 保留大写开头的单词（可能是专有名词）
                if w[0].isupper() or w_lower not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'has', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say']:
                    filtered.append(w_lower if not w[0].isupper() else w)
                prev_punct = False
        
        return ' '.join(filtered)


###############################################################################




