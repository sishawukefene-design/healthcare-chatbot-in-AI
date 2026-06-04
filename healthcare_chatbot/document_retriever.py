import json
import re
import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize
from advanced_preprocessor import TextPreprocessor, PreprocessorConfig
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ==================== DATA CLASSES ====================
@dataclass
class Document:
    """Represents a healthcare document"""
    index: int
    topic: str
    content: str
    sentences: List[str] = field(default_factory=list)
    sentence_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    """Represents a search result"""
    answer: str
    main_sentence: str
    topic: str
    score: float
    confidence: str
    document_index: int
    sentence_index: int
    keywords: List[str]
    context_range: Tuple[int, int]
    retrieval_time_ms: float

@dataclass
class RetrievalConfig:
    """Configuration for document retriever"""
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_max_features: int = 5000
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.85
    tfidf_sublinear_tf: bool = True
    tfidf_use_idf: bool = True
    similarity_threshold: float = 0.12
    similarity_metric: str = "cosine"  # cosine, euclidean, manhattan
    context_size: int = 1  # Number of neighboring sentences
    max_answer_length: int = 500  # Maximum answer length in characters
    enable_caching: bool = True
    cache_size: int = 1000
    enable_dimensionality_reduction: bool = False
    svd_components: int = 100
    batch_processing_size: int = 50

    # Ranking configuration
    enable_keyword_boosting: bool = True
    keyword_boost_factor: float = 0.2
    enable_length_normalization: bool = True

    # Advanced features
    enable_query_expansion: bool = False
    enable_multiple_answers: bool = True
    max_answers: int = 3

    def validate(self) -> bool:
        """Validate configuration parameters"""
        assert 0 <= self.similarity_threshold <= 1, "Threshold must be between 0 and 1"
        assert self.context_size >= 0, "Context size must be non-negative"
        assert self.tfidf_max_features > 0, "Max features must be positive"
        return True

# ==================== ADVANCED DOCUMENT RETRIEVER ====================
class DocumentRetriever:
    def __init__(self, json_file_path: str, config: Optional[RetrievalConfig] = None):

        self.json_file_path = json_file_path
        self.config = config or RetrievalConfig()
        self.config.validate()
        # Initialize components
        self.preprocessor = self._initialize_preprocessor()

        # Data structures
        self.documents: List[Document] = []
        self.sentences: List[str] = []
        self.sentence_to_doc: List[Dict[str, Any]] = []

        # Vector space
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.svd: Optional[TruncatedSVD] = None

        # Cache for query results
        self._query_cache = {}

        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'avg_retrieval_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'documents_loaded': 0,
            'sentences_indexed': 0
        }

        # Load and index data
        self.load_data()
        self.build_vector_space()

        logger.info(f"DocumentRetriever initialized with {len(self.documents)} documents, "
                    f"{len(self.sentences)} sentences")

    def _initialize_preprocessor(self) -> TextPreprocessor:
        """
        Initialize the advanced text preprocessor with optimal settings.

        Returns:
            Configured TextPreprocessor instance
        """
        preprocessor_config = PreprocessorConfig(
            enable_caching=True,
            extract_bigrams=True,
            prefer_lemmatization_over_stemming=True,
            extract_medical_entities=True,
            parallel_processing=True
        )
        return TextPreprocessor(preprocessor_config)

    def load_data(self):

        start_time = time.time()
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Handle different JSON formats
            if isinstance(data, list):
                documents_raw = data
            elif isinstance(data, dict):
                # Try common keys for document array
                for key in ['healthcare_data', 'documents', 'data', 'entries']:
                    if key in data:
                        documents_raw = data[key]
                        break
                else:
                    # If no key found, assume first key contains the array
                    documents_raw = next(iter(data.values())) if data else []
            else:
                raise Exception(f"Unsupported JSON format: {type(data)}")
            # Validate and load documents
            if not documents_raw:
                raise Exception("No documents found in JSON file")

            for idx, doc_data in enumerate(documents_raw):
                # Extract topic and content with fallbacks
                topic = doc_data.get('topic', doc_data.get('title', f"Document {idx + 1}"))
                content = doc_data.get('content', doc_data.get('text', doc_data.get('description', '')))
                if not content:
                    logger.warning(f"Document {idx} has no content, skipping")
                    continue
                # Create document object
                document = Document(
                    index=idx,
                    topic=topic,
                    content=content
                )
                # Tokenize content into sentences
                try:
                    sentences = sent_tokenize(content)
                except Exception:
                    # Fallback sentence tokenization
                    sentences = [s.strip() + '.' for s in content.split('.') if s.strip()]
                    sentences = [s for s in sentences if len(s) > 10]
                if not sentences:
                    sentences = [content]
                document.sentences = sentences
                self.documents.append(document)
                # Index sentences
                for sent_idx, sentence in enumerate(sentences):
                    if sentence.strip():
                        self.sentences.append(sentence)
                        self.sentence_to_doc.append({
                            'doc_index': idx,
                            'sentence_index': sent_idx,
                            'topic': topic,
                            'full_content': content,
                            'sentence_count': len(sentences)
                        })

                document.sentence_indices = list(range(
                    len(self.sentences) - len(sentences),
                    len(self.sentences)
                ))
                # Extract metadata
                document.metadata = {
                    'sentence_count': len(sentences),
                    'content_length': len(content),
                    'word_count': len(content.split())
                }
            self.metrics['documents_loaded'] = len(self.documents)
            self.metrics['sentences_indexed'] = len(self.sentences)
            load_time = (time.time() - start_time) * 1000
            logger.info(
                f"Loaded {len(self.documents)} documents and {len(self.sentences)} sentences in {load_time:.2f}ms")
        except FileNotFoundError:
            raise Exception(f"JSON file not found: {self.json_file_path}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {e}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    def build_vector_space(self):
        start_time = time.time()
        # Preprocess sentences in batches
        logger.info(f"Preprocessing {len(self.sentences)} sentences...")
        # Use batch processing for efficiency
        preprocessed_sentences = self.preprocessor.batch_preprocess(self.sentences)
        # Create TF-IDF vectorizer with optimal parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.config.tfidf_ngram_range,
            max_features=self.config.tfidf_max_features,
            min_df=self.config.tfidf_min_df,
            max_df=self.config.tfidf_max_df,
            sublinear_tf=self.config.tfidf_sublinear_tf,
            use_idf=self.config.tfidf_use_idf,
            norm='l2'
        )
        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_sentences)
        # Apply dimensionality reduction if enabled
        if self.config.enable_dimensionality_reduction:
            n_components = min(
                self.config.svd_components,
                self.tfidf_matrix.shape[1] - 1,
                self.tfidf_matrix.shape[0] - 1
            )

            if n_components > 0:
                logger.info(f"Applying dimensionality reduction to {n_components} components...")
                self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                self.tfidf_matrix = self.svd.fit_transform(self.tfidf_matrix)
                self.tfidf_matrix = normalize(self.tfidf_matrix)

        build_time = (time.time() - start_time) * 1000

        logger.info(f"✓ TF-IDF vector space created with {self.tfidf_matrix.shape[1]} features "
                    f"in {build_time:.2f}ms")

    def _expand_query(self, query: str) -> List[str]:
        if not self.config.enable_query_expansion:
            return [query]

        expansions = [query]

        # Extract medical entities and use as expansion terms
        entities = self.preprocessor.extract_medical_entities(query)
        for category, terms in entities.items():
            if terms:
                expansion = query + " " + " ".join(terms[:2])
                expansions.append(expansion)

        # Extract keywords and add them as expansion
        keywords = self.preprocessor.extract_keywords(query, top_n=3, method="rake")
        if keywords:
            expansion = query + " " + " ".join(keywords)
            expansions.append(expansion)

        return expansions[:3]  # Limit expansions

    def _calculate_confidence_level(self, score: float) -> str:
        if score >= 0.6:
            return "Very High"
        elif score >= 0.4:
            return "High"
        elif score >= 0.25:
            return "Medium"
        elif score >= 0.15:
            return "Low"
        else:
            return "Very Low"

    def _apply_boosting(self, base_score: float, query: str, sentence: str, topic: str) -> float:
        boosted_score = base_score

        # Keyword boosting
        if self.config.enable_keyword_boosting:
            query_keywords = self.preprocessor.extract_keywords(query, top_n=5)
            if query_keywords:
                keyword_match_count = sum(1 for kw in query_keywords if kw in sentence.lower())
                keyword_boost = (keyword_match_count / len(query_keywords)) * self.config.keyword_boost_factor
                boosted_score += keyword_boost

        # Topic relevance boost
        query_words = set(query.lower().split())
        topic_words = set(topic.lower().split())
        topic_overlap = len(query_words & topic_words) / max(len(query_words), 1)
        boosted_score += topic_overlap * 0.1

        # Length normalization (penalize very short answers)
        if self.config.enable_length_normalization:
            if len(sentence) < 20:
                boosted_score *= 0.8

        return min(boosted_score, 1.0)  # Cap at 1.0

    def _get_enhanced_context(self, sentence_idx: int) -> str:
        context_size = self.config.context_size
        start = max(0, sentence_idx - context_size)
        end = min(len(self.sentences), sentence_idx + context_size + 1)

        current_doc_idx = self.sentence_to_doc[sentence_idx]['doc_index']
        context_sentences = []

        # Get context from same document
        for i in range(start, end):
            if i < len(self.sentence_to_doc):
                doc_info = self.sentence_to_doc[i]
                if doc_info['doc_index'] == current_doc_idx:
                    context_sentences.append(self.sentences[i])

        # Join with proper spacing
        context = ' '.join(context_sentences)

        # Clean up context
        context = re.sub(r'\s+', ' ', context).strip()

        # Limit length if needed
        if len(context) > self.config.max_answer_length:
            # Truncate at sentence boundary
            truncated = context[:self.config.max_answer_length]
            last_period = truncated.rfind('.')
            if last_period > 0:
                context = truncated[:last_period + 1]
        return context
    @lru_cache(maxsize=1000)
    def _cached_query_process(self, query: str) -> Tuple[np.ndarray, List[str]]:
        expanded_queries = self._expand_query(query)

        # Process all query variations
        vectors = []
        for eq in expanded_queries:
            processed = self.preprocessor.preprocess(eq)
            vector = self.tfidf_vectorizer.transform([processed])

            # Apply SVD if enabled
            if self.svd is not None:
                vector = self.svd.transform(vector)
                vector = normalize(vector)
            vectors.append(vector)
        # Combine vectors (average for multiple expansions)
        if len(vectors) > 1:
            combined = np.mean(vectors, axis=0)
        else:
            combined = vectors[0]

        return combined, expanded_queries

    def find_best_answer(self, query: str) -> Tuple[Optional[SearchResult], float, Optional[str]]:
        start_time = time.time()
        self.metrics['total_queries'] += 1

        # Check cache
        cache_key = query.lower().strip()
        if self.config.enable_caching and cache_key in self._query_cache:
            self.metrics['cache_hits'] += 1
            cached_result = self._query_cache[cache_key]
            retrieval_time = (time.time() - start_time) * 1000
            self._update_metrics(retrieval_time)
            return cached_result

        self.metrics['cache_misses'] += 1

        # Handle empty query
        if not query or not query.strip():
            return None, 0.0, None
        try:
            query_vector, expanded_queries = self._cached_query_process(query)
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[-min(self.config.max_answers * 2, len(similarities)):][::-1]
            best_result = None
            best_score = 0.0
            best_topic = None
            for idx in top_indices:
                score = similarities[idx]
                if score < self.config.similarity_threshold:
                    continue
                source_info = self.sentence_to_doc[idx]
                sentence = self.sentences[idx]
                topic = source_info['topic']
                boosted_score = self._apply_boosting(score, query, sentence, topic)
                if boosted_score > best_score:
                    best_score = boosted_score
                    best_topic = topic
                    # Get enhanced context
                    context = self._get_enhanced_context(idx)
                    keywords = self.preprocessor.extract_keywords(query, top_n=3, method="rake")
                    # Get context range
                    start_ctx = max(0, idx - self.config.context_size)
                    end_ctx = min(len(self.sentences), idx + self.config.context_size + 1)
                    best_result = SearchResult(
                        answer=context,
                        main_sentence=sentence,
                        topic=topic,
                        score=best_score,
                        confidence=self._calculate_confidence_level(best_score),
                        document_index=source_info['doc_index'],
                        sentence_index=source_info['sentence_index'],
                        keywords=keywords,
                        context_range=(start_ctx, end_ctx),
                        retrieval_time_ms=0.0  # Will be set below
                    )
            # Cache the result
            retrieval_time = (time.time() - start_time) * 1000
            if best_result:
                best_result.retrieval_time_ms = retrieval_time
                result_tuple = (best_result, best_score, best_topic)
            else:
                result_tuple = (None, best_score, None)

            if self.config.enable_caching:
                self._query_cache[cache_key] = result_tuple
                # Limit cache size
                if len(self._query_cache) > self.config.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]

            self._update_metrics(retrieval_time)
            return result_tuple
        except Exception as e:
            logger.error(f"Error finding best answer: {e}")
            retrieval_time = (time.time() - start_time) * 1000
            self._update_metrics(retrieval_time)
            return None, 0.0, None

    def _update_metrics(self, retrieval_time_ms: float):
        total_time = self.metrics['avg_retrieval_time_ms'] * (self.metrics['total_queries'] - 1)
        total_time += retrieval_time_ms
        self.metrics['avg_retrieval_time_ms'] = total_time / self.metrics['total_queries']

    def get_multiple_answers(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self.config.enable_multiple_answers:
            result, _, _ = self.find_best_answer(query)
            return [result] if result else []

        results = []
        processed_query = self.preprocessor.preprocess(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])

        if self.svd is not None:
            query_vector = self.svd.transform(query_vector)
            query_vector = normalize(query_vector)

        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        for idx in top_indices:
            score = similarities[idx]
            if score < self.config.similarity_threshold:
                continue

            source_info = self.sentence_to_doc[idx]
            context = self._get_enhanced_context(idx)
            keywords = self.preprocessor.extract_keywords(query, top_n=3, method="rake")

            result = SearchResult(
                answer=context,
                main_sentence=self.sentences[idx],
                topic=source_info['topic'],
                score=score,
                confidence=self._calculate_confidence_level(score),
                document_index=source_info['doc_index'],
                sentence_index=source_info['sentence_index'],
                keywords=keywords,
                context_range=(max(0, idx - 1), min(len(self.sentences), idx + 2)),
                retrieval_time_ms=0.0
            )
            results.append(result)

        return results

    def get_fallback_response(self, query: str) -> str:
        # Extract keywords for better suggestions
        keywords = self.preprocessor.extract_keywords(query, top_n=3, method="rake")
        medical_entities = self.preprocessor.extract_medical_entities(query)
        if keywords or medical_entities:
            all_terms = keywords + list(medical_entities.keys())
            terms_str = ', '.join(all_terms[:3])
            fallback_messages = [
                f"I'm sorry, I couldn't find specific information about {terms_str}. "
                f"Please rephrase your question or consult a healthcare professional.",

                f"I don't have enough information about {terms_str}. "
                f"Could you ask about symptoms, treatments, or prevention methods?",

                f"I apologize, but I need more details about {terms_str}. "
                f"Try asking a different health-related question."
            ]
        else:
            fallback_messages = [
                "I'm sorry, I couldn't find information matching your query. "
                "Please rephrase your question or ask about specific diseases, symptoms, or treatments.",

                "I apologize, but I don't have enough information to answer that accurately. "
                "Could you please ask about symptoms, diseases, treatments, or prevention methods?",

                "I'm still learning! Could you provide more details or ask a different health-related question?"
            ]
        import random
        return random.choice(fallback_messages)
    def get_keyword_suggestions(self, query: str) -> str:
        keywords = self.preprocessor.extract_keywords(query, top_n=3, method="rake")
        if keywords:
            return f"Based on your query about {', '.join(keywords)}, "
        return ""
    def get_document_by_index(self, doc_index: int) -> Optional[Document]:
        if 0 <= doc_index < len(self.documents):
            return self.documents[doc_index]
        return None
    def search_by_topic(self, topic_keyword: str) -> List[Document]:
        topic_keyword = topic_keyword.lower()
        matches = []
        for doc in self.documents:
            if topic_keyword in doc.topic.lower():
                matches.append(doc)
        return matches
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'documents': {
                'total': len(self.documents),
                'total_sentences': len(self.sentences),
                'avg_sentences_per_doc': len(self.sentences) / len(self.documents) if self.documents else 0
            },
            'vector_space': {
                'dimensions': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
                'sparsity': (1 - (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[
                    1]))) * 100 if self.tfidf_matrix is not None else 0
            },
            'performance': self.metrics.copy(),
            'config': {
                'similarity_threshold': self.config.similarity_threshold,
                'context_size': self.config.context_size,
                'caching_enabled': self.config.enable_caching
            }
        }
        # Add preprocessor stats
        stats['preprocessor'] = self.preprocessor.get_performance_stats()
        return stats
    def clear_cache(self):
        """Clear all caches."""
        self._query_cache.clear()
        self._cached_query_process.cache_clear()
        self.preprocessor.reset_cache()
        logger.info("All caches cleared")
    def reload_data(self):
        """Reload data from JSON file and rebuild vector space."""
        logger.info("Reloading data...")
        self.documents.clear()
        self.sentences.clear()
        self.sentence_to_doc.clear()
        self._query_cache.clear()
        self.load_data()
        self.build_vector_space()
        logger.info("Data reloaded successfully")
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"DocumentRetriever(documents={len(self.documents)}, "
                f"sentences={len(self.sentences)}, "
                f"features={self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0})")
