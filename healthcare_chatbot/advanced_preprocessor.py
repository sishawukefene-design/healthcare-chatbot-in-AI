
import re
import nltk
import hashlib
import functools
import logging
import time
from typing import List, Set, Dict, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from enum import Enum
import warnings

warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
try:
    from nltk.tag import pos_tag

    POS_TAGGING_AVAILABLE = True
except ImportError:
    POS_TAGGING_AVAILABLE = False
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== ENUMS AND CONFIGURATION ====================

class StemmingAlgorithm(Enum):
    """Enumeration of available stemming algorithms"""
    PORTER = "porter"
    SNOWBALL = "snowball"
    NONE = "none"

class LemmatizationStrategy(Enum):
    """Enumeration of lemmatization strategies"""
    WORDNET = "wordnet"
    NONE = "none"

@dataclass
class PreprocessorConfig:
    remove_numbers: bool = True
    remove_special_chars: bool = True
    remove_extra_spaces: bool = True
    lowercase: bool = True
    remove_emojis: bool = True
    min_token_length: int = 2
    max_token_length: int = 35
    preserve_acronyms: bool = True
    use_general_stopwords: bool = True
    use_medical_stopwords: bool = True
    use_custom_stopwords: bool = True
    custom_stopwords: Set[str] = field(default_factory=lambda: {
        'like', 'would', 'could', 'also', 'however', 'therefore',
        'hence', 'thus', 'accordingly', 'consequently', 'furthermore',
        'moreover', 'nevertheless', 'nonetheless'
    })

    medical_stopwords: Set[str] = field(default_factory=lambda: {
        'patient', 'doctor', 'hospital', 'clinic', 'medical',
        'healthcare', 'treatment', 'condition', 'disease'
    })

    stemming_algorithm: StemmingAlgorithm = StemmingAlgorithm.SNOWBALL
    lemmatization_strategy: LemmatizationStrategy = LemmatizationStrategy.WORDNET
    prefer_lemmatization_over_stemming: bool = True

    extract_bigrams: bool = True
    extract_trigrams: bool = False
    extract_medical_entities: bool = True
    enable_caching: bool = True
    cache_size: int = 2000
    parallel_processing: bool = True
    max_workers: int = max(2, cpu_count() - 1)
    chunk_size: int = 100
    enable_pos_tagging: bool = False
    medical_terms_preserve: Set[str] = field(default_factory=lambda: {
        # Diseases
        'diabetes', 'hypertension', 'asthma', 'covid', 'influenza',
        'pneumonia', 'tuberculosis', 'hepatitis', 'cancer', 'arthritis',
        'osteoarthritis', 'migraine', 'depression', 'insomnia', 'gerd',
        'stroke', 'seizure', 'anemia', 'leukemia', 'melanoma',

        # Medications
        'antibiotic', 'antihistamine', 'corticosteroid', 'analgesic',
        'antidepressant', 'antipsychotic', 'antiviral', 'antifungal',
        'insulin', 'metformin', 'lisinopril', 'atorvastatin',

        # Medical procedures
        'surgery', 'biopsy', 'endoscopy', 'colonoscopy', 'mammogram',
        'vaccination', 'immunization', 'transplant', 'dialysis',

        # Medical terms
        'symptom', 'symptoms', 'treatment', 'treatments', 'prevention',
        'diagnosis', 'therapy', 'prognosis', 'etiology', 'pathology'
    })
    def validate(self) -> bool:
        assert self.min_token_length >= 1, "min_token_length must be >= 1"
        assert self.max_token_length > self.min_token_length, "max_token_length must be > min_token_length"
        assert self.cache_size > 0, "cache_size must be > 0"
        assert self.max_workers > 0, "max_workers must be > 0"
        return True

# ==================== ADVANCED TEXT PREPROCESSOR ====================
class TextPreprocessor:
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self.config.validate()
        self._initialize_nltk_resources()
        self._initialize_processors()
        self._initialize_stopwords()
        self._initialize_cache()
        self.stats = {
            'total_processed': 0,
            'total_processing_time_ms': 0.0,
            'avg_processing_time_ms': 0.0,
            'errors': 0
        }

        logger.info(f"TextPreprocessor initialized with lemmatization={self.config.lemmatization_strategy.value}, "
                    f"caching={self.config.enable_caching}")

    def _initialize_nltk_resources(self):
        """Initialize all NLTK resources with proper error handling."""
        resources = [
            ('punkt', 'tokenizers/punkt'),
            ('punkt_tab', 'tokenizers/punkt_tab'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet')
        ]

        for resource_name, resource_path in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info(f"Downloading {resource_name}...")
                nltk.download(resource_name, quiet=True)
    def _initialize_processors(self):
        if self.config.stemming_algorithm == StemmingAlgorithm.PORTER:
            self.stemmer = PorterStemmer()
        elif self.config.stemming_algorithm == StemmingAlgorithm.SNOWBALL:
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = None

        # Lemmatizer
        if self.config.lemmatization_strategy == LemmatizationStrategy.WORDNET:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
    def _initialize_stopwords(self):
        """Build comprehensive stopwords set."""
        self.stop_words = set()
        # General English stopwords
        if self.config.use_general_stopwords:
            try:
                self.stop_words.update(stopwords.words('english'))
            except:
                nltk.download('stopwords', quiet=True)
                self.stop_words.update(stopwords.words('english'))
        if self.config.use_medical_stopwords:
            self.stop_words.update(self.config.medical_stopwords)
        if self.config.use_custom_stopwords:
            self.stop_words.update(self.config.custom_stopwords)

        logger.debug(f"Initialized {len(self.stop_words)} stopwords")

    def _initialize_cache(self):
        """Initialize LRU cache for preprocessing."""
        if self.config.enable_caching:
            self._full_preprocess_impl_cached = functools.lru_cache(
                maxsize=self.config.cache_size
            )(self._full_preprocess_impl)
            self.preprocess = self._preprocess_with_cache
        else:
            self.preprocess = self._preprocess_without_cache

    def _get_wordnet_pos(self, word: str) -> str:
        if not self.config.enable_pos_tagging or not POS_TAGGING_AVAILABLE:
            return wordnet.NOUN
        try:
            tag = pos_tag([word])[0][1][0].upper()
            tag_dict = {
                'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'R': wordnet.ADV,
                'V': wordnet.VERB
            }
            return tag_dict.get(tag, wordnet.NOUN)
        except:
            return wordnet.NOUN
    # ==================== TEXT CLEANING METHODS ====================
    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        if self.config.remove_emojis:
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags
                                       "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        if self.config.lowercase:
            text = text.lower()
        if self.config.remove_special_chars:
            if self.config.remove_numbers:
                text = re.sub(r'[^a-zA-Z\s]', '', text)
            else:
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        if self.config.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        return text
    # ==================== TOKENIZATION METHODS ====================
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        tokens = [
            token for token in tokens
            if self.config.min_token_length <= len(token) <= self.config.max_token_length
        ]
        if self.config.preserve_acronyms:
            acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
            tokens.extend([acronym.lower() for acronym in acronyms])
        return tokens
    # ==================== STOPWORD REMOVAL ====================
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        filtered = []
        for token in tokens:
            if token in self.config.medical_terms_preserve:
                filtered.append(token)
            elif token not in self.stop_words:
                filtered.append(token)
        return filtered
    # ==================== STEMMING AND LEMMATIZATION ====================
    def lemmatize(self, tokens: List[str]) -> List[str]:
        if not self.lemmatizer:
            return tokens
        lemmatized = []
        for token in tokens:
            pos = self._get_wordnet_pos(token)
            lemmatized_token = self.lemmatizer.lemmatize(token, pos)
            lemmatized.append(lemmatized_token)
        return lemmatized
    def stem(self, tokens: List[str]) -> List[str]:
        if not self.stemmer:
            return tokens
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return stemmed

    # ==================== N-GRAM EXTRACTION ====================
    def extract_ngrams(self, tokens: List[str]) -> List[str]:
        ngrams = list(tokens)
        if self.config.extract_bigrams and len(tokens) >= 2:
            bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
            ngrams.extend(bigrams)
        if self.config.extract_trigrams and len(tokens) >= 3:
            trigrams = [f"{tokens[i]}_{tokens[i + 1]}_{tokens[i + 2]}" for i in range(len(tokens) - 2)]
            ngrams.extend(trigrams)
        return ngrams
    # =================== MEDICAL ENTITY EXTRACTION ====================
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        if not self.config.extract_medical_entities:
            return {}
        entities = defaultdict(list)
        # Define patterns for different medical entities
        patterns = {
            'symptoms': r'\b(symptom|sign|pain|ache|swelling|fever|cough|fatigue|nausea|dizziness|headache|rash|vomiting|diarrhea|constipation)\b',
            'diseases': r'\b(diabetes|hypertension|asthma|cancer|arthritis|pneumonia|hepatitis|tuberculosis|malaria|influenza|covid|stroke|seizure|depression|insomnia)\b',
            'treatments': r'\b(treatment|therapy|surgery|medication|drug|prescription|injection|vaccination|chemotherapy|physical therapy)\b',
            'medications': r'\b(antibiotic|antihistamine|analgesic|corticosteroid|insulin|statin|antidepressant|antiviral|antifungal)\b',
            'preventions': r'\b(prevention|preventive|vaccination|immunization|screening|diet|exercise|lifestyle)\b'
        }
        text_lower = text.lower()
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities[category].extend(list(set(matches)))
        # Remove duplicates
        for category in entities:
            entities[category] = list(set(entities[category]))
        return dict(entities)

    # ==================== RAKE KEYWORD EXTRACTION ====================
    def extract_keywords_rake(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        sentences = sent_tokenize(text)
        phrase_candidates = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            current_phrase = []
            for word in words:
                if word not in self.stop_words and len(word) >= 3:
                    current_phrase.append(word)
                else:
                    if len(current_phrase) > 0:
                        phrase_candidates.append(' '.join(current_phrase))
                        current_phrase = []
            if len(current_phrase) > 0:
                phrase_candidates.append(' '.join(current_phrase))
        # Calculate word scores
        word_frequency = Counter()
        word_degree = defaultdict(float)
        for phrase in phrase_candidates:
            words_in_phrase = phrase.split()
            for word in words_in_phrase:
                word_frequency[word] += 1
            for word in words_in_phrase:
                word_degree[word] += len(words_in_phrase) - 1
        word_scores = {}
        for word in word_frequency:
            if word_frequency[word] > 0:
                word_scores[word] = word_degree[word] / word_frequency[word]
            else:
                word_scores[word] = 0
        phrase_scores = {}
        for phrase in set(phrase_candidates):
            words_in_phrase = phrase.split()
            score = sum(word_scores.get(word, 0) for word in words_in_phrase)
            phrase_scores[phrase] = score
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_phrases[:top_n]

    # ==================== MAIN PREPROCESSING PIPELINE ====================
    def _full_preprocess_impl(self, text: str) -> str:
        start_time = time.time()
        try:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            tokens = self.remove_stopwords(tokens)
            if self.config.prefer_lemmatization_over_stemming and self.lemmatizer:
                tokens = self.lemmatize(tokens)
            elif self.stemmer:
                tokens = self.stem(tokens)
            tokens = self.extract_ngrams(tokens)
            result = ' '.join(tokens)
            processing_time = (time.time() - start_time) * 1000
            self.stats['total_processed'] += 1
            self.stats['total_processing_time_ms'] += processing_time
            self.stats['avg_processing_time_ms'] = (
                    self.stats['total_processing_time_ms'] / self.stats['total_processed']
            )
            return result
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Preprocessing failed: {e}")
            return ""
    def _preprocess_with_cache(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        return self._full_preprocess_impl_cached(text)

    def _preprocess_without_cache(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        return self._full_preprocess_impl(text)
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        if self.config.parallel_processing and len(texts) > self.config.chunk_size:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(self.preprocess, texts))
        else:
            results = [self.preprocess(text) for text in texts]
        return results

    # ==================== KEYWORD EXTRACTION METHODS ====================
    def extract_keywords(self, text: str, top_n: int = 10, method: str = "rake") -> List[str]:
        if not text:
            return []
        if method == "rake":
            keywords_with_scores = self.extract_keywords_rake(text, top_n)
            return [keyword for keyword, score in keywords_with_scores]
        else:
            processed = self.preprocess(text)
            words = processed.split()
            freq_dist = Counter(words)
            filtered = [(word, freq) for word, freq in freq_dist.items() if len(word) >= 3]
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [word for word, freq in filtered[:top_n]]

    def extract_keywords_with_scores(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:

        return self.extract_keywords_rake(text, top_n)

    # ==================== UTILITY METHODS ====================

    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        processed = self.preprocess(text)
        tokens = processed.split()
        entities = self.extract_medical_entities(text)
        keywords = self.extract_keywords(text, top_n=5)

        return {
            'original_length': len(text),
            'processed_length': len(processed),
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'average_word_length': sum(len(w) for w in tokens) / len(tokens) if tokens else 0,
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
            'medical_entities_found': sum(len(v) for v in entities.values()),
            'keywords': keywords,
            'entities': entities
        }
    def get_performance_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()

        if self.config.enable_caching:
            try:
                cache_info = self._full_preprocess_impl_cached.cache_info()
                stats['cache_hits'] = cache_info.hits
                stats['cache_misses'] = cache_info.misses
                stats['cache_size'] = cache_info.currsize
                stats['cache_maxsize'] = cache_info.maxsize
            except:
                pass
        return stats
    def reset_cache(self):
        """Reset the preprocessing cache."""
        if self.config.enable_caching:
            self._full_preprocess_impl_cached = functools.lru_cache(
                maxsize=self.config.cache_size
            )(self._full_preprocess_impl)
            logger.info("Cache reset successfully")
    def clear_stats(self):
        self.stats = {
            'total_processed': 0,
            'total_processing_time_ms': 0.0,
            'avg_processing_time_ms': 0.0,
            'errors': 0
        }
    def __repr__(self) -> str:
        return (f"TextPreprocessor(processed={self.stats['total_processed']}, "
                f"errors={self.stats['errors']}, "
                f"avg_time={self.stats['avg_processing_time_ms']:.2f}ms)")
