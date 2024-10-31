# src/nlp/language_analyzer.py

from collections import Counter, defaultdict
import json
import logging
from typing import List, Dict, Any, Set, Tuple
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LanguageStyleAnalyzer:
    """
    Analyzes and extracts linguistic patterns, common phrases, and unique language features
    from subreddit content to help chatbots mimic community-specific communication styles.
    """
    
    def __init__(self, min_phrase_freq: int = 3, max_ngram_size: int = 3):
        """
        Initialize the language style analyzer.
        
        Args:
            min_phrase_freq (int): Minimum frequency for phrases to be considered common
            max_ngram_size (int): Maximum size of n-grams to analyze
        """
        self.min_phrase_freq = min_phrase_freq
        self.max_ngram_size = max_ngram_size
        
        # Initialize spaCy for better text processing
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        
        # Get stop words but keep some important ones
        self.stop_words = set(stopwords.words('english')) - {
            'not', 'no', 'never', 'can', 'should', 'must', 'need',
            'would', 'could', 'might', 'may', 'will'
        }
        
        # Initialize storage for analysis results
        self.common_phrases = defaultdict(Counter)
        self.slang_terms = Counter()
        self.sentence_patterns = defaultdict(Counter)
        self.subreddit_specific_terms = Counter()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Cleaned and preprocessed text
        """
        # Convert to lowercase and remove URLs
        text = re.sub(r'http\S+|www\S+', '', text.lower())
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^a-z0-9\s\']+', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_ngrams(self, text: str) -> Dict[int, Counter]:
        """
        Extract n-grams from text.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            Dict[int, Counter]: Dictionary of n-gram sizes to their frequency counts
        """
        tokens = word_tokenize(text)
        ngram_counts = {}
        
        for n in range(2, self.max_ngram_size + 1):
            text_ngrams = ngrams(tokens, n)
            ngram_counts[n] = Counter(' '.join(gram) for gram in text_ngrams)
        
        return ngram_counts
    
    def _identify_slang(self, text: str, common_words: Set[str]) -> List[str]:
        """
        Identify potential slang terms and informal language.
        
        Args:
            text (str): Preprocessed text
            common_words (Set[str]): Set of common English words
            
        Returns:
            List[str]: List of identified slang terms
        """
        # Tokenize and analyze with spaCy
        doc = self.nlp(text)
        
        slang_candidates = []
        for token in doc:
            word = token.text.lower()
            # Consider as slang if:
            # 1. Not in common words
            # 2. Not a stop word
            # 3. Not purely numeric
            # 4. Length > 2
            if (word not in common_words and 
                word not in self.stop_words and 
                not word.isnumeric() and 
                len(word) > 2):
                slang_candidates.append(word)
        
        return slang_candidates
    
    def _analyze_sentence_patterns(self, text: str) -> List[str]:
        """
        Extract common sentence structures and patterns.
        
        Args:
            text (str): Raw text content
            
        Returns:
            List[str]: List of sentence pattern templates
        """
        sentences = sent_tokenize(text)
        patterns = []
        
        for sentence in sentences:
            doc = self.nlp(sentence)
            # Create pattern template using POS tags
            pattern = ' '.join([token.pos_ for token in doc])
            patterns.append(pattern)
        
        return patterns
    
    def analyze_content(self, texts: List[str], reference_texts: List[str] = None):
        """
        Analyze a collection of texts to extract language patterns.
        
        Args:
            texts (List[str]): List of texts from the subreddit
            reference_texts (List[str], optional): General English texts for comparison
        """
        logger.info("Starting content analysis...")
        
        # Load common English words for comparison
        common_words = set(word.lower() for word in nltk.corpus.words.words())
        
        for text in tqdm(texts, desc="Analyzing texts"):
            if not text or not isinstance(text, str):
                continue
                
            processed_text = self._preprocess_text(text)
            
            # Extract and count n-grams
            ngram_counts = self._extract_ngrams(processed_text)
            for n, counts in ngram_counts.items():
                self.common_phrases[n].update(counts)
            
            # Identify slang and informal language
            slang = self._identify_slang(processed_text, common_words)
            self.slang_terms.update(slang)
            
            # Analyze sentence patterns
            patterns = self._analyze_sentence_patterns(text)
            self.sentence_patterns['templates'].update(patterns)
        
        # Filter and process results
        self._process_results(reference_texts)
        
    def _process_results(self, reference_texts: List[str] = None):
        """
        Process and filter analysis results.
        
        Args:
            reference_texts (List[str], optional): Reference texts for comparison
        """
        # Filter phrases by frequency
        for n in self.common_phrases:
            self.common_phrases[n] = Counter({
                phrase: count for phrase, count in self.common_phrases[n].items()
                if count >= self.min_phrase_freq
            })
        
        # If reference texts provided, identify subreddit-specific terms
        if reference_texts:
            reference_terms = Counter()
            for text in reference_texts:
                processed = self._preprocess_text(text)
                tokens = word_tokenize(processed)
                reference_terms.update(tokens)
            
            # Compare frequencies to identify unique terms
            for term, count in self.slang_terms.items():
                if term not in reference_terms or \
                   (count / sum(self.slang_terms.values()) > 
                    reference_terms[term] / sum(reference_terms.values()) * 2):
                    self.subreddit_specific_terms[term] = count
    
    def save_results(self, output_path: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            output_path (str): Path to save the results
        """
        results = {
            'common_phrases': {
                f'{n}_grams': dict(self.common_phrases[n].most_common(50))
                for n in self.common_phrases
            },
            'slang_terms': dict(self.slang_terms.most_common(100)),
            'sentence_patterns': {
                'common_templates': dict(
                    self.sentence_patterns['templates'].most_common(50)
                )
            },
            'subreddit_specific_terms': dict(
                self.subreddit_specific_terms.most_common(100)
            )
        }
        
        # Add metadata
        results['metadata'] = {
            'total_phrases_analyzed': sum(
                len(counter) for counter in self.common_phrases.values()
            ),
            'total_slang_terms': len(self.slang_terms),
            'total_sentence_patterns': len(self.sentence_patterns['templates']),
            'analysis_parameters': {
                'min_phrase_freq': self.min_phrase_freq,
                'max_ngram_size': self.max_ngram_size
            }
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Analysis results saved to {output_path}")
        
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the analysis.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        return {
            'total_unique_phrases': {
                f'{n}_grams': len(counter)
                for n, counter in self.common_phrases.items()
            },
            'total_slang_terms': len(self.slang_terms),
            'total_sentence_patterns': len(self.sentence_patterns['templates']),
            'top_phrases': {
                f'{n}_grams': dict(counter.most_common(10))
                for n, counter in self.common_phrases.items()
            },
            'top_slang': dict(self.slang_terms.most_common(10)),
            'common_patterns': dict(
                self.sentence_patterns['templates'].most_common(10)
            )
        }
