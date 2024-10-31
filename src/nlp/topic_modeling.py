# src/nlp/topic_modeling.py

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TopicModeler:
    def __init__(self, 
                 method: str = 'lda',
                 n_topics: int = 10,
                 max_features: int = 10000,
                 batch_size: int = 128,
                 n_jobs: int = -1):
        """
        Initialize the topic modeling system.
        
        Args:
            method (str): Topic modeling method ('lda' or 'nmf')
            n_topics (int): Number of topics to extract
            max_features (int): Maximum number of features for vocabulary
            batch_size (int): Batch size for processing
            n_jobs (int): Number of jobs for parallel processing
        """
        self.method = method.lower()
        self.n_topics = n_topics
        self.max_features = max_features
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        
        # Initialize vectorizers
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            max_df=0.95,  # Ignore terms that appear in >95% of docs
            min_df=2      # Ignore terms that appear in <2 documents
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        
        # Initialize topic model
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=20,
                learning_method='online',
                batch_size=batch_size,
                n_jobs=n_jobs,
                random_state=42
            )
        elif self.method == 'nmf':
            self.model = NMF(
                n_components=n_topics,
                init='nndsvd',
                max_iter=200,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        self.feature_names = None
        self.document_topics = None
        self.topic_terms = None
        
    def preprocess_texts(self, texts: List[str]) -> np.ndarray:
        """
        Preprocess and vectorize the input texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Document-term matrix
        """
        logger.info(f"Preprocessing {len(texts)} documents...")
        
        try:
            if self.method == 'lda':
                # Use count vectorization for LDA
                dtm = self.count_vectorizer.fit_transform(texts)
                self.feature_names = self.count_vectorizer.get_feature_names_out()
            else:
                # Use TF-IDF for NMF
                dtm = self.tfidf_vectorizer.fit_transform(texts)
                self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
            logger.info(f"Vocabulary size: {len(self.feature_names)}")
            return dtm
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
            
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the topic model and transform documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            Document-topic matrix
        """
        try:
            # Preprocess texts
            dtm = self.preprocess_texts(texts)
            
            logger.info("Fitting topic model...")
            self.document_topics = self.model.fit_transform(dtm)
            
            # Get topic-term matrix
            self.topic_terms = self.model.components_
            
            return self.document_topics
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            raise
            
    def get_topic_terms(self, n_terms: int = 10) -> List[List[str]]:
        """
        Get the top terms for each topic.
        
        Args:
            n_terms: Number of terms to return per topic
            
        Returns:
            List of top terms for each topic
        """
        topics = []
        for topic_idx in range(self.n_topics):
            top_term_indices = self.topic_terms[topic_idx].argsort()[:-n_terms-1:-1]
            topics.append([
                self.feature_names[i] for i in top_term_indices
            ])
        return topics
        
    def get_document_topics(self, 
                          threshold: float = 0.1) -> List[List[Tuple[int, float]]]:
        """
        Get topic assignments for each document.
        
        Args:
            threshold: Minimum probability threshold for topic assignment
            
        Returns:
            List of (topic_id, probability) tuples for each document
        """
        doc_topics = []
        for doc_topic_dist in self.document_topics:
            # Normalize probabilities
            probs = normalize(doc_topic_dist.reshape(1, -1))[0]
            
            # Get topics above threshold
            topic_probs = [
                (topic_idx, prob) 
                for topic_idx, prob in enumerate(probs)
                if prob > threshold
            ]
            
            # Sort by probability
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            doc_topics.append(topic_probs)
            
        return doc_topics
        
    def get_topic_summary(self, n_terms: int = 10) -> List[Dict[str, Any]]:
        """
        Get a summary of all topics with their top terms.
        
        Args:
            n_terms: Number of terms to include per topic
            
        Returns:
            List of topic summaries with terms and term weights
        """
        summaries = []
        for topic_idx in range(self.n_topics):
            # Get top terms and their weights
            term_weights = self.topic_terms[topic_idx]
            top_term_indices = term_weights.argsort()[:-n_terms-1:-1]
            
            terms = []
            for idx in top_term_indices:
                terms.append({
                    'term': self.feature_names[idx],
                    'weight': float(term_weights[idx])
                })
            
            summaries.append({
                'topic_id': topic_idx,
                'terms': terms
            })
            
        return summaries
        
    def save_topic_model(self, output_dir: str):
        """
        Save topic model results to files.
        
        Args:
            output_dir: Directory to save files
        """
        try:
            # Save topic terms
            topic_terms = self.get_topic_terms()
            pd.DataFrame(topic_terms).to_csv(
                f"{output_dir}/topic_terms.csv",
                index=True,
                header=[f"term_{i}" for i in range(len(topic_terms[0]))]
            )
            
            # Save topic summaries
            summaries = self.get_topic_summary()
            pd.DataFrame(summaries).to_json(
                f"{output_dir}/topic_summaries.json",
                orient='records',
                indent=2
            )
            
            logger.info(f"Saved topic model results to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving topic model: {e}")
            raise
