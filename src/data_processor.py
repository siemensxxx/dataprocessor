# src/data_processor.py

import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Generator, Iterator, Optional
from sklearn.model_selection import train_test_split
from dataclasses import asdict, dataclass, field
import nltk


import torch
from tqdm import tqdm
import gc
from .nlp.topic_modeling import TopicModeler
import psutil



from .data.data_loader import DataLoader
from .nlp.analyzer import NLPAnalyzer
from .processors.post_processor import PostProcessor
from .processors.comment_processor import CommentProcessor
from .processors.conversation_processor import ConversationProcessor
from .models.data_classes import RedditPost, RedditComment
from .nlp.language_analyzer import LanguageStyleAnalyzer


import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Generator, Iterator, Optional
from sklearn.model_selection import train_test_split
from dataclasses import asdict
import torch
from tqdm import tqdm
import gc
from datetime import datetime
import time
from .nlp.topic_modeling import TopicModeler
from .nlp.analyzer import NLPAnalyzer
from .data.data_loader import DataLoader
from .processors.post_processor import PostProcessor
from .processors.comment_processor import CommentProcessor
from .processors.conversation_processor import ConversationProcessor
from .models.data_classes import RedditPost, RedditComment
import sys
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    BarColumn,
    TextColumn
)
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

logger = logging.getLogger(__name__)



def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure enhanced logging with rich handler and file output"""
    log_file = output_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True, console=console),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

class PerformanceMetrics:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

class ProcessingStats:
    """Track processing statistics, timing, and system performance"""
    def __init__(self):
        self.start_time = time.time()
        self.operation_times = {}
        self.operation_counts = {}
        self.errors = []
        self.warnings = []
        self.performance_history: List[PerformanceMetrics] = []
        self.monitoring_interval = 1.0  # seconds
        self.last_monitored = time.time()
        
    def _get_gpu_metrics(self) -> tuple[Optional[float], Optional[float]]:
        """Get GPU utilization and memory usage if available"""
        if torch.cuda.is_available():
            try:
                gpu_util = torch.cuda.utilization()
                gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                return gpu_util, gpu_mem
            except:
                return None, None
        return None, None

    #def update_performance_metrics(self):
        """Update performance metrics if monitoring interval has elapsed."""
        current_time = time.time()
        if current_time - self.last_monitored >= self.monitoring_interval:
            gpu_util, gpu_mem = self._get_gpu_metrics()
            metrics = PerformanceMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                gpu_utilization=gpu_util,
                gpu_memory_used=gpu_mem
            )
            self.performance_history.append(metrics)
            self.last_monitored = current_time
        
    def start_operation(self, operation_name: str):
        """Start timing an operation and record initial performance metrics"""
        self.operation_times[operation_name] = {
            'start': time.time(),
            'start_metrics': PerformanceMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                *self._get_gpu_metrics()
            )
        }
        
    def end_operation(self, operation_name: str, success: bool = True):
        """End timing an operation and record final performance metrics"""
        if operation_name in self.operation_times:
            end_time = time.time()
            duration = end_time - self.operation_times[operation_name]['start']
            self.operation_times[operation_name].update({
                'duration': duration,
                'success': success,
                'end_metrics': PerformanceMetrics(
                    cpu_percent=psutil.cpu_percent(),
                    memory_percent=psutil.virtual_memory().percent,
                    *self._get_gpu_metrics()
                )
            })
            
    def get_performance_summary(self) -> Dict:
        """Generate summary of performance metrics"""
        if not self.performance_history:
            return {}
            
        cpu_percentages = [m.cpu_percent for m in self.performance_history]
        memory_percentages = [m.memory_percent for m in self.performance_history]
        gpu_utils = [m.gpu_utilization for m in self.performance_history if m.gpu_utilization is not None]
        gpu_mems = [m.gpu_memory_used for m in self.performance_history if m.gpu_memory_used is not None]
        
        return {
            'cpu': {
                'avg': sum(cpu_percentages) / len(cpu_percentages),
                'max': max(cpu_percentages),
                'min': min(cpu_percentages)
            },
            'memory': {
                'avg': sum(memory_percentages) / len(memory_percentages),
                'max': max(memory_percentages),
                'min': min(memory_percentages)
            },
            'gpu': {
                'avg_utilization': sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
                'max_utilization': max(gpu_utils) if gpu_utils else None,
                'avg_memory': sum(gpu_mems) / len(gpu_mems) if gpu_mems else None,
                'max_memory': max(gpu_mems) if gpu_mems else None
            }
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive processing report including performance metrics"""
        total_duration = time.time() - self.start_time
        
        return {
            'total_duration': total_duration,
            'operations': self.operation_times,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'performance': self.get_performance_summary()
        }

class ProcessingStats:
    """Track processing statistics and timing"""
    def __init__(self):
        self.start_time = time.time()
        self.operation_times = {}
        self.operation_counts = {}
        self.errors = []
        self.warnings = []
        
    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.operation_times[operation_name] = {'start': time.time()}
        
    def end_operation(self, operation_name: str, success: bool = True):
        """End timing an operation and record statistics"""
        if operation_name in self.operation_times:
            end_time = time.time()
            duration = end_time - self.operation_times[operation_name]['start']
            self.operation_times[operation_name]['duration'] = duration
            self.operation_times[operation_name]['success'] = success
            
    def add_error(self, operation: str, error: str):
        """Record an error"""
        self.errors.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'error': error
        })
        
    def add_warning(self, operation: str, warning: str):
        """Record a warning"""
        self.warnings.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'warning': warning
        })
        
    def generate_report(self) -> Dict:
        """Generate comprehensive processing report"""
        total_duration = time.time() - self.start_time
        
        return {
            'total_duration': total_duration,
            'operations': self.operation_times,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }


class GPUOptimizedProcessor:
    def __init__(self, 
                 posts_file: str, 
                 comments_file: str, 
                 output_dir: str, 
                 batch_size: int = 32,
                 chunk_size: int = 1000):
        """
        Initialize the Reddit data processor optimized for GPU processing.
        
        Args:
            posts_file (str): Path to the posts JSON file
            comments_file (str): Path to the comments JSON file
            output_dir (str): Directory for output files
            batch_size (int): Size of batches for NLP processing
            chunk_size (int): Size of chunks for data processing
        """
        self.data_loader = DataLoader(posts_file, comments_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU optimized settings
        self.CHUNK_SIZE = chunk_size
        self.BATCH_SIZE = batch_size
        
        # Initialize components
        self.nlp_analyzer = self._initialize_nlp()
        self.post_processor = PostProcessor(self.nlp_analyzer)
        self.comment_processor = CommentProcessor(self.nlp_analyzer)
        self.conversation_processor = ConversationProcessor()
        
        # Initialize state tracking
        self.processed_posts: Optional[List[RedditPost]] = None
        self.processed_comments: Optional[List[RedditComment]] = None
        self.posts_dict: Optional[Dict[str, RedditPost]] = None

    def _initialize_nlp(self) -> NLPAnalyzer:
        """Initialize NLP analyzer with GPU support"""
        if torch.cuda.is_available():
            logger.info(f"Initializing NLP analyzer with GPU support: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available, falling back to CPU")
        
        return NLPAnalyzer(batch_size=self.BATCH_SIZE)

    def process_in_chunks(self, 
                         items: Iterator, 
                         processor_func, 
                         desc: str,
                         chunk_size: Optional[int] = None) -> Generator:
        """
        Process items in chunks optimized for GPU.
        
        Args:
            items: Iterator of items to process
            processor_func: Function to process each item
            desc: Description for progress bar
            chunk_size: Optional override for chunk size
            
        Yields:
            Processed items one at a time
        """
        chunk_size = chunk_size or self.CHUNK_SIZE
        chunk = []
        total_processed = 0
        
        for item in tqdm(items, desc=desc):
            chunk.append(item)
            if len(chunk) >= chunk_size:
                for processed_item in self._process_chunk(chunk, processor_func):
                    yield processed_item
                    total_processed += 1
                    
                    if total_processed % (chunk_size * 5) == 0:
                        logger.info(f"Processed {total_processed} items")
                        self._cleanup_gpu_memory()
                
                chunk = []
        
        # Process remaining items
        if chunk:
            yield from self._process_chunk(chunk, processor_func)

    def _process_chunk(self, chunk: List, processor_func) -> List:
        """
        Process a single chunk of data with error handling.
        
        Args:
            chunk: List of items to process
            processor_func: Function to process each item
            
        Returns:
            List of processed items
        """
        try:
            processed_items = []
            for item in chunk:
                try:
                    processed = processor_func(item)
                    if processed is not None:
                        processed_items.append(processed)
                except Exception as e:
                    logger.error(f"Error processing individual item: {e}")
            return processed_items
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return []

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def save_to_file(self, 
                     data: List[Dict], 
                     filename: str, 
                     mode: str = 'w',
                     chunk_size: int = 1000):
        """
        Save processed data to file with chunking for large datasets.
        
        Args:
            data: List of data to save
            filename: Output filename
            mode: Write mode ('w' for write, 'a' for append)
            chunk_size: Size of chunks for writing
        """
        file_path = self.output_dir / filename
        
        if mode == 'w' and file_path.exists():
            file_path.unlink()
        
        try:
            if filename.endswith('.csv'):
                self._save_csv(data, file_path, mode, chunk_size)
            elif filename.endswith('.json'):
                self._save_json(data, file_path, mode, chunk_size)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
                
        except Exception as e:
            logger.error(f"Error saving to {filename}: {e}")
            raise

    def _save_csv(self, 
                  data: List[Dict], 
                  file_path: Path, 
                  mode: str, 
                  chunk_size: int):
        """Save data to CSV file"""
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            df = pd.DataFrame(chunk)
            df.to_csv(
                file_path, 
                mode=mode, 
                header=(i == 0 or mode == 'w'), 
                index=False
            )

    def _save_json(self, 
                   data: List[Dict], 
                   file_path: Path, 
                   mode: str, 
                   chunk_size: int):
        """Save data to JSON file"""
        if mode == 'w':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                with open(file_path, 'a') as f:
                    if i == 0:
                        f.write('[\n')
                    json.dump(chunk, f, indent=2)
                    if i + chunk_size < len(data):
                        f.write(',\n')
                    else:
                        f.write('\n]')

    def _process_posts(self, raw_posts: List[Dict]) -> List[RedditPost]:
        """Process all posts"""
        logger.info("Processing posts...")
        return list(self.process_in_chunks(
            raw_posts,
            self.post_processor.process_post,
            "Processing posts"
        ))

    def _process_comments(self, raw_comments: List[Dict]) -> List[RedditComment]:
        """Process all comments"""
        logger.info("Processing comments...")
        return list(self.process_in_chunks(
            raw_comments,
            self.comment_processor.process_comment,
            "Processing comments"
        ))

    def _create_conversation_pairs(self) -> List[Dict]:
        """Create conversation pairs from processed posts and comments"""
        logger.info("Creating conversation pairs...")
        self.posts_dict = self.conversation_processor.build_conversation_tree(
            self.processed_posts,
            self.processed_comments
        )
        return self.conversation_processor.create_conversation_pairs(self.posts_dict)
    

    def _split_and_save_data(self, conversation_pairs):
        """Split data into train/test sets and save if non-empty."""
        if not conversation_pairs:
            logger.warning("No conversation pairs available. Skipping train/test split and save.")
            return

        try:
            # Perform train-test split
            train_pairs, test_pairs = train_test_split(
                conversation_pairs,
                test_size=0.2,
                random_state=42
            )

            # Save training data
            train_path = self.output_dir / 'train_conversations.csv'
            pd.DataFrame(train_pairs).to_csv(train_path, index=False)
            logger.info(f"Training data saved to {train_path}")

            # Save test data
            test_path = self.output_dir / 'test_conversations.csv'
            pd.DataFrame(test_pairs).to_csv(test_path, index=False)
            logger.info(f"Test data saved to {test_path}")

        except ValueError as e:
            logger.error(f"Error splitting conversation pairs: {e}")

    def _analyze_language_style(self):
        """Analyze language style and common phrases"""
        logger.info("Analyzing language style and common phrases...")
        
        # Initialize language analyzer
        language_analyzer = LanguageStyleAnalyzer(
            min_phrase_freq=3,
            max_ngram_size=3
        )
        
        # Combine post and comment texts
        texts = []
        
        # Add post texts
        for post in self.processed_posts:
            if post.title:
                texts.append(post.title)
            if post.content:
                texts.append(post.content)
                
        # Add comment texts
        for comment in self.processed_comments:
            if comment.content:
                texts.append(comment.content)
        
        # Analyze content
        language_analyzer.analyze_content(texts)
        
        # Save results
        language_analyzer.save_results(
            self.output_dir / 'language_analysis.json'
        )
        
        # Log summary statistics
        stats = language_analyzer.get_summary_statistics()
        logger.info("Language analysis summary:")
        logger.info(f"Total unique phrases: {stats['total_unique_phrases']}")
        logger.info(f"Total slang terms: {stats['total_slang_terms']}")
        logger.info(f"Total sentence patterns: {stats['total_sentence_patterns']}")


    def process_data(self):
        try:
            stats = ProcessingStats()
            logger.info("Starting data processing with GPU optimization...")
            
            # Load raw data
            raw_posts, raw_comments = self.data_loader.load_data()
            
            # Process posts
            self.processed_posts = self._process_posts(raw_posts)
            
            # Process comments
            self.processed_comments = self._process_comments(raw_comments)
            
            # Analyze language style
            self._analyze_language_style()
            
            # Extract topics
            self._extract_topics()
            
            # Create conversation pairs
            conversation_pairs = self._create_conversation_pairs()
            
            # Split and save data
            self._split_and_save_data(conversation_pairs)
            
            # Update performance metrics
            #stats.update_performance_metrics()
            
            logger.info("Processing completed successfully!")
            logger.info(f"Number of conversation pairs: {len(conversation_pairs)}")
        
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
        
        finally:
            self._cleanup_gpu_memory()
            
    def _extract_topics(self):
        """Extracts topics from the processed posts and comments."""
        logger.info("Starting topic extraction...")
        
        # Instantiate the topic modeler with LDA (or NMF) parameters
        topic_modeler = TopicModeler(
            method='lda',         # You can switch to 'nmf' if desired
            n_topics=10,          # Set the number of topics you want
            max_features=10000    # Set max features based on your text data
        )
        
        # Collect text data for topic extraction (posts + comments)
        texts = []
        for post in self.processed_posts:
            if post.content:
                texts.append(post.content)
            if post.title:
                texts.append(post.title)
        for comment in self.processed_comments:
            if comment.content:
                texts.append(comment.content)

        # Step 1: Preprocess and transform texts for topic modeling
        try:
            document_topic_matrix = topic_modeler.fit_transform(texts)
        except Exception as e:
            logger.error(f"Error during topic modeling: {e}")
            return
        
        # Step 2: Retrieve and structure the topics with top terms
        try:
            topic_terms = topic_modeler.get_topic_terms(n_terms=10)  # Get top terms for each topic
            topic_summary = topic_modeler.get_topic_summary(n_terms=10)
            
            # Log summary for review
            for idx, topic in enumerate(topic_summary):
                logger.info(f"Topic {idx + 1}: {', '.join([term['term'] for term in topic['terms']])}")

            # Save topic summaries to a JSON file
            summary_path = self.output_dir / "topic_summary.json"
            with open(summary_path, "w") as f:
                json.dump(topic_summary, f, indent=2)
            logger.info(f"Topic summaries saved to {summary_path}")

            # Save document-topic matrix to CSV
            dtm_path = self.output_dir / "document_topic_matrix.csv"
            pd.DataFrame(document_topic_matrix).to_csv(dtm_path, index=False)
            logger.info(f"Document-topic matrix saved to {dtm_path}")

        except Exception as e:
            logger.error(f"Error saving topic results: {e}")

        logger.info("Topic extraction completed successfully.")



    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed data"""
        stats = {
            "total_posts": len(self.processed_posts) if self.processed_posts else 0,
            "total_comments": len(self.processed_comments) if self.processed_comments else 0,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "batch_size": self.BATCH_SIZE,
            "chunk_size": self.CHUNK_SIZE
        }
        
        if self.posts_dict:
            stats["posts_with_comments"] = sum(
                1 for post in self.posts_dict.values() if post.comments
            )
            
        return stats

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_gpu_memory()