# src/data_processor.py

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

from .data.data_loader import DataLoader
from .nlp.analyzer import NLPAnalyzer
from .processors.post_processor import PostProcessor
from .processors.comment_processor import CommentProcessor
from .processors.conversation_processor import ConversationProcessor
from .models.data_classes import RedditPost, RedditComment

logger = logging.getLogger(__name__)

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

    def _split_and_save_data(self, conversation_pairs: List[Dict]):
        """Split data into train/test sets and save"""
        logger.info("Splitting and saving data...")
        train_pairs, test_pairs = train_test_split(
            conversation_pairs,
            test_size=0.2,
            random_state=42
        )
        
        # Save training data
        self.save_to_file(train_pairs, 'train_conversations.csv')
        
        # Save test data
        self.save_to_file(test_pairs, 'test_conversations.csv')
        
        # Save processed posts
        self.save_to_file(
            [asdict(post) for post in self.processed_posts],
            'processed_posts.json'
        )

    def process_data(self):
        """
        Main method to process the Reddit data with GPU optimization.
        
        This method orchestrates the entire processing pipeline:
        1. Loads raw data
        2. Processes posts and comments
        3. Creates conversation pairs
        4. Splits into train/test sets
        5. Saves all processed data
        """
        try:
            logger.info("Starting data processing with GPU optimization...")
            
            # Load raw data
            raw_posts, raw_comments = self.data_loader.load_data()
            
            # Process posts
            self.processed_posts = self._process_posts(raw_posts)
            
            # Process comments
            self.processed_comments = self._process_comments(raw_comments)
            
            # Create conversation pairs
            conversation_pairs = self._create_conversation_pairs()
            
            # Split and save data
            self._split_and_save_data(conversation_pairs)
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
        finally:
            self._cleanup_gpu_memory()

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