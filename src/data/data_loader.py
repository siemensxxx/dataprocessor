import json
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, posts_file: str, comments_file: str):
        self.posts_file = Path(posts_file)
        self.comments_file = Path(comments_file)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load posts and comments from JSON files in chunks"""
        try:
            logger.info(f"Loading posts from {self.posts_file}")
            posts = []
            with open(self.posts_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            posts.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipped invalid JSON line in posts file")
                    if len(posts) % 1000 == 0:
                        logger.info(f"Loaded {len(posts)} posts...")
                        
            logger.info(f"Loading comments from {self.comments_file}")
            comments = []
            with open(self.comments_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            comments.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipped invalid JSON line in comments file")
                    if len(comments) % 5000 == 0:
                        logger.info(f"Loaded {len(comments)} comments...")
                        
            logger.info(f"Loaded {len(posts)} posts and {len(comments)} comments")
            return posts, comments
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise