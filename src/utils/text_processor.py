import re
import emoji
from typing import List, Any
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emoji but keep the text representation
        text = emoji.demojize(text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    @staticmethod
    def parse_reddit_id(full_id: str) -> str:
        """Extract the base ID from Reddit's fullname format"""
        if full_id and '_' in full_id:
            return full_id.split('_')[1]
        return full_id
    
        
        
