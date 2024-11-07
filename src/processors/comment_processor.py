from datetime import datetime
from typing import Dict, Any
import logging
from ..models.data_classes import RedditComment
from ..utils.text_processor import TextProcessor
from ..nlp.analyzer import NLPAnalyzer

logger = logging.getLogger(__name__)

class CommentProcessor:
    def __init__(self, nlp_analyzer: NLPAnalyzer):
        self.nlp_analyzer = nlp_analyzer

    def process_comment(self, comment_data: Dict[str, Any]) -> RedditComment:
        """Process a single Reddit comment"""
        try:
            comment_id = TextProcessor.parse_reddit_id(comment_data.get('name', ''))
            post_id = TextProcessor.parse_reddit_id(comment_data.get('link_id', ''))
            parent_id = TextProcessor.parse_reddit_id(comment_data.get('parent_id', ''))
            
            # Clean the content
            cleaned_content = TextProcessor.clean_text(comment_data.get('body', ''))
            
            comment = RedditComment(
                comment_id=comment_id,
                post_id=post_id,
                parent_id=parent_id,
                content=cleaned_content,
                author=comment_data.get('author', '[deleted]'),
                timestamp=datetime.fromtimestamp(comment_data.get('created_utc', 0)),
                score=comment_data.get('score', 0),
                edited=bool(comment_data.get('edited', False))
            )
            
            # Add intent and sentiment analysis
            if cleaned_content:
                comment.intent = self.nlp_analyzer.detect_intent(cleaned_content)
                                
            return comment
        except Exception as e:
            logger.error(f"Error processing comment {comment_data.get('name', 'unknown')}: {e}")
            raise