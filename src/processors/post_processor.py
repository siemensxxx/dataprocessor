from datetime import datetime
from typing import Dict, Any
import logging
from ..models.data_classes import RedditPost
from ..utils.text_processor import TextProcessor
from ..nlp.analyzer import NLPAnalyzer

logger = logging.getLogger(__name__)

class PostProcessor:
    def __init__(self, nlp_analyzer: NLPAnalyzer):
        self.nlp_analyzer = nlp_analyzer

    def process_post(self, post_data: Dict[str, Any]) -> RedditPost:
        """Process a single Reddit post"""
        try:
            post_id = TextProcessor.parse_reddit_id(post_data.get('name', ''))
            
            # Clean the content
            cleaned_content = TextProcessor.clean_text(post_data.get('selftext', ''))
            cleaned_title = TextProcessor.clean_text(post_data.get('title', ''))
            
            post = RedditPost(
                post_id=post_id,
                title=cleaned_title,
                content=cleaned_content,
                author=post_data.get('author', '[deleted]'),
                timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                score=post_data.get('score', 0),
                num_comments=post_data.get('num_comments', 0),
                upvote_ratio=post_data.get('upvote_ratio', 0.0),
                over_18=post_data.get('over_18', False),
                edited=bool(post_data.get('edited', False)),
                comments=[]
            )
            
            # Add intent and sentiment analysis
            if cleaned_content:
                post.intent = self.nlp_analyzer.detect_intent(cleaned_content)
                post.sentiment = self.nlp_analyzer.analyze_sentiment(cleaned_content)
                
            return post
        except Exception as e:
            logger.error(f"Error processing post {post_data.get('name', 'unknown')}: {e}")
            raise