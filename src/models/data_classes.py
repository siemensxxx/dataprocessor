from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class RedditComment:
    """Data class for storing normalized Reddit comment data"""
    comment_id: str
    post_id: str  # The ID of the parent post (link_id)
    parent_id: str  # Could be post_id or another comment_id
    content: str
    author: str
    timestamp: datetime
    score: int
    edited: bool
    intent: str = None
    topics: List[dict[str, float]] = field(default_factory=list)  # List of {topic_id: probability} mappings
    dominant_topic: Optional[int] = None  # ID of the topic with highest probability
    topic_probabilities: dict[int, float] = field(default_factory=dict)  # Full topic distribution
    replies: List['RedditComment'] = field(default_factory=list) # List to store replies
    



@dataclass
class RedditPost:
    """Data class for storing normalized Reddit post data"""
    post_id: str
    title: str
    content: str
    author: str
    timestamp: datetime
    score: int
    num_comments: int
    upvote_ratio: float
    over_18: bool
    edited: bool
    comments: List[RedditComment] = field(default_factory=list)
    intent: str = None
     # New topic-related fields
    topics: List[dict[str, float]] = field(default_factory=list)  # List of {topic_id: probability} mappings
    title_topics: List[dict[str, float]] = field(default_factory=list)  # Topic distribution for title
    content_topics: List[dict[str, float]] = field(default_factory=list)  # Topic distribution for content
    dominant_topic: Optional[int] = None  # ID of the topic with highest probability
    topic_probabilities: dict[int, float] = field(default_factory=dict)  # Full topic distribution
    conversation_analysis: Optional[List[dict[str, any]]] = None # Add field for conversation analysis
 

