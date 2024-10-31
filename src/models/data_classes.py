from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from typing import Dict, List

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
    sentiment: float = None
     # New topic-related fields
    topics: List[Dict[str, float]] = field(default_factory=list)  # List of {topic_id: probability} mappings
    dominant_topic: Optional[int] = None  # ID of the topic with highest probability
    topic_probabilities: Dict[int, float] = field(default_factory=dict)  # Full topic distribution


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
    sentiment: float = None
     # New topic-related fields
    topics: List[Dict[str, float]] = field(default_factory=list)  # List of {topic_id: probability} mappings
    title_topics: List[Dict[str, float]] = field(default_factory=list)  # Topic distribution for title
    content_topics: List[Dict[str, float]] = field(default_factory=list)  # Topic distribution for content
    dominant_topic: Optional[int] = None  # ID of the topic with highest probability
    topic_probabilities: Dict[int, float] = field(default_factory=dict)  # Full topic distribution