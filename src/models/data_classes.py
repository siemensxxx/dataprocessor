from dataclasses import dataclass, field
from datetime import datetime
from typing import List

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