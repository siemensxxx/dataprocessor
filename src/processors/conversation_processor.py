from typing import Dict, List, Any
from ..models.data_classes import RedditPost, RedditComment

class ConversationProcessor:
    @staticmethod
    def build_conversation_tree(posts: List[RedditPost], comments: List[RedditComment]) -> Dict[str, RedditPost]:
        """Build conversation trees by linking comments to their posts and parent comments"""
        # Create a dictionary of posts for easy lookup
        posts_dict = {post.post_id: post for post in posts}
        
        # Create a dictionary of comments for easy lookup
        comments_dict = {comment.comment_id: comment for comment in comments}
        
        # Link comments to their parents
        for comment in comments:
            if comment.post_id in posts_dict:
                # If parent is a post, add to post's comments
                if comment.parent_id == comment.post_id:
                    posts_dict[comment.post_id].comments.append(comment)
                # If parent is another comment, it's a reply
                elif comment.parent_id in comments_dict:
                    parent_comment = comments_dict[comment.parent_id]
                    if not hasattr(parent_comment, 'replies'):
                        parent_comment.replies = []
                    parent_comment.replies.append(comment)
        
        return posts_dict

    @staticmethod
    def create_conversation_pairs(posts_dict: Dict[str, RedditPost]) -> List[Dict[str, Any]]:
        """Create conversation pairs from posts and comments"""
        conversation_pairs = []
        
        for post in posts_dict.values():
            # Create pairs between post and direct comments
            for comment in post.comments:
                pair = {
                    'post_id': post.post_id,
                    'context': post.content,
                    'response': comment.content,
                    'context_intent': post.intent,
                    'response_intent': comment.intent,
                    'context_sentiment': post.sentiment,
                    'response_sentiment': comment.sentiment,
                    'context_author': post.author,
                    'response_author': comment.author,
                    'score': comment.score,
                    'timestamp': comment.timestamp.isoformat()
                }
                conversation_pairs.append(pair)
                
                # If comment has replies, create pairs between comment and replies
                if hasattr(comment, 'replies'):
                    for reply in comment.replies:
                        pair = {
                            'post_id': post.post_id,
                            'context': comment.content,
                            'response': reply.content,
                            'context_intent': comment.intent,
                            'response_intent': reply.intent,
                            'context_sentiment': comment.sentiment,
                            'response_sentiment': reply.sentiment,
                            'context_author': comment.author,
                            'response_author': reply.author,
                            'score': reply.score,
                            'timestamp': reply.timestamp.isoformat()
                        }
                        conversation_pairs.append(pair)
        
        return conversation_pairs