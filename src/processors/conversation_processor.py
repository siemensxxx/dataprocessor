from typing import Dict, List, Any
from ..models.data_classes import RedditPost, RedditComment

class ConversationProcessor:
    @staticmethod
    def build_conversation_tree(posts: List[RedditPost], comments: List[RedditComment]) -> Dict[str, RedditPost]:
        """Builds a complete conversation tree, including nested replies, and sorts comments by score."""

        posts_dict = {post.post_id: post for post in posts}
        comments_dict = {comment.comment_id: comment for comment in comments}

        # Build the tree recursively
        def attach_replies(comment_id: str, current_level: list):
            replies = [c for c in comments if c.parent_id == comment_id]
            # Sort replies by score in descending order
            replies.sort(key=lambda x: x.score, reverse=True)
            for reply in replies:
                current_level.append(reply)
                attach_replies(reply.comment_id, reply.replies)  # Add nested replies

        for post in posts_dict.values():
            top_level_comments = [c for c in comments if c.parent_id == post.post_id]
            top_level_comments.sort(key=lambda x: x.score, reverse=True)  # Sort by score
            post.comments = top_level_comments
            for comment in top_level_comments:
                comment.replies = [] #Initialize replies for each comment
                attach_replies(comment.comment_id, comment.replies)
        return posts_dict


    @staticmethod
    def create_conversation_pairs(posts_dict: Dict[str, RedditPost]) -> List[Dict[str, Any]]:
        """Creates conversation pairs, handling nested replies."""
        conversation_pairs = []

        def traverse_tree(context: str, context_intent: str, context_author: str, current_level: list, post_id: str):
            for item in current_level:
                pair = {
                    'post_id': post_id,
                    'context': context,
                    'response': item.content,
                    'context_intent': context_intent or 'unknown',
                    'response_intent': item.intent or 'unknown',
                    'context_author': context_author,
                    'response_author': item.author,
                    'score': item.score,
                    'timestamp': item.timestamp.isoformat()
                }
                conversation_pairs.append(pair)
                traverse_tree(item.content, item.intent, item.author, item.replies, post_id)

        for post in posts_dict.values():
            for comment in post.comments:
                pair = {
                    'post_id': post.post_id,
                    'context': post.content,
                    'response': comment.content,
                    'context_intent': post.intent,
                    'response_intent': comment.intent or 'unknown',
                    'context_author': post.author or 'unknown',
                    'response_author': comment.author,
                    'score': comment.score,
                    'timestamp': comment.timestamp.isoformat()
                }
                conversation_pairs.append(pair)
                traverse_tree(comment.content, comment.intent, comment.author, comment.replies, post.post_id)

        return conversation_pairs