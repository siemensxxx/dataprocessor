from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from typing import List, Any
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans  # Or DBSCAN, AgglomerativeClustering, etc.
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression  # Or other multi-label classifiers
from sentence_transformers import SentenceTransformer, util
import umap
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

class NLPAnalyzer:
    def __init__(self, batch_size: int = 128, use_gpu: bool = True):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        
        logger.info(f"Initializing NLP models on {self.device}")
        
        # Initialize tokenizer for length checking
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.max_length = 512  # Maximum sequence length for the model
        
        
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() and use_gpu else -1,
            batch_size=self.batch_size
        )

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within model's maximum sequence length"""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > self.max_length:
            logger.debug(f"Truncating text from {len(tokens)} tokens to {self.max_length} tokens")
            truncated_tokens = tokens[:self.max_length - 1] + [tokens[-1]]  # Keep [SEP] token
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return text

    def detect_intent(self, text: str) -> str:
        """Detect the intent of the text using zero-shot classification"""
        if not text.strip():
            return "unknown"
            
        try:
            truncated_text = self._truncate_text(text)
            result = self.intent_classifier(
                truncated_text,
                candidate_labels=["question", "opinion", "answer", "discussion"],
                hypothesis_template="This text is expressing a {}."
            )
            return result['labels'][0]
        except Exception as e:
            logger.warning(f"Error detecting intent: {e}")
            return "unknown"
    

    def process_batch(self, texts: List[str], processor_fn) -> List[Any]:
        """Process a batch of texts using the specified processor function"""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = [processor_fn(text) for text in batch]
            results.extend(batch_results)
        return results
    
    
class RedditDataAnalyzer:
    def __init__(self, posts_data: List[Dict], comments_data: List[Dict], conversation_trees: Dict[str, Any]):
        self.posts_data = posts_data
        self.comments_data = comments_data
        self.conversation_trees = conversation_trees
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        self.target_topics = ["Injury", "Routine", "Supplements", "Lifestyle", "Personal Stories", "Size Discussion"]
        self.stop_words = set(stopwords.words('english'))

    def embed_content(self, content: str) -> np.ndarray:
        return self.sbert_model.encode(content)

    def cluster_posts(self, embeddings: np.ndarray, n_clusters: int = 6) -> np.ndarray:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def classify_posts(self, embeddings: np.ndarray, labels: np.ndarray) -> Any:
        classifier = MultiOutputClassifier(LogisticRegression())
        classifier.fit(embeddings, self._convert_to_multilabel(labels))
        return classifier

    def _convert_to_multilabel(self, labels):
        multi_labels = np.zeros((len(labels), len(self.target_topics)))
        for i, label in enumerate(labels):
            multi_labels[i, label] = 1
        return multi_labels

    def extract_key_phrases(self, text):
        words = word_tokenize(text)
        filtered_words = [w for w in words if w not in self.stop_words and w.isalnum()]
        text = " ".join(filtered_words)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        phrase_scores = tfidf_matrix.toarray()[0]
        key_phrases = [phrase for phrase, score in zip(feature_names, phrase_scores) if score > 0]
        return key_phrases

    def analyze_conversation_tree(self, post_id: str):
        tree = self.conversation_trees.get(post_id)
        if not tree:
            return None

        conversation_flow = []
        root_embedding = self.embed_content(tree.content)
        MAX_DEPTH = 10

        def traverse_tree(node, parent_embedding, depth=0):
            if depth > MAX_DEPTH:
                return
            for comment in node.comments:
                comment_embedding = self.embed_content(comment.content)
                similarity = util.cos_sim(parent_embedding, comment_embedding).item()
                conversation_flow.append({
                    'comment': comment.content,
                    'similarity_to_parent': similarity
                })
                traverse_tree(comment, comment_embedding, depth + 1)

        traverse_tree(tree, root_embedding)
        return conversation_flow

    def visualize_clusters(self, embeddings, labels):
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        for i in np.unique(labels):
            plt.scatter(embeddings_2d[labels == i, 0], embeddings_2d[labels == i, 1], label=str(i))
        plt.legend()
        plt.title('UMAP Visualization of Post Clusters')
        plt.show()

    def visualize_conversation_tree(self, post_id):
        net = Network(notebook=True, cdn_resources='in_line')
        tree = self.conversation_trees[post_id]

        def add_to_graph(node, parent_id=None):
            node_id = node.post_id if hasattr(node, 'post_id') else node.comment_id
            content = getattr(node, 'content', '')
            net.add_node(node_id, label=self.get_shortened_text(content))
            if parent_id:
                net.add_edge(parent_id, node_id)

            for comment in getattr(node, 'comments', []):
                add_to_graph(comment, node_id)

        add_to_graph(tree)
        net.show("conversation_tree.html")

    def get_shortened_text(self, text, max_length=20):
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def export_to_markdown(self, clusters, output_path='analysis_summary.md'):
        with open(output_path, 'w') as md_file:
            md_file.write("# Analysis Summary\n\n")
            md_file.write(f"## Total Clusters: {len(clusters)}\n\n")
            for idx, cluster in enumerate(clusters):
                md_file.write(f"### Cluster {idx + 1}\n")
                md_file.write(f"- **Cluster Name**: {cluster['name']}\n")
                md_file.write(f"- **Number of Conversations**: {len(cluster['conversations'])}\n")
                md_file.write("- **Top Terms**:\n")
                for term in cluster['top_terms']:
                    md_file.write(f"  - {term}\n")
                md_file.write("\n#### Conversation Trees\n")
                for conv in cluster['conversations']:
                    self._write_conversation_tree(conv, md_file)
            logger.info(f"Exported analysis summary to {output_path}")

    def _write_conversation_tree(self, conversation, md_file):
        for comment in conversation['comments']:
            md_file.write(f"  - {comment['text']} (by {comment['user']})\n")

    def run_analysis(self):
        post_embeddings = [self.embed_content(post.content) for post in self.posts_data]
        post_embeddings = np.array(post_embeddings)
        cluster_labels = self.cluster_posts(post_embeddings)
        classifier = self.classify_posts(post_embeddings, cluster_labels)
        predicted_multi_labels = classifier.predict(post_embeddings)
        
        clustered_posts = {}
        for i, label in enumerate(cluster_labels):
            if label not in clustered_posts:
                clustered_posts[label] = []
            clustered_posts[label].append(self.posts_data[i])

        cluster_keywords = {}
        for label, posts in clustered_posts.items():
            all_text = " ".join([post.content for post in posts])
            keywords = self.extract_key_phrases(all_text)
            cluster_keywords[label] = keywords
        
        for label, keywords in cluster_keywords.items():
            logger.info(f"Cluster {label}: {keywords}")

        for post in self.posts_data:
            conversation_analysis = self.analyze_conversation_tree(post.post_id)
            post.conversation_analysis = conversation_analysis
            if conversation_analysis:
                logger.info(f"Conversation analysis for post {post['post_id']}: {conversation_analysis}")

        self.visualize_clusters(post_embeddings, cluster_labels)
        for post_id in self.conversation_trees:
            self.visualize_conversation_tree(post_id)
            break





