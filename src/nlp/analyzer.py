from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

class NLPAnalyzer:
    def __init__(self, batch_size: int = 64, use_gpu: bool = True):
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