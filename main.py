import logging
from src.data_processor import GPUOptimizedProcessor
import torch
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt_tab')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function with GPU optimization"""
    try:
        # Configure paths
        posts_file = "data/gettingbigger_submissions.json"
        comments_file = "data/gettingbigger_comments.json"
        output_dir = "processed_data"
        
        # Log GPU availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Initialize and run the GPU-optimized processor
        processor = GPUOptimizedProcessor(posts_file, comments_file, output_dir)
        processor.process_data()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
