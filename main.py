import logging
from src.data_processor import GPUOptimizedProcessor, DataLoader  # Import DataLoader
import torch
import nltk
import typer
from rich.console import Console
from rich.table import Table
from datetime import datetime
from src.nlp.analyzer import RedditDataAnalyzer  # Import the analyzer


app = typer.Typer()
console = Console()

def display_data_stats(posts, comments):
    """Displays statistics about the loaded data in a rich table."""
    table = Table(title="Reddit Data Statistics")
    table.add_column("Data Type", justify="left", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_row("Posts", str(len(posts)))
    table.add_row("Comments", str(len(comments)))
    console.print(table)


@app.command()
def process(
        posts_file: str = "data/gettingbigger_submissions.json",
        comments_file: str = "data/gettingbigger_comments.json",
        output_dir: str = "processed_data",
):

    """Processes Reddit data to create conversation pairs."""
    try:
        # Configure logging (if not already configured)
        log_file = f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file)
            ]
        )
        logger = logging.getLogger(__name__)


        # Download NLTK resources (if needed)
        nltk.download('stopwords', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('punkt', quiet=True)  # Use punkt instead of punkt_tab
        nltk.download('averaged_perceptron_tagger', quiet=True)

        # Log GPU availability and information
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Load data first to display stats
        data_loader = DataLoader(posts_file, comments_file) # Initialize DataLoader
        raw_posts, raw_comments = data_loader.load_data()

        # Display data statistics
        display_data_stats(raw_posts, raw_comments)


        if typer.confirm("Do you want to start the processing pipeline?", default=True):
            processor = GPUOptimizedProcessor(posts_file, comments_file, output_dir)
            processor.process_data()
            analyzer = RedditDataAnalyzer(processor.processed_posts, processor.processed_comments, processor.posts_dict)
            analyzer.run_analysis()
            processing_stats = processor.get_processing_stats()
            
            logger.info("Processing statistics:")
            for key, value in processing_stats.items():
                logger.info(f"- {key}: {value}")
        else:
            logger.info("Processing pipeline aborted by the user.")

    except Exception as e:
        logger.error(f"Error in processing: {e}")
        raise



if __name__ == "__main__":
    app()