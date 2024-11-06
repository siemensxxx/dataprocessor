import json
import pandas as pd
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize logging and console output
logger = logging.getLogger(__name__)
console = Console()

class ExecutiveSummaryGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.topic_summary_file = self.output_dir / "topic_summary.json"
        self.language_analysis_file = self.output_dir / "language_analysis.json"
        self.document_topic_matrix_file = self.output_dir / "document_topic_matrix.csv"

        # Load the data
        self.topic_summary = self._load_json(self.topic_summary_file)
        self.language_analysis = self._load_json(self.language_analysis_file)
        self.document_topic_matrix = pd.read_csv(self.document_topic_matrix_file)

    def _load_json(self, file_path: Path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return {}

    def generate_summary(self):
        """Generate and save an executive summary report."""
        console.print("[bold cyan]Executive Summary Report[/bold cyan]")
        
        # 1. Top 5 Topics with Key Terms
        self._generate_top_topics_summary()

        # 2. Sentiment Overview
        self._generate_sentiment_overview()

        # 3. Language Style Insights
        self._generate_language_style_insights()

        # 4. Save summary to text file
        self._save_summary_to_file()

    def _generate_top_topics_summary(self):
        """Generate a summary of the top 5 topics and display key terms."""
        console.print("\n[bold green]Top 5 Topics[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Topic ID", style="dim", width=12)
        table.add_column("Key Terms", width=40)
        
        # Get top 5 topics based on weight
        for topic in self.topic_summary[:5]:
            topic_id = str(topic['topic_id'])
            terms = ', '.join([term['term'] for term in topic['terms']])
            table.add_row(topic_id, terms)
        
        console.print(table)

    def _generate_sentiment_overview(self):
        """Generate a sentiment overview from the document-topic matrix."""
        console.print("\n[bold green]Sentiment Overview by Topic[/bold green]")

        # Example sentiment extraction (you'll need to calculate sentiment per topic based on document scores)
        # Assuming `self.document_topic_matrix` contains a column "sentiment" with positive/negative/neutral scores
        sentiments = self.document_topic_matrix.groupby('topic_id')['sentiment'].mean().reset_index()
        sentiments.columns = ['Topic ID', 'Average Sentiment']

        # Plotting sentiment overview
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Topic ID', y='Average Sentiment', data=sentiments)
        plt.title('Average Sentiment by Topic')
        plt.ylabel('Average Sentiment')
        plt.xlabel('Topic ID')

        # Save and show plot
        sentiment_plot_path = self.output_dir / "sentiment_overview.png"
        plt.savefig(sentiment_plot_path)
        console.print(f"\nSentiment overview saved to {sentiment_plot_path}")
        plt.show()

    def _generate_language_style_insights(self):
        """Generate insights from the language analysis."""
        console.print("\n[bold green]Language Style Insights[/bold green]")
        slang_terms = self.language_analysis.get('slang_terms', {})
        common_phrases = self.language_analysis.get('common_phrases', {}).get('2_grams', {})

        # Display common phrases and slang terms
        console.print(f"[bold]Top 5 Common 2-Gram Phrases[/bold]:")
        for phrase, freq in list(common_phrases.items())[:5]:
            console.print(f"- {phrase}: {freq} times")

        console.print(f"\n[bold]Top 5 Slang Terms[/bold]:")
        for term, freq in list(slang_terms.items())[:5]:
            console.print(f"- {term}: {freq} times")

    def _save_summary_to_file(self):
        """Save the executive summary to a text file."""
        summary_file = self.output_dir / "executive_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Executive Summary Report\n")
            f.write("=======================\n\n")
            f.write("1. Top 5 Topics\n")
            for topic in self.topic_summary[:5]:
                terms = ', '.join([term['term'] for term in topic['terms']])
                f.write(f"Topic {topic['topic_id']}: {terms}\n")
            
            f.write("\n2. Sentiment Overview\n")
            f.write("See the sentiment_overview.png plot for the average sentiment per topic.\n")

            f.write("\n3. Language Style Insights\n")
            slang_terms = self.language_analysis.get('slang_terms', {})
            common_phrases = self.language_analysis.get('common_phrases', {}).get('2_grams', {})
            f.write("Top 5 Common 2-Gram Phrases:\n")
            for phrase, freq in list(common_phrases.items())[:5]:
                f.write(f"- {phrase}: {freq} times\n")
            f.write("\nTop 5 Slang Terms:\n")
            for term, freq in list(slang_terms.items())[:5]:
                f.write(f"- {term}: {freq} times\n")

        console.print(f"\n[bold green]Executive summary saved to {summary_file}[/bold green]")

if __name__ == "__main__":
    output_dir = "processed_data"
    summary_generator = ExecutiveSummaryGenerator(output_dir)
    summary_generator.generate_summary()
