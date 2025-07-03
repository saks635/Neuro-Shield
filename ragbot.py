import pandas as pd
import numpy as np
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import List, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

# Colab-specific imports
try:
    from google.colab import files
    IN_COLAB = True
    print("✅ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("ℹ️ Running outside of Google Colab")

def upload_csv_file():
    """Helper function to upload CSV file in Google Colab"""
    if not IN_COLAB:
        print("This function is only available in Google Colab")
        return None

    print("📁 Please select your CSV file to upload:")
    uploaded = files.upload()

    if uploaded:
        filename = list(uploaded.keys())[0]
        print(f"✅ Uploaded: {filename}")
        return f"/content/{filename}"
    else:
        print("❌ No file uploaded")
        return None

class EEGAlzheimersRAG:
    def __init__(self, csv_file_path: str, use_sentence_transformers: bool = True):
        """
        Initialize the RAG system with CSV data and FAISS indexing

        Args:
            csv_file_path: Path to the CSV file containing questions and answers
            use_sentence_transformers: Whether to use sentence transformers (better) or TF-IDF
        """
        self.df = None
        self.faiss_index = None
        self.vectorizer = None
        self.sentence_model = None
        self.use_sentence_transformers = use_sentence_transformers
        self.question_texts = []
        self.embedding_dim = None

        print("Initializing EEG Alzheimer's RAG system...")
        self.load_data(csv_file_path)
        self.setup_embeddings()
        self.build_faiss_index()

    def load_data(self, csv_file_path: str):
        """Load and preprocess the CSV data"""
        try:
            # Try different encodings in case of encoding issues
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(csv_file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if self.df is None:
                raise ValueError("Could not read CSV file with any encoding")

            # Automatically detect question and answer columns
            columns = self.df.columns.tolist()
            question_col = None
            answer_col = None

            # Look for common question/answer column names
            for col in columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['question', 'q', 'query', 'ask']):
                    question_col = col
                elif any(word in col_lower for word in ['answer', 'a', 'response', 'reply']):
                    answer_col = col

            # If not found, use first two columns
            if question_col is None or answer_col is None:
                if len(columns) >= 2:
                    question_col = columns[0]
                    answer_col = columns[1]
                else:
                    raise ValueError("CSV must have at least 2 columns")

            self.question_col = question_col
            self.answer_col = answer_col

            # Clean the data
            self.df = self.df.dropna(subset=[question_col, answer_col])
            self.df[question_col] = self.df[question_col].astype(str)
            self.df[answer_col] = self.df[answer_col].astype(str)

            # Store cleaned question texts
            self.question_texts = [self.preprocess_text(q) for q in self.df[self.question_col]]

            print(f"✅ Loaded {len(self.df)} Q&A pairs from CSV")
            print(f"Question column: {question_col}")
            print(f"Answer column: {answer_col}")

        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep more characters for better semantic understanding
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        return text.strip()

    def setup_embeddings(self):
        """Set up embedding model (Sentence Transformers or TF-IDF)"""
        if self.df is None:
            raise ValueError("Data not loaded yet")

        if self.use_sentence_transformers:
            print("🔄 Loading Sentence Transformer model...")
            try:
                # Use a medical/scientific domain model if available, otherwise use general model
                model_options = [
                    'all-MiniLM-L6-v2',  # Fast and good general purpose
                    'all-mpnet-base-v2',  # Better quality, slower
                    'paraphrase-MiniLM-L6-v2'  # Good for semantic similarity
                ]

                for model_name in model_options:
                    try:
                        self.sentence_model = SentenceTransformer(model_name)
                        print(f"✅ Loaded model: {model_name}")
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to load {model_name}: {e}")
                        continue

                if self.sentence_model is None:
                    raise Exception("Could not load any sentence transformer model")

                # Get embedding dimension
                sample_embedding = self.sentence_model.encode(["test"])
                self.embedding_dim = sample_embedding.shape[1]
                print(f"Embedding dimension: {self.embedding_dim}")

            except Exception as e:
                print(f"⚠️ Sentence Transformers failed: {e}")
                print("🔄 Falling back to TF-IDF...")
                self.use_sentence_transformers = False
                self.setup_tfidf()
        else:
            self.setup_tfidf()

    def setup_tfidf(self):
        """Set up TF-IDF vectorizer as fallback"""
        print("🔄 Setting up TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )

        # Fit and get dimension
        sample_vectors = self.vectorizer.fit_transform(self.question_texts)
        self.embedding_dim = sample_vectors.shape[1]
        print(f"✅ TF-IDF setup complete. Dimension: {self.embedding_dim}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        if self.use_sentence_transformers and self.sentence_model:
            embeddings = self.sentence_model.encode(texts, show_progress_bar=False)
            return embeddings.astype('float32')
        else:
            # Use TF-IDF
            tfidf_matrix = self.vectorizer.transform(texts)
            return tfidf_matrix.toarray().astype('float32')

    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        print("🔄 Building FAISS index...")

        # Get embeddings for all questions
        question_embeddings = self.get_embeddings(self.question_texts)

        # Create FAISS index
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(question_embeddings)

        # Add vectors to index
        self.faiss_index.add(question_embeddings)

        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")

    def find_similar_questions(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str, str]]:
        """
        Find the most similar questions to the query using FAISS

        Args:
            query: User's question
            top_k: Number of similar questions to return

        Returns:
            List of tuples (index, similarity_score, question, answer)
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built yet")

        # Preprocess and get embedding for query
        processed_query = self.preprocess_text(query)
        query_embedding = self.get_embeddings([processed_query])

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity > 0.1:  # Threshold for minimum similarity
                results.append((
                    int(idx),
                    float(similarity),
                    self.df.iloc[idx][self.question_col],
                    self.df.iloc[idx][self.answer_col]
                ))

        return results

    def answer_question(self, query: str, top_k: int = 3) -> str:
        """
        Answer a question using RAG approach with FAISS

        Args:
            query: User's question
            top_k: Number of similar questions to consider

        Returns:
            Generated answer
        """
        if not query.strip():
            return "Please ask a question about EEG and Alzheimer's disease."

        try:
            # Find similar questions using FAISS
            similar_questions = self.find_similar_questions(query, top_k)

            if not similar_questions:
                return "❌ I couldn't find any relevant information in the database for your question. Please try rephrasing or ask about EEG and Alzheimer's related topics."

            # If the top match has very high similarity, return its answer directly
            if similar_questions[0][1] > 0.85:
                return f"🎯 **Direct Answer** (Confidence: {similar_questions[0][1]:.1%}):\n\n{similar_questions[0][3]}"

            # Otherwise, provide a comprehensive answer based on multiple similar questions
            response = f"🔍 **Found {len(similar_questions)} relevant answer(s):**\n\n"

            for i, (idx, similarity, question, answer) in enumerate(similar_questions):
                confidence = similarity * 100
                response += f"**📋 Result {i+1}** (Confidence: {confidence:.1f}%):\n"
                response += f"*❓ Similar Question:* {question}\n"
                response += f"*✅ Answer:* {answer}\n\n"
                response += "---\n\n"

            # Add a summary note
            if len(similar_questions) > 1:
                response += "💡 **Note:** Multiple relevant answers found. "
                response += "The answers above are ranked by similarity to your question. "
                response += "If you need more specific information, please refine your question."

            return response

        except Exception as e:
            return f"❌ An error occurred while processing your question: {str(e)}"

    def get_index_stats(self) -> dict:
        """Get statistics about the FAISS index"""
        return {
            "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "embedding_dimension": self.embedding_dim,
            "embedding_type": "Sentence Transformers" if self.use_sentence_transformers else "TF-IDF",
            "model_name": getattr(self.sentence_model, '_model_name', 'TF-IDF') if hasattr(self, 'sentence_model') and self.sentence_model else 'TF-IDF'
        }

def create_rag_interface(csv_file_path: str):
    """Create Gradio interface for the RAG system with FAISS"""

    try:
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag_system = EEGAlzheimersRAG(csv_file_path, use_sentence_transformers=True)

        def process_question(question: str, num_results: int = 3) -> str:
            if not question.strip():
                return "Please enter a question."
            return rag_system.answer_question(question, num_results)

        def get_dataset_info() -> str:
            stats = rag_system.get_index_stats()
            info = f"""
🔍 **Dataset Information:**
- **Total Q&A pairs:** {len(rag_system.df)}
- **Question column:** {rag_system.question_col}
- **Answer column:** {rag_system.answer_col}
- **FAISS index vectors:** {stats['total_vectors']}
- **Embedding dimension:** {stats['embedding_dimension']}
- **Embedding type:** {stats['embedding_type']}

📝 **Sample questions from dataset:**
"""

            # Show first 5 questions as examples
            for i in range(min(5, len(rag_system.df))):
                info += f"\n{i+1}. {rag_system.df.iloc[i][rag_system.question_col]}"

            return info

        def search_similar_questions(query: str, num_results: int = 5) -> str:
            """Search for similar questions without full answer formatting"""
            if not query.strip():
                return "Please enter a search query."

            try:
                similar = rag_system.find_similar_questions(query, num_results)
                if not similar:
                    return "No similar questions found."

                result = f"🔍 **Found {len(similar)} similar questions:**\n\n"
                for i, (idx, similarity, question, answer) in enumerate(similar):
                    result += f"**{i+1}.** (Similarity: {similarity:.3f})\n"
                    result += f"**Q:** {question}\n"
                    result += f"**A:** {answer[:200]}{'...' if len(answer) > 200 else ''}\n\n"

                return result
            except Exception as e:
                return f"Error: {str(e)}"

        # Create Gradio interface
        with gr.Blocks(title="EEG Alzheimer's RAG System with FAISS", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🧠 EEG Alzheimer's Question-Answer System (FAISS-Powered)

            This system uses **Retrieval-Augmented Generation (RAG)** with **FAISS** vector search to answer questions
            about EEG and Alzheimer's disease based on your dataset. FAISS provides fast and accurate similarity search
            using advanced embeddings.
            """)

            with gr.Tab("🤖 Ask Questions"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the EEG patterns in Alzheimer's disease?",
                            lines=2
                        )
                        num_results = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Number of results to retrieve"
                        )
                        submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Tips for better results:
                        - Ask specific questions about EEG and Alzheimer's
                        - Use medical terminology when appropriate
                        - Try different phrasings if results aren't optimal
                        - Increase number of results for broader context
                        """)

                answer_output = gr.Textbox(
                    label="📝 Answer",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )

                submit_btn.click(
                    fn=process_question,
                    inputs=[question_input, num_results],
                    outputs=answer_output
                )

                # Add example questions
                gr.Markdown("### 🎯 Example Questions:")
                example_questions = [
                    "What are the main EEG changes in Alzheimer's disease?",
                    "How does EEG help in early diagnosis of dementia?",
                    "What frequency bands are affected in Alzheimer's?",
                    "Can EEG differentiate between different types of dementia?",
                    "What are the limitations of EEG in Alzheimer's diagnosis?"
                ]

                with gr.Row():
                    for i, example in enumerate(example_questions):
                        if i < 3:  # First row
                            gr.Button(example, size="sm").click(
                                lambda x=example: x,
                                outputs=question_input
                            )

                with gr.Row():
                    for i, example in enumerate(example_questions):
                        if i >= 3:  # Second row
                            gr.Button(example, size="sm").click(
                                lambda x=example: x,
                                outputs=question_input
                            )

            with gr.Tab("🔍 Search Similar"):
                gr.Markdown("### Find questions similar to your query without full answer formatting")

                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter keywords or a question to find similar content",
                        lines=2
                    )
                    search_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of results"
                    )

                search_btn = gr.Button("🔍 Search Similar Questions", variant="secondary")
                search_output = gr.Textbox(
                    label="Similar Questions",
                    lines=12,
                    interactive=False,
                    show_copy_button=True
                )

                search_btn.click(
                    fn=search_similar_questions,
                    inputs=[search_input, search_results],
                    outputs=search_output
                )

            with gr.Tab("📊 Dataset Info"):
                gr.Markdown("### Dataset and FAISS Index Information")
                dataset_info = gr.Textbox(
                    label="Dataset Statistics",
                    value=get_dataset_info(),
                    lines=20,
                    interactive=False,
                    show_copy_button=True
                )

                refresh_btn = gr.Button("🔄 Refresh Info")
                refresh_btn.click(
                    fn=get_dataset_info,
                    outputs=dataset_info
                )

        return interface

    except Exception as e:
        # Create error interface if something goes wrong
        with gr.Blocks() as error_interface:
            gr.Markdown(f"""
            # ❌ Error Loading Dataset

            **Error:** {str(e)}

            **Possible solutions:**
            1. Make sure your CSV file exists and is readable
            2. Check that the CSV has at least 2 columns (questions and answers)
            3. Ensure the CSV file encoding is UTF-8, Latin-1, or CP1252
            4. Install required dependencies: `pip install sentence-transformers faiss-cpu`

            **Required file format:**
            - CSV file with headers
            - At least 2 columns (questions and answers)
            - Column names should contain 'question'/'q' and 'answer'/'a' (case insensitive)
            - Or the first two columns will be used as questions and answers
            """)

        return error_interface

def find_csv_files():
    """Find all CSV files in the current directory"""
    csv_files = []

    # Check /content directory (Colab)
    if os.path.exists("/content"):
        content_csvs = [f"/content/{f}" for f in os.listdir("/content") if f.endswith('.csv')]
        csv_files.extend(content_csvs)

    # Check current directory
    current_csvs = [f for f in os.listdir(".") if f.endswith('.csv')]
    csv_files.extend(current_csvs)

    return csv_files

def launch_rag_system(csv_file_path: str = None):
    """
    Launch the RAG system with the specified CSV file

    Args:
        csv_file_path: Path to your CSV file containing Q&A pairs
    """
    if csv_file_path is None:
        # Auto-detect CSV files
        csv_files = find_csv_files()

        if csv_files:
            csv_file_path = csv_files[0]
            print(f"✅ Auto-detected CSV file: {csv_file_path}")
        else:
            print("❌ No CSV files found!")
            print("\n🔧 To add your CSV file:")
            print("1. Upload via Colab file browser (left sidebar)")
            print("2. Or run: csv_path = upload_csv_file()")
            print("3. Then run: launch_rag_system(csv_path)")
            return None

    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        return None

    try:
        print(f"🚀 Launching RAG system with: {csv_file_path}")
        interface = create_rag_interface(csv_file_path)
        interface.launch(
            share=True,  # Creates a public link for sharing
            server_name="0.0.0.0",
            server_port=7860,
            debug=False,
            show_error=True
        )
        return interface
    except Exception as e:
        print(f"❌ Error launching RAG system: {str(e)}")
        return None

# Convenience functions for easy use
def quick_start():
    """Quick start function - upload and launch in one go"""
    print("🚀 Quick Start Guide:")
    if IN_COLAB:
        print("1. Uploading CSV file...")
        csv_path = upload_csv_file()
        if csv_path:
            print("2. Launching RAG system...")
            return launch_rag_system(csv_path)
        else:
            print("❌ No file uploaded")
            return None
    else:
        print("Upload function only available in Colab")
        return launch_rag_system()

def start_with_file(file_path: str):
    """Start RAG system with specific file path"""
    return launch_rag_system(file_path)

# Auto-run when imported in Colab
print("="*60)
print("🧠 EEG Alzheimer's RAG System - Ready!")
print("="*60)
print("\n📚 Available functions:")
print("• quick_start() - Upload file and launch")
print("• launch_rag_system() - Auto-detect CSV and launch")
print("• start_with_file('/path/to/file.csv') - Launch with specific file")
print("• upload_csv_file() - Upload CSV file")
print("\n🚀 Quick Start:")
print("Run: quick_start()")
print("="*60)
