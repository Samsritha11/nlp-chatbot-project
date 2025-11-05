import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import glob
import os

class ChatBot:
    def __init__(self):
        # Load Sentence Transformer model for semantic search
        print("🔍 Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Automatically detect CSV file
        files = glob.glob("chatbot*.csv")
        if not files:
            raise FileNotFoundError("❌ No chatbot dataset file found!")
        dataset_file = files[0]
        print(f"✅ Using dataset: {os.path.basename(dataset_file)}")

        # Load and normalize dataset
        df = pd.read_csv(dataset_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'question' in df.columns and 'answer' in df.columns:
            df.rename(columns={'question': 'pattern', 'answer': 'response'}, inplace=True)
        elif 'patterns' in df.columns and ('tag' in df.columns or 'tags' in df.columns):
            df.rename(columns={'patterns': 'pattern', 'tag': 'intent', 'tags': 'intent'}, inplace=True)
            if 'responses' in df.columns:
                df.rename(columns={'responses': 'response'}, inplace=True)

        # Drop empty rows
        df.dropna(subset=['pattern', 'response'], inplace=True)
        self.df = df

        # Precompute embeddings for all questions
        print("⚙️ Computing question embeddings...")
        self.question_embeddings = self.model.encode(self.df['pattern'].tolist(), convert_to_tensor=True)
        print("✅ Chatbot ready!")

    def get_response(self, user_input):
        # Compute embedding for user input
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)

        # Find best match via cosine similarity
        cosine_scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
        best_match_id = torch.argmax(cosine_scores).item()

        best_question = self.df.iloc[best_match_id]['pattern']
        best_answer = self.df.iloc[best_match_id]['response']

        print(f"Matched: '{best_question}'  (Score: {cosine_scores[best_match_id]:.2f})")
        return best_answer
