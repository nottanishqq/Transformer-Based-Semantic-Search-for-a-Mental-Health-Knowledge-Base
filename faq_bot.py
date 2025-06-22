import os
import logging
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import json

class FaqBot:
    def __init__(self, model_dir='model_assets', model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model_dir = model_dir
        self.vectors_path = os.path.join(self.model_dir, 'question_vectors.npy')
        self.df_path = os.path.join(self.model_dir, 'processed_data.pkl')
        self.model_name = model_name
        
        self.sentence_model = None
        self.question_vectors = None
        self.qa_data = None
    
    def load(self):
        logging.info(f"Loading model assets from '{self.model_dir}'...")
        if not os.path.isdir(self.model_dir):
            logging.error(f"Model directory not found at '{self.model_dir}'. Please run train.py first.")
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        try:
            logging.info(f"Loading Sentence Transformer model: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name)
            
            self.question_vectors = np.load(self.vectors_path)
            with open(self.df_path, 'rb') as f:
                self.qa_data = pickle.load(f)
            logging.info("Bot loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model assets. Error: {e}")
            raise

    def get_answer(self, user_question: str, threshold: float = 0.7, top_k: int = 5) -> Dict:
        if not self.sentence_model:
            raise RuntimeError("Bot is not loaded. Please call the `load()` method first.")
            
        if not self.qa_data:
            raise RuntimeError("QA data is not loaded. Please call the `load()` method first.")
            
        if not user_question.strip():
            return {
                "answer": "I'm sorry, I couldn't understand your question. Please try rephrasing.",
                "similarity": 0.0,
                "confidence": "low",
                "top_matches": []
            }
        
        user_embedding = self.sentence_model.encode([user_question])
        
        similarities = cosine_similarity(user_embedding, self.question_vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        top_matches = []
        for idx, sim in zip(top_indices, top_similarities):
            if idx < len(self.qa_data) and self.qa_data[idx] is not None:
                top_matches.append({
                    "question": self.qa_data[idx]['Question'],
                    "answer": self.qa_data[idx]['Answer'],
                    "similarity": float(sim),
                    "faq_id": int(idx)
                })
        best_match_idx = top_indices[0]
        max_similarity = top_similarities[0]
        
        if max_similarity > 0.8:
            confidence = "high"
        elif max_similarity > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        logging.debug(f"Max similarity: {max_similarity:.4f}")

        if max_similarity > threshold:
            return {
                "answer": self.qa_data[best_match_idx]['Answer'],
                "similarity": float(max_similarity),
                "confidence": confidence,
                "faq_id": int(best_match_idx),
                "top_matches": top_matches
            }
        else:
            return {
                "answer": "I'm sorry, I don't have a relevant answer. Could you ask in a different way?",
                "similarity": float(max_similarity),
                "confidence": "low",
                "top_matches": top_matches
            }
    
    def get_top_k_answers(self, user_question: str, k: int = 5) -> List[Dict]:
        if not self.sentence_model:
            raise RuntimeError("Bot is not loaded. Please call the `load()` method first.")
        
        if not self.qa_data:
            raise RuntimeError("QA data is not loaded. Please call the `load()` method first.")
        
        if not user_question.strip():
            return []
        
        user_embedding = self.sentence_model.encode([user_question])
        
        similarities = cosine_similarity(user_embedding, self.question_vectors)[0]
                
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        
        results = []
        for idx, sim in zip(top_indices, top_similarities):
            if idx < len(self.qa_data) and self.qa_data[idx] is not None:
                results.append({
                    "faq_id": int(idx),
                    "question": self.qa_data[idx]['Question'],
                    "answer": self.qa_data[idx]['Answer'],
                    "similarity": float(sim)
                })
        
        return results