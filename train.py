import pandas as pd
import numpy as np
import os
import logging
import pickle
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaqBotTrainer:
    def __init__(self, dataset_path: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', model_dir: str = 'model_assets'):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.model_dir = model_dir
        
        self.vectors_path = os.path.join(self.model_dir, 'question_vectors.npy')
        self.df_path = os.path.join(self.model_dir, 'processed_data.pkl')
        
        self.sentence_model = None

    def train_and_save(self):
        logging.info("Starting new training process with Sentence Transformers...")
        
        try:
            df = pd.read_csv(self.dataset_path)
            if 'input' in df.columns and 'output' in df.columns:
                df.rename(columns={'input': 'Question', 'output': 'Answer'}, inplace=True)
            df.dropna(subset=['Question', 'Answer'], inplace=True)
            if 'Question' not in df.columns or 'Answer' not in df.columns:
                raise ValueError("CSV must have 'Question'/'Answer' or 'input'/'output' columns.")
        except FileNotFoundError:
            logging.error(f"Dataset file not found at '{self.dataset_path}'. Cannot train.")
            raise
        
        logging.info(f"Loaded {len(df)} FAQ entries")
        
        logging.info(f"Loading Sentence Transformer model: {self.model_name}")
        self.sentence_model = SentenceTransformer(self.model_name)
        
        logging.info("Encoding all FAQ questions...")
        questions = df['Question'].tolist()
        question_vectors = self.sentence_model.encode(questions, show_progress_bar=True)
        
        logging.info(f"Saving model assets to '{self.model_dir}'...")
        os.makedirs(self.model_dir, exist_ok=True)
        
        np.save(self.vectors_path, question_vectors)
        
        with open(self.df_path, 'wb') as f:
            records = [{'Question': row['Question'], 'Answer': row['Answer']} for _, row in df.iterrows()]
            pickle.dump(records, f)
        
        logging.info("Training complete. Artifacts saved successfully.")
        
        logging.info(f"Model: {self.model_name}")
        if isinstance(question_vectors, np.ndarray):
            vector_dim = question_vectors.shape[1]
        elif isinstance(question_vectors, list) and len(question_vectors) > 0:
            vector_dim = len(question_vectors[0])
        else:
            vector_dim = "unknown"
        logging.info(f"Vector dimension: {vector_dim}")
        logging.info(f"Number of FAQ entries: {len(df)}")
        
        return df, question_vectors

def create_test_set(df: pd.DataFrame, num_samples: int = 100) -> List[Dict]:
    test_cases = []
    
    sample_size = min(num_samples // 2, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    for idx, row in sampled_df.iterrows():
        try:
            if idx is not None:
                faq_id = int(str(idx))
            else:
                faq_id = 0
        except (ValueError, TypeError):
            faq_id = 0
            
        test_cases.append({
            "query": row['Question'],
            "expected_faq_id": faq_id,
            "type": "exact_match"
        })
    
    paraphrases = [
        ("I think I might be developing a substance abuse problem", "I'm worried about my alcohol consumption"),
        ("I've been feeling really anxious lately", "My anxiety has been getting worse"),
        ("I can't seem to get out of bed in the morning", "I'm having trouble with morning motivation"),
        ("I feel like I'm always sad", "I'm experiencing persistent sadness"),
        ("I'm having trouble sleeping", "I can't fall asleep at night"),
        ("I feel overwhelmed by everything", "Everything feels too much to handle"),
        ("I don't want to see anyone", "I'm avoiding social contact"),
        ("I keep having negative thoughts", "My mind is full of negative thinking"),
        ("I'm not eating properly", "I've lost my appetite"),
        ("I feel like giving up", "I'm losing hope")
    ]
    
    for paraphrase, original in paraphrases:
        best_match = None
        best_score = 0
        
        for idx, row in df.iterrows():
            question = str(row['Question']).lower()
            score = 0
            for word in paraphrase.lower().split():
                if word in question:
                    score += 1
            if score > best_score:
                best_score = score
                best_match = idx
        
        if best_match is not None:
            try:
                if best_match is not None:
                    faq_id = int(str(best_match))
                else:
                    faq_id = 0
            except (ValueError, TypeError):
                faq_id = 0
                
            test_cases.append({
                "query": paraphrase,
                "expected_faq_id": faq_id,
                "type": "paraphrase"
            })
    
    edge_cases = [
        ("", None, "empty_query"),
        ("hello", None, "greeting"),
        ("thank you", None, "gratitude"),
        ("what is mental health?", None, "general_question"),
        ("how do I get help?", None, "help_request")
    ]
    
    for query, expected_id, case_type in edge_cases:
        test_cases.append({
            "query": query,
            "expected_faq_id": expected_id,
            "type": case_type
        })
    
    return test_cases

def evaluate_model(faq_bot, test_cases: List[Dict], k_values: List[int] = [1, 3, 5]) -> Dict:
    logging.info("Starting evaluation...")
    
    mrr_scores = []
    recall_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        expected_faq_id = test_case["expected_faq_id"]
        
        if not query.strip() or expected_faq_id is None:
            continue
            
        top_k_results = faq_bot.get_top_k_answers(query, k=max(k_values))
        
        if not top_k_results:
            continue
        
        for rank, result in enumerate(top_k_results, 1):
            if result["faq_id"] == expected_faq_id:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)
        
        retrieved_ids = [result["faq_id"] for result in top_k_results]
        
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            
            if expected_faq_id in top_k_ids:
                recall_scores[k].append(1.0)
            else:
                recall_scores[k].append(0.0)
            
            if expected_faq_id in top_k_ids:
                precision_scores[k].append(1.0)
            else:
                precision_scores[k].append(0.0)
    
    results = {
        "num_test_cases": len(test_cases),
        "num_valid_cases": len(mrr_scores),
        "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
        "recall": {f"recall@{k}": np.mean(recall_scores[k]) for k in k_values},
        "precision": {f"precision@{k}": np.mean(precision_scores[k]) for k in k_values}
    }
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the FAQ Bot model using Sentence Transformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter      
    )

    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the input CSV dataset for training. The CSV must contain 'Question' and 'Answer' columns."
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model to use"
    )
    
    parser.add_argument(
        "--create-test-set",
        action="store_true",
        help="Create and save a test set for evaluation"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training"
    )

    args = parser.parse_args()
    
    trainer = FaqBotTrainer(dataset_path=args.dataset_path, model_name=args.model_name)
    df, question_vectors = trainer.train_and_save()
    
    if args.create_test_set:
        logging.info("Creating test set...")
        test_cases = create_test_set(df, num_samples=100)
        
        test_set_path = os.path.join(trainer.model_dir, 'test_set.json')
        with open(test_set_path, 'w') as f:
            json.dump(test_cases, f, indent=2)
        
        logging.info(f"Test set saved to {test_set_path}")
        logging.info(f"Created {len(test_cases)} test cases")
    
    if args.evaluate:
        logging.info("Running evaluation...")
        
        from faq_bot import FaqBot
        
        bot = FaqBot(model_dir=trainer.model_dir, model_name=args.model_name)
        bot.load()
        
        if not args.create_test_set:
            test_cases = create_test_set(df, num_samples=100)
        else:
            with open(test_set_path, 'r') as f:
                test_cases = json.load(f)
        
        results = evaluate_model(bot, test_cases)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of test cases: {results['num_test_cases']}")
        print(f"Valid test cases: {results['num_valid_cases']}")
        print(f"Mean Reciprocal Rank (MRR): {results['mrr']:.4f}")
        print("\nRecall@k:")
        for k, recall in results['recall'].items():
            print(f"  {k}: {recall:.4f}")
        print("\nPrecision@k:")
        for k, precision in results['precision'].items():
            print(f"  {k}: {precision:.4f}")
        print("="*50)
        
        eval_path = os.path.join(trainer.model_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Evaluation results saved to {eval_path}")
    
    print("\nTraining finished. The `model_assets` directory is ready for your application.")
