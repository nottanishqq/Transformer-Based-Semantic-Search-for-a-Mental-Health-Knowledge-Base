#!/usr/bin/env python3

import os
import json
import logging
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple
from faq_bot import FaqBot
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_test_set(test_set_path: str) -> List[Dict]:
    with open(test_set_path, 'r') as f:
        return json.load(f)

def create_comprehensive_test_set() -> List[Dict]:
    test_cases = []
    
    mental_health_queries = [
        "I think I might be developing a substance abuse problem",
        "I've been feeling really anxious lately",
        "I can't seem to get out of bed in the morning",
        "I feel like I'm always sad",
        "I'm having trouble sleeping",
        "I feel overwhelmed by everything",
        "I don't want to see anyone",
        "I keep having negative thoughts",
        "I'm not eating properly",
        "I feel like giving up",
        "I'm worried about my alcohol consumption",
        "My anxiety has been getting worse",
        "I'm having trouble with morning motivation",
        "I'm experiencing persistent sadness",
        "I can't fall asleep at night",
        "Everything feels too much to handle",
        "I'm avoiding social contact",
        "My mind is full of negative thinking",
        "I've lost my appetite",
        "I'm losing hope"
    ]
    
    for i, query in enumerate(mental_health_queries):
        test_cases.append({
            "query": query,
            "expected_faq_id": None,
            "type": "mental_health_query",
            "category": "substance_abuse" if "substance" in query.lower() or "alcohol" in query.lower() else
                       "anxiety" if "anxious" in query.lower() or "anxiety" in query.lower() else
                       "depression" if "sad" in query.lower() or "depression" in query.lower() or "hope" in query.lower() else
                       "sleep" if "sleep" in query.lower() or "bed" in query.lower() else
                       "social" if "social" in query.lower() or "anyone" in query.lower() else
                       "general_mental_health"
        })
    
    edge_cases = [
        ("", None, "empty_query"),
        ("hello", None, "greeting"),
        ("thank you", None, "gratitude"),
        ("what is mental health?", None, "general_question"),
        ("how do I get help?", None, "help_request"),
        ("what's the weather like?", None, "unrelated"),
        ("I need to buy groceries", None, "unrelated"),
        ("Can you help me with math?", None, "unrelated"),
        ("What time is it?", None, "unrelated"),
        ("I love pizza", None, "unrelated")
    ]
    
    for query, expected_id, case_type in edge_cases:
        test_cases.append({
            "query": query,
            "expected_faq_id": expected_id,
            "type": case_type,
            "category": "edge_case"
        })
    
    return test_cases

def evaluate_model_detailed(faq_bot: FaqBot, test_cases: List[Dict], k_values: List[int] = [1, 3, 5]) -> Dict:
    logging.info("Starting comprehensive evaluation...")
    
    results = {
        "overall_metrics": {},
        "by_type": defaultdict(lambda: {"correct": 0, "total": 0, "mrr": []}),
        "by_difficulty": defaultdict(lambda: {"correct": 0, "total": 0, "mrr": []}),
        "error_analysis": [],
        "top_matches_analysis": [],
        "threshold_analysis": {}
    }
    
    mrr_scores = []
    recall_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        expected_faq_id = test_case["expected_faq_id"]
        case_type = test_case.get("type", "unknown")
        difficulty = test_case.get("difficulty", "unknown")
        
        if not query.strip() or expected_faq_id is None:
            continue
            
        top_k_results = faq_bot.get_top_k_answers(query, k=max(k_values))
        
        if not top_k_results:
            results["by_type"][case_type]["total"] += 1
            results["by_difficulty"][difficulty]["total"] += 1
            continue
        
        mrr = 0.0
        for rank, result in enumerate(top_k_results, 1):
            if result["faq_id"] == expected_faq_id:
                mrr = 1.0 / rank
                break
        
        mrr_scores.append(mrr)
        results["by_type"][case_type]["mrr"].append(mrr)
        results["by_difficulty"][difficulty]["mrr"].append(mrr)
        
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
        
        if expected_faq_id == top_k_results[0]["faq_id"]:
            results["by_type"][case_type]["correct"] += 1
            results["by_difficulty"][difficulty]["correct"] += 1
        
        results["by_type"][case_type]["total"] += 1
        results["by_difficulty"][difficulty]["total"] += 1
        
        if expected_faq_id != top_k_results[0]["faq_id"]:
            results["error_analysis"].append({
                "query": query,
                "expected_id": expected_faq_id,
                "predicted_id": top_k_results[0]["faq_id"],
                "expected_similarity": top_k_results[0]["similarity"],
                "type": case_type,
                "difficulty": difficulty,
                "top_3_predictions": [(r["faq_id"], r["similarity"]) for r in top_k_results[:3]]
            })
        
        results["top_matches_analysis"].append({
            "query": query,
            "expected_id": expected_faq_id,
            "top_match_id": top_k_results[0]["faq_id"],
            "top_match_similarity": top_k_results[0]["similarity"],
            "is_correct": expected_faq_id == top_k_results[0]["faq_id"],
            "type": case_type,
            "difficulty": difficulty
        })
    
    results["overall_metrics"] = {
        "num_test_cases": len(test_cases),
        "num_valid_cases": len(mrr_scores),
        "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
        "recall": {f"recall@{k}": np.mean(recall_scores[k]) for k in k_values},
        "precision": {f"precision@{k}": np.mean(precision_scores[k]) for k in k_values}
    }
    
    for category, data in results["by_type"].items():
        if data["total"] > 0:
            data["accuracy"] = data["correct"] / data["total"]
            data["avg_mrr"] = np.mean(data["mrr"]) if data["mrr"] else 0.0
    
    for category, data in results["by_difficulty"].items():
        if data["total"] > 0:
            data["accuracy"] = data["correct"] / data["total"]
            data["avg_mrr"] = np.mean(data["mrr"]) if data["mrr"] else 0.0
    
    similarities = [item["top_match_similarity"] for item in results["top_matches_analysis"]]
    correct_similarities = [item["top_match_similarity"] for item in results["top_matches_analysis"] if item["is_correct"]]
    incorrect_similarities = [item["top_match_similarity"] for item in results["top_matches_analysis"] if not item["is_correct"]]
    
    results["threshold_analysis"] = {
        "all_similarities": {
            "mean": np.mean(similarities),
            "std": np.std(similarities),
            "min": np.min(similarities),
            "max": np.max(similarities)
        },
        "correct_similarities": {
            "mean": np.mean(correct_similarities) if correct_similarities else 0,
            "std": np.std(correct_similarities) if correct_similarities else 0,
            "min": np.min(correct_similarities) if correct_similarities else 0,
            "max": np.max(correct_similarities) if correct_similarities else 0
        },
        "incorrect_similarities": {
            "mean": np.mean(incorrect_similarities) if incorrect_similarities else 0,
            "std": np.std(incorrect_similarities) if incorrect_similarities else 0,
            "min": np.min(incorrect_similarities) if incorrect_similarities else 0,
            "max": np.max(incorrect_similarities) if incorrect_similarities else 0
        }
    }
    
    return results

def print_detailed_results(results: Dict):
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    
    overall = results["overall_metrics"]
    print(f"\nOVERALL METRICS:")
    print(f"  Number of test cases: {overall['num_test_cases']}")
    print(f"  Valid test cases: {overall['num_valid_cases']}")
    print(f"  Mean Reciprocal Rank (MRR): {overall['mrr']:.4f}")
    
    print(f"\nRECALL METRICS:")
    for k, recall in overall['recall'].items():
        print(f"  {k}: {recall:.4f}")
    
    print(f"\nPRECISION METRICS:")
    for k, precision in overall['precision'].items():
        print(f"  {k}: {precision:.4f}")
    
    print(f"\nPERFORMANCE BY TEST TYPE:")
    for case_type, data in results["by_type"].items():
        if data["total"] > 0:
            accuracy = data["correct"] / data["total"]
            print(f"  {case_type}: {accuracy:.4f} ({data['correct']}/{data['total']})")
            if data["mrr"]:
                print(f"    Avg MRR: {data['avg_mrr']:.4f}")
    
    print(f"\nPERFORMANCE BY DIFFICULTY:")
    for difficulty, data in results["by_difficulty"].items():
        if data["total"] > 0:
            accuracy = data["correct"] / data["total"]
            print(f"  {difficulty}: {accuracy:.4f} ({data['correct']}/{data['total']})")
            if data["mrr"]:
                print(f"    Avg MRR: {data['avg_mrr']:.4f}")
    
    print(f"\nSIMILARITY THRESHOLD ANALYSIS:")
    threshold = results["threshold_analysis"]
    print(f"  Correct predictions: {threshold['correct_similarities']['mean']:.4f} ± {threshold['correct_similarities']['std']:.4f}")
    print(f"  Incorrect predictions: {threshold['incorrect_similarities']['mean']:.4f} ± {threshold['incorrect_similarities']['std']:.4f}")
    
    print(f"\nERROR ANALYSIS (Top 5 failures):")
    error_cases = sorted(results["error_analysis"], key=lambda x: x["expected_similarity"], reverse=True)[:5]
    for i, error in enumerate(error_cases, 1):
        print(f"  {i}. Query: '{error['query']}'")
        print(f"     Expected: {error['expected_id']}, Predicted: {error['predicted_id']}")
        print(f"     Similarity: {error['expected_similarity']:.4f}")
        print(f"     Type: {error['type']}")
        print()

def suggest_improvements(results: Dict) -> List[str]:
    suggestions = []
    
    overall = results["overall_metrics"]
    if overall["mrr"] < 0.7:
        suggestions.append("Overall MRR is below 0.7. Consider:")
        suggestions.append("   - Using a different sentence transformer model")
        suggestions.append("   - Improving data quality and preprocessing")
        suggestions.append("   - Adding more training examples")
    
    for case_type, data in results["by_type"].items():
        if data["total"] > 0 and data["accuracy"] < 0.5:
            suggestions.append(f"{case_type} accuracy is low ({data['accuracy']:.2f}). Consider:")
            if case_type == "mental_health_query":
                suggestions.append("   - Adding more mental health specific examples")
                suggestions.append("   - Fine-tuning on domain-specific vocabulary")
            elif case_type == "paraphrase":
                suggestions.append("   - Adding more paraphrased versions of questions")
                suggestions.append("   - Using data augmentation techniques")
            else:
                suggestions.append("   - Adding more examples for this query type")
                suggestions.append("   - Adding domain-specific vocabulary")
    
    for difficulty, data in results["by_difficulty"].items():
        if data["total"] > 0 and data["accuracy"] < 0.3:
            suggestions.append(f"{difficulty} difficulty cases are challenging ({data['accuracy']:.2f}). Consider:")
            if difficulty == "hard":
                suggestions.append("   - Adding more complex training examples")
                suggestions.append("   - Using a larger model")
            elif difficulty == "medium":
                suggestions.append("   - Improving preprocessing for medium complexity queries")
            else:
                suggestions.append("   - Adding fallback responses for unclear queries")
    
    threshold = results["threshold_analysis"]
    correct_mean = threshold["correct_similarities"]["mean"]
    incorrect_mean = threshold["incorrect_similarities"]["mean"]
    
    if correct_mean - incorrect_mean < 0.1:
        suggestions.append("Similarity scores don't clearly separate correct/incorrect predictions.")
        suggestions.append("   - Consider adjusting the similarity threshold")
        suggestions.append("   - Review the model's embedding quality")
    
    return suggestions

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the FAQ Bot using Sentence Transformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model_assets",
        help="Directory containing model assets"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name"
    )
    
    parser.add_argument(
        "--test-set",
        type=str,
        help="Path to test set JSON file (optional)"
    )
    
    parser.add_argument(
        "--create-test-set",
        action="store_true",
        help="Create and save a comprehensive test set"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="model_assets/detailed_evaluation.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    logging.info("Loading FAQ bot...")
    bot = FaqBot(model_dir=args.model_dir, model_name=args.model_name)
    bot.load()
    
    if args.test_set:
        test_cases = load_test_set(args.test_set)
        logging.info(f"Loaded {len(test_cases)} test cases from {args.test_set}")
    else:
        test_cases = create_comprehensive_test_set()
        logging.info(f"Created {len(test_cases)} test cases")
    
    if args.create_test_set:
        test_set_path = os.path.join(args.model_dir, 'comprehensive_test_set.json')
        with open(test_set_path, 'w') as f:
            json.dump(test_cases, f, indent=2)
        logging.info(f"Test set saved to {test_set_path}")
    
    results = evaluate_model_detailed(bot, test_cases)
    
    print_detailed_results(results)
    
    suggestions = suggest_improvements(results)
    if suggestions:
        print("\nIMPROVEMENT SUGGESTIONS:")
        for suggestion in suggestions:
            print(suggestion)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main() 