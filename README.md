# Mental Health FAQ Bot with Sentence Transformers

A sophisticated FAQ bot for mental health counseling that uses state-of-the-art Sentence Transformers for semantic similarity matching. This implementation replaces the previous Word2Vec and NLTK pipeline with more powerful pre-trained language models.

## 🚀 Features

- **Advanced Semantic Matching**: Uses Sentence Transformers (all-MiniLM-L6-v2) for superior semantic understanding
- **Comprehensive Evaluation**: Built-in evaluation metrics including MRR, Recall@k, and Precision@k
- **Interactive Testing**: Command-line interface for testing and evaluation
- **Web Interface**: Streamlit-based web application for easy interaction
- **Confidence Scoring**: Provides confidence levels and similarity scores for responses
- **Top-K Results**: Returns multiple relevant answers with similarity scores

## 📊 Performance Metrics

The bot is evaluated using standard information retrieval metrics:

- **Mean Reciprocal Rank (MRR)**: Measures the quality of ranking
- **Recall@k**: Fraction of relevant documents retrieved in top-k results
- **Precision@k**: Fraction of retrieved documents that are relevant
- **Category Analysis**: Performance breakdown by query type

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-faq-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (if not already installed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🏃‍♂️ Quick Start

### 1. Training the Model

Train the FAQ bot with your dataset:

```bash
python train.py Synthetic_Data_Simplified.csv --create-test-set --evaluate
```

Options:
- `--model-name`: Choose a different Sentence Transformer model (default: all-MiniLM-L6-v2)
- `--create-test-set`: Generate a test set for evaluation
- `--evaluate`: Run evaluation after training

### 2. Running the Web Interface

```bash
streamlit run streamlit_app.py
```

### 3. Interactive Testing

```bash
python evaluate.py --interactive
```

### 4. Comprehensive Evaluation

```bash
python evaluate.py --create-test-set --k-values 1 3 5 10
```

## 📁 Project Structure

```
mental-health-faq-bot/
├── faq_bot.py              # Main FAQ bot implementation
├── train.py                # Training script with evaluation
├── evaluate.py             # Standalone evaluation script
├── streamlit_app.py        # Web interface
├── requirements.txt        # Python dependencies
├── Synthetic_Data_Simplified.csv  # Training dataset
├── model_assets/           # Generated model files
│   ├── question_vectors.npy
│   ├── processed_data.pkl
│   ├── test_set.json
│   ├── better_test_set.json
│   ├── evaluation_results.json
│   └── detailed_evaluation.json
└── README.md
```

## 🔧 Configuration

### Model Options

The bot supports various Sentence Transformer models:

- `sentence-transformers/all-MiniLM-L6-v2` (default) - Fast and efficient, best overall performance
- `sentence-transformers/all-mpnet-base-v2` - Higher quality, slower
- `msmarco-distilbert-base-v4` - Optimized for question-answering
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

### Training Parameters

```python
# Example: Train with a different model
trainer = FaqBotTrainer(
    dataset_path="your_data.csv",
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Recommended model
)
```

## 📈 Evaluation Results

The evaluation script provides comprehensive metrics:

```
EVALUATION RESULTS
============================================================
Model: sentence-transformers/all-MiniLM-L6-v2
Number of test cases: 30
Valid test cases: 25
Mean Reciprocal Rank (MRR): 0.8234

Recall@k:
  recall@1: 0.7600
  recall@3: 0.8800
  recall@5: 0.9200
  recall@10: 0.9600

Precision@k:
  precision@1: 0.7600
  precision@3: 0.2933
  precision@5: 0.1840
  precision@10: 0.0960

CATEGORY ANALYSIS
------------------------------------------------------------
MENTAL_HEALTH_QUERIES:
  Count: 20
  Average Similarity: 0.7234

EDGE_CASES:
  Count: 10
  Average Similarity: 0.2345
```

## 🎯 Usage Examples

### Basic Usage

```python
from faq_bot import FaqBot

# Initialize and load the bot
bot = FaqBot()
bot.load()

# Get an answer
response = bot.get_answer("I've been feeling anxious lately")
print(response["answer"])
print(f"Confidence: {response['confidence']}")
print(f"Similarity: {response['similarity']:.3f}")
```

### Get Top-K Results

```python
# Get top 5 most similar answers
results = bot.get_top_k_answers("I can't sleep at night", k=5)
for i, result in enumerate(results, 1):
    print(f"{i}. FAQ {result['faq_id']}: {result['similarity']:.3f}")
```

### Interactive Testing

```bash
python evaluate.py --interactive
```

Commands:
- `quit` - Exit testing mode
- `help` - Show available commands
- `top5 <query>` - Show top 5 results for a query

## 🔍 Understanding the Results

### Response Format

The bot returns a dictionary with:

```python
{
    "answer": "The actual answer text",
    "similarity": 0.8234,  # Similarity score (0-1)
    "confidence": "high",  # low/medium/high
    "faq_id": 123,        # ID of the matched FAQ
    "top_matches": [      # List of top similar questions
        {
            "question": "Original FAQ question",
            "answer": "FAQ answer",
            "similarity": 0.8234,
            "faq_id": 123
        }
    ]
}
```

### Confidence Levels

- **High**: Similarity > 0.8
- **Medium**: Similarity > 0.6
- **Low**: Similarity ≤ 0.6

## 🧪 Testing and Evaluation

### Creating Test Sets

```bash
# Create a comprehensive test set
python evaluate.py --create-test-set
```

### Running Evaluations

```bash
# Run full evaluation with custom k-values
python evaluate.py --k-values 1 3 5 10 20
```

### Comparing Models

```bash
# Train with different models and compare
python train.py data.csv --model-name sentence-transformers/all-MiniLM-L6-v2 --evaluate
python train.py data.csv --model-name sentence-transformers/all-mpnet-base-v2 --evaluate
```

## 🚀 Performance Improvements

### Over Previous Implementation

1. **Better Semantic Understanding**: Sentence Transformers capture context better than Word2Vec
2. **No Manual Preprocessing**: Eliminates NLTK tokenization, lemmatization, and stopword removal
3. **Pre-trained Models**: Leverages models trained on large-scale datasets
4. **Consistent Embeddings**: Fixed-size embeddings regardless of input length
5. **Multilingual Support**: Can use multilingual models if needed

### Model Comparison

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| all-MiniLM-L6-v2 | Fast | Good | Small | Production |
| all-mpnet-base-v2 | Medium | Better | Medium | High Quality |
| msmarco-distilbert-base-v4 | Fast | Good | Small | QA Focused |

## 🔧 Troubleshooting

### Common Issues

1. **PyTorch Import Errors**: If you see `ImportError: cannot import name 'Tensor' from 'torch'`, reinstall PyTorch:
   ```bash
   pip uninstall torch -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Missing Dependencies**: Install required packages:
   ```bash
   pip install sentence-transformers transformers
   ```

3. **Large Dataset Processing**: For very large datasets, the training may take several minutes. Be patient and check CPU usage.

4. **Memory Issues**: If you encounter memory errors, try using a smaller model or processing the dataset in chunks.

### Debugging Steps

1. **Test PyTorch Installation**:
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

2. **Test Sentence Transformers**:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; print('Import successful!')"
   ```

3. **Check File Structure**:
   ```bash
   ls model_assets/
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Sentence Transformers library by UKP Lab
- Hugging Face for pre-trained models
- Streamlit for the web interface
- The mental health community for valuable insights

---

**Developed with <3 for mental health support** 