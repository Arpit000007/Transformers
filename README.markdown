# Transformer Model for Machine Translation
This project implements a transformer model from scratch using PyTorch for English-French translation. The implementation adheres to academic standards, avoiding any prebuilt modules related to transformers.

# Key Features:
Self-Attention Mechanism: Utilizes self-attention to effectively capture long-range dependencies in the sequence data.
Positional Encodings: Incorporates advanced positional encodings (e.g., learnable and relative positional encodings) to enhance the model's ability to recognize token positions in sequences.
Hyperparameter Tuning: Conducted experiments with varying the number of encoder/decoder layers, attention heads, embedding dimensions, and dropout rates.
Performance Summary:
The model's performance was evaluated using BLEU scores for translation accuracy.

# Optimal Configuration:

Layers: 6

Attention Heads: 8

Embedding Dimensions: 512

Dropout: 0.2

BLEU Score: 11.66

Achieved the lowest training and validation losses (3.9336 and 4.3549).

Configurations with higher complexity (e.g., 8 layers, 0.4 dropout) showed reduced performance, indicating overfitting or inefficiency.

# Results and Analysis:

Training and Validation Loss: Demonstrated that increasing model complexity beyond the optimal configuration degrades performance.

BLEU Score Insights: The optimal configuration provided the best translation quality, outperforming other settings.

# Recommendations:

Focus on moderate model complexity (e.g., 6 layers, 0.2 dropout) for balanced performance.

Further experiments can explore different learning rates and regularization techniques to enhance translation quality.
