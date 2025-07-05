# Sentiment Analysis on Product Reviews

This repository contains a comprehensive workflow for sentiment analysis on product reviews, including data loading, preprocessing, feature engineering, model training (classical ML, LSTM, BERT), and evaluation.

## Project Structure

```
.
├── Sentiment_Analysis_Complete.ipynb
├── bert_features.py
├── evaluate_bert.py
├── feature_engineering.py
├── generate_pie_chart.py
├── load_data.py
├── preprocess_data.py
├── train_bert.py
├── train_logistic_regression.py
├── train_lstm.py
├── train_naive_bayes.py
├── train_random_forest.py
├── train_svm.py
├── train_xgboost.py
├── word2vec_features.py
├── data/
│   ├── preprocessed_reviews.csv
│   └── Reviews.csv
├── models/
│   ├── bert_cm.png
│   ├── bert_history.png
│   ├── bert_model_intermediate_batch100.pt
│   └── ...
└── processed_data/
    └── ...
```

## Features

- **Data Preprocessing:** Cleans and prepares raw review data.
- **Feature Engineering:** Generates features using TF-IDF, Word2Vec, and BERT embeddings.
- **Model Training:** Supports Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost, LSTM, and BERT fine-tuning.
- **Evaluation:** Outputs accuracy, classification reports, and confusion matrices for all models.
- **Visualization:** Saves and displays training histories and confusion matrices.

## Getting Started

### 1. Install Dependencies

Install the required Python packages:

```sh
pip install -r requirements.txt
```

Typical requirements include:
- pandas, numpy, scikit-learn, matplotlib, seaborn, gensim, torch, transformers, xgboost, joblib, tqdm, pillow

### 2. Prepare Data

Place your raw reviews CSV file in the `data/` directory as `Reviews.csv`.

### 3. Run the Notebook

Open and run all cells in [Sentiment_Analysis_Complete.ipynb](Sentiment_Analysis_Complete.ipynb) for the full workflow.

### 4. Results

- Model outputs, reports, and plots are saved in the `models/` directory.
- Preprocessed data and features are saved in `data/` and `processed_data/`.

## Scripts

You can also run individual scripts for specific tasks:
- Data preprocessing: [`preprocess_data.py`](preprocess_data.py)
- Feature engineering: [`feature_engineering.py`](feature_engineering.py), [`word2vec_features.py`](word2vec_features.py), [`bert_features.py`](bert_features.py)
- Model training: [`train_logistic_regression.py`](train_logistic_regression.py), [`train_naive_bayes.py`](train_naive_bayes.py), [`train_random_forest.py`](train_random_forest.py), [`train_svm.py`](train_svm.py), [`train_xgboost.py`](train_xgboost.py), [`train_lstm.py`](train_lstm.py), [`train_bert.py`](train_bert.py)

## Results and Comparison

The notebook summarizes and compares the performance of all models, including accuracy, classification reports, and confusion matrices.

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

---

*This project was developed as part of a major project at IIT Delhi.*
