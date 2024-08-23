
# **Financial Sentiment Analysis using DistilBERT**

## **Project Overview**

This project applies Natural Language Processing (NLP) techniques to perform sentiment analysis on financial news and tweets. Using Hugging Face's `DistilBERT`, we classify the sentiment of text data into Negative, Neutral, and Positive categories. The project involves model fine-tuning, cross-validation, regularization techniques, and performance visualization.

## **Project Structure**

```plaintext
.
├── data/               # Datasets (Financial News, Twitter Financial Sentiment)
├── results/            # Outputs (logs, models, visualizations)
├── src/                # Source code (preprocessing, training, evaluation)
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```

## **Getting Started**

### **Prerequisites**

Ensure you have Python 3.8+ and create a virtual environment. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### **Dataset**

The project uses two datasets:
1. **Financial News Dataset**
2. **Twitter Financial Sentiment Dataset**

These datasets are included in the `data/` directory.

## **Workflow**

1. **Data Preprocessing**: Tokenization and text cleaning.
2. **Model Training**: Fine-tune `DistilBERT` using cross-validation and regularization (dropout, weight decay).
3. **Evaluation**: Assess the model's performance using accuracy, F1-score, ROC curves, and confusion matrices.

## **Model Training**

- We use `DistilBERT` for sentiment classification, with cross-validation to ensure robustness.
- Regularization techniques (dropout, weight decay) and early stopping help prevent overfitting.
- The model achieves an average accuracy of approximately **85.14%** across multiple folds.

## **Evaluation & Visualizations**

- **Accuracy & Loss**: We track accuracy and loss across cross-validation folds.
- **Confusion Matrix**: Visualizes the model’s prediction errors.
- **ROC Curves**: Show the performance of the model in distinguishing between sentiment classes.

## **Usage**

### **Train the Model**
Run the training script to fine-tune the model using cross-validation and regularization.

### **Evaluate the Model**
Run the evaluation script to assess the model’s performance and generate visualizations.

## **Results**

- The model demonstrates strong generalization with an average accuracy of **85.14%**.
- ROC Curves and Confusion Matrices help visualize performance across sentiment classes.

## **Potential Improvements**

- **Data Augmentation**: Incorporate more diverse data to improve model performance.
- **Hyperparameter Tuning**: Further refine model parameters to boost performance.
- **Ensemble Methods**: Leverage multiple models to improve sentiment classification.

## **Contributing**

Contributions are welcome! Feel free to submit issues or pull requests.

## **License**

This project is licensed under the MIT License.


## **Acknowledgements**

- Hugging Face for their `transformers` library.
- The contributors of `datasets` and the open datasets for providing financial sentiment data.
- Matplotlib and scikit-learn for providing visualization and evaluation tools.

---

This version provides an overview of the project while staying concise and focused, ideal for showcasing on GitHub.
