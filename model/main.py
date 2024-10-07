import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import torch

# Load the datasets
financial_news_df = pd.read_csv("./FinancialNews.csv", encoding='ISO-8859-1')
financial_tweets_df = pd.read_csv("./TwitterFinancial.csv", encoding='ISO-8859-1')

# Inspect the column names and data
print(financial_news_df.columns)
print(financial_tweets_df.columns)

# Rename columns for consistency
financial_news_df.rename(columns={'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .': 'Text', 'neutral': 'label'}, inplace=True)
financial_tweets_df.rename(columns={'Sentence': 'Text', 'Sentiment': 'label'}, inplace=True)

# Combine datasets
news_dataset = Dataset.from_pandas(financial_news_df[['Text', 'label']])
tweets_dataset = Dataset.from_pandas(financial_tweets_df[['Text', 'label']])

# Convert sentiment labels to integers (if not already integers)
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
news_dataset = news_dataset.map(lambda x: {'label': label_mapping[x['label']]})
tweets_dataset = tweets_dataset.map(lambda x: {'label': label_mapping[x['label']]})

# Split datasets into training and testing sets
news_dataset = news_dataset.train_test_split(test_size=0.2)
tweets_dataset = tweets_dataset.train_test_split(test_size=0.2)

# Load the tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the preprocessing function for tokenization
def preprocess_function(examples):
    return tokenizer(examples['Text'], truncation=True, padding='max_length', max_length=128)

# Apply tokenization to the datasets
news_dataset = news_dataset.map(preprocess_function, batched=True)
tweets_dataset = tweets_dataset.map(preprocess_function, batched=True)

# Remove the 'Text' column since we no longer need it after tokenization
news_dataset = news_dataset.remove_columns(['Text'])
tweets_dataset = tweets_dataset.remove_columns(['Text'])

# Load the pre-trained model with a randomly initialized classification head
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Load evaluation metrics
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    # Unpack the evaluation prediction (logits and labels)
    logits, labels = eval_pred
    # Convert logits to a PyTorch tensor if they are not already
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    # Get the predictions by taking the argmax of logits along the last dimension
    predictions = torch.argmax(logits, dim=-1)
    # Compute the accuracy using the metric
    return metric.compute(predictions=predictions, references=labels)

# Set up the Trainer for the news dataset
trainer_news = Trainer(
    model=model,
    args=training_args,
    train_dataset=news_dataset['train'],
    eval_dataset=news_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Set up the Trainer for the tweets dataset
trainer_tweets = Trainer(
    model=model,
    args=training_args,
    train_dataset=tweets_dataset['train'],
    eval_dataset=tweets_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and evaluate the financial news model
trainer_news.train()
trainer_news.evaluate()

# Train and evaluate the financial tweets model
trainer_tweets.train()
trainer_tweets.evaluate()