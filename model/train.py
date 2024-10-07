from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, load_metric
import torch
import numpy as np

# Load the pre-trained model with a randomly initialized classification head
model_name = "distilbert-base-uncased"
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