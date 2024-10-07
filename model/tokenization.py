from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

# Load the datasets
financial_news_df = pd.read_csv("./FinancialNews.csv", encoding='ISO-8859-1')
financial_tweets_df = pd.read_csv("./TwitterFinancial.csv", encoding='ISO-8859-1')

# Rename columns for consistency
financial_news_df.rename(columns={'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .': 'Text', 'neutral': 'label'}, inplace=True)
financial_tweets_df.rename(columns={'Sentence': 'Text', 'Sentiment': 'label'}, inplace=True)

# Combine datasets
news_dataset = Dataset.from_pandas(financial_news_df[['Text', 'label']])
tweets_dataset = Dataset.from_pandas(financial_tweets_df[['Text', 'label']])

# Convert sentiment labels to integers
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