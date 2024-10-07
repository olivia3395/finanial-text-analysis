from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DistilBertForSequenceClassification
from sklearn.model_selection import KFold
import numpy as np
import torch

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_accuracies = []
cross_val_losses = []
all_preds = []
all_labels = []

# Convert the training dataset to a pandas DataFrame
news_dataset_train_df = news_dataset['train'].to_pandas()

for fold, (train_index, test_index) in enumerate(kf.split(news_dataset_train_df)):
    print(f"Training fold {fold + 1}/{kf.n_splits}")
    
    # Select the rows for train and test using the indices from KFold
    train_split = news_dataset['train'].select(train_index.tolist())
    test_split = news_dataset['train'].select(test_index.tolist())
    
    # Reinitialize model for each fold
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, dropout=dropout_rate)
    model.to(torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu"))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True
    )
    
    # Implement early stopping
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=test_split,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate()

    # Store cross-validation results
    cross_val_accuracies.append(eval_result['eval_accuracy'])
    cross_val_losses.append(eval_result['eval_loss'])
    
    # Store predictions and labels for ROC and Confusion Matrix
    predictions = trainer.predict(test_split)
    preds = np.argmax(predictions.predictions, axis=-1)
    all_preds.extend(preds)
    all_labels.extend(predictions.label_ids)

# Calculate and visualize results
average_accuracy = np.mean(cross_val_accuracies)
average_loss = np.mean(cross_val_losses)