import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_and_prepare_data(dataset_path):
    """Loads data, removes duplicates, and splits into train/val sets."""
    df = pd.read_csv(dataset_path).drop_duplicates(subset=["text", "label"])

    # Split into train and validation sets (80-20 split)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Check for overlap (optional print)
    overlap = pd.merge(train_df, val_df, on=["text", "label"])
    print("Overlap rows between train and val:", len(overlap))

    return train_df, val_df

def create_datasets(train_df, val_df):
    """Converts pandas DataFrames to Hugging Face Datasets."""
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
    return dataset

def encode_labels(dataset, df):
    """Creates label mappings and encodes labels in the dataset."""
    unique_labels = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    def _encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    encoded_dataset = dataset.map(_encode_labels)
    return encoded_dataset, label2id, id2label

def tokenize_dataset(dataset, tokenizer_name="distilbert-base-uncased", max_length=128):
    """Tokenizes the text data in the dataset."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

    def _tokenize_function(example):
        """Function to tokenize a piece of text"""
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # return the data in the form of a PyTorch dataset
    tokenized_dataset = dataset.map(_tokenize_function, batched=True)
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return tokenized_dataset, tokenizer

def load_model(model_name, num_labels, id2label, label2id):
    """Loads the pre-trained model for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model

def setup_training_args(output_dir="./distilbert-rps-finetuned"):
    """Defines the training arguments."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch", # Note: 'evaluation_strategy' is preferred in newer versions
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # Add report_to=[] if you want to disable logging to external tools like W&B
        # report_to=[]
    )
    return training_args

def compute_metrics(eval_pred):
    """Function to compute post-training metrics."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"],
    }

def train_and_evaluate(
    model,
    training_args,
    tokenized_dataset,
    tokenizer,
    compute_metrics_fn
):
    """Creates the trainer, trains the model, and evaluates it."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics_fn, # Pass the function
        tokenizer=tokenizer,
    )

    # Train and Evaluate
    trainer.train()
    results = trainer.evaluate()
    return trainer, results

def save_model_and_tokenizer(trainer, tokenizer, save_path="./distilbert-rps-finetuned"):
    """Saves the trained model and tokenizer."""
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

def main():
    """Main execution function."""
    try:
        dataset_path_env = os.getenv("DATASET_PATH")
        if not dataset_path_env:
            raise ValueError("DATASET_PATH environment variable not set.")
        dataset_path = os.path.join(dataset_path_env)

        # 1. Load and Prepare Data
        train_df, val_df = load_and_prepare_data(dataset_path)

        # 2. Create Datasets
        dataset = create_datasets(train_df, val_df)

        # 3. Encode Labels
        dataset, label2id, id2label = encode_labels(dataset, pd.concat([train_df, val_df])) # Pass full df for unique labels
        num_labels = len(label2id)

        # 4. Tokenize Dataset
        tokenized_dataset, tokenizer = tokenize_dataset(dataset)

        # 5. Load Model
        model = load_model(
            "distilbert-base-uncased", num_labels, id2label, label2id
        )

        # 6. Setup Training Arguments
        training_args = setup_training_args()

        # 7. Train and Evaluate
        trainer, results = train_and_evaluate(
            model, training_args, tokenized_dataset, tokenizer, compute_metrics
        )
        print("Evaluation Results:", results)

        # 8. Save Model and Tokenizer
        save_model_and_tokenizer(trainer, tokenizer)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        # Optionally re-raise the exception if you want the script to fail
        # raise

if __name__ == "__main__":
    main()