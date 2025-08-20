import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data(dataset_path, seed_size=50):
    """
    Load and split the dataset into seed (labeled), unlabeled, and validation sets.

    The data is first deduplicated. Then, it is split into a small seed set (for initial training)
    and a larger temporary pool. The temporary pool is further split into an unlabeled set (for
    active learning querying) and a validation set (for evaluation).

    Args:
        dataset_path (str): Path to the CSV file containing 'text' and 'label' columns.
        seed_size (int, optional): Number of samples to use in the initial labeled seed set.
                                   Must be less than total dataset size. Defaults to 50.

    Returns:
        tuple: A tuple containing:
            - seed_df (pd.DataFrame): Initial labeled training set.
            - unlabeled_df (pd.DataFrame): Pool of unlabeled data for querying.
            - val_df (pd.DataFrame): Held-out set for model evaluation.
    """
    df = pd.read_csv(dataset_path).drop_duplicates(subset=["text", "label"])
    train, temp = train_test_split(df, train_size=seed_size, stratify=df["label"], random_state=42)
    val, unlabeled = train_test_split(temp, test_size=0.8, stratify=temp["label"], random_state=42)
    return train.copy(), unlabeled.reset_index(drop=True), val.reset_index(drop=True)


def get_label_mappings(dfs):
    """
    Create label-to-ID and ID-to-label mappings from a list of DataFrames.

    Combines multiple DataFrames to extract all unique labels and creates consistent
    mappings for model training. Labels are sorted alphabetically for reproducibility.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames containing a 'label' column.

    Returns:
        tuple: A tuple containing:
            - label2id (dict): Mapping from label string to integer ID.
            - id2label (dict): Mapping from integer ID to label string.
    """
    combined = pd.concat(dfs).reset_index(drop=True)
    labels = sorted(combined["label"].unique())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    return label2id, id2label


def tokenize_and_encode(dataset, tokenizer, max_length=128):
    """
    Tokenize text and ensure labels are integers in a Hugging Face Dataset.

    Applies tokenization (truncation and padding) and converts string labels to integers
    using the provided tokenizer. The output dataset is formatted for PyTorch.

    Args:
        dataset (DatasetDict or Dataset): Input dataset with 'text' and 'label' columns.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer (e.g., DistilBertTokenizer).
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 128.

    Returns:
        Dataset: Tokenized and encoded dataset with PyTorch formatting.
    """
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", 
                         max_length=max_length)

    def to_int(examples):
        return {"label": [int(l) for l in examples["label"]]}

    ds = dataset.map(tokenize, batched=True)
    ds = ds.map(to_int, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def get_uncertainty(model, tokenizer, texts, device):
    """
    Compute uncertainty (entropy) of model predictions for a list of texts.

    Uses entropy of predicted class probabilities as a measure of uncertainty.
    Higher entropy indicates higher uncertainty, which is used in uncertainty sampling.

    Args:
        model (PreTrainedModel): Fine-tuned classification model (e.g., DistilBertForSequenceClassification).
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
        texts (list of str): List of input texts to evaluate.
        device (torch.device): Device (CPU, CUDA, or MPS) to run inference on.

    Returns:
        np.ndarray: Array of entropy scores (uncertainty) for each input text.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy.cpu().numpy()


def query_uncertain(model, tokenizer, df, batch_size, device):
    """
    Select the most uncertain samples from the unlabeled pool using entropy.

    Computes prediction entropy for all texts in the DataFrame and returns the top-K
    most uncertain samples. Handles out-of-memory errors by recursively reducing batch size.

    Args:
        model (PreTrainedModel): Trained model for uncertainty estimation.
        tokenizer (PreTrainedTokenizer): Tokenizer for preprocessing text.
        df (pd.DataFrame): Unlabeled data pool with 'text' and 'label' columns.
        batch_size (int): Number of samples to query.
        device (torch.device): Device to run inference on.

    Returns:
        pd.DataFrame: Subset of `df` containing the most uncertain `batch_size` samples.
    """
    try:
        scores = get_uncertainty(model, tokenizer, df["text"].tolist(), device)
        idx = np.argsort(scores)[-batch_size:]
        return df.iloc[idx].copy()
    except RuntimeError:
        print("OOM during querying. Reducing batch size.")
        return query_uncertain(model, tokenizer, df, max(1, batch_size // 2), device)


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics (accuracy, F1, precision, recall) for classification.

    Metrics are computed using the `evaluate` library. F1, precision, and recall use
    macro averaging to handle multi-class imbalance.

    Args:
        eval_pred (tuple): Tuple containing:
            - predictions (np.ndarray): Model output logits.
            - label_ids (np.ndarray): True labels.

    Returns:
        dict: Dictionary with keys 'accuracy', 'f1', 'precision', and 'recall'.
    """
    preds = np.argmax(eval_pred.predictions, axis=1)
    refs = eval_pred.label_ids
    return {
        "accuracy": evaluate.load("accuracy").compute(predictions=preds, references=refs)["accuracy"],
        "f1": evaluate.load("f1").compute(predictions=preds, references=refs, average="macro")["f1"],
        "precision": evaluate.load("precision").compute(predictions=preds, references=refs, average="macro")["precision"],
        "recall": evaluate.load("recall").compute(predictions=preds, references=refs, average="macro")["recall"],
    }


def active_learning(
    model, tokenizer, seed_df, unlabeled_df, val_df, device, 
    iterations=5, query_size=20
):
    """
    Run active learning loop using uncertainty sampling and return learning curve data.

    At each iteration:
    1. Trains the model on the growing labeled pool.
    2. Evaluates on the validation set.
    3. Queries the most uncertain samples from the unlabeled pool.
    4. Updates the labeled pool with newly queried samples.

    Tracks training size and F1 score over iterations for analysis.

    Args:
        model (PreTrainedModel): Base model to fine-tune (e.g., DistilBertForSequenceClassification).
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        seed_df (pd.DataFrame): Initial labeled seed set.
        unlabeled_df (pd.DataFrame): Pool of unlabeled data for querying.
        val_df (pd.DataFrame): Validation set for evaluation.
        device (torch.device): Device to run training and inference.
        iterations (int, optional): Number of active learning iterations. Defaults to 5.
        query_size (int, optional): Number of samples to query per iteration. Defaults to 20.

    Returns:
        tuple: A tuple containing:
            - model (PreTrainedModel): Final trained model.
            - tokenizer (PreTrainedTokenizer): Tokenizer (unchanged).
            - training_sizes (list of int): Number of labeled samples at each iteration.
            - f1_scores (list of float): Validation F1 score at each iteration.
    """
    training_sizes = []
    f1_scores = []
    labeled_pool = seed_df.copy()

    # Training setup
    args = TrainingArguments(
        output_dir="./tmp",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2 if device.type == "mps" else 1,
        report_to="none"
    )

    for i in range(iterations):
        print(f"\n--- AL Iteration {i+1} ---")
        if device.type == "mps":
            torch.mps.empty_cache()

        # Update dataset
        dataset = DatasetDict({
            "train": Dataset.from_pandas(labeled_pool),
            "validation": Dataset.from_pandas(val_df)
        })
        label2id, id2label = get_label_mappings([labeled_pool, val_df])
        encoded = dataset.map(lambda x: {"label": label2id[x["label"]]})
        tokenized = tokenize_and_encode(encoded, tokenizer)

        # Train
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        trainer.train()

        # Evaluate
        results = trainer.evaluate()
        f1 = results["eval_f1"]
        size = len(labeled_pool)
        training_sizes.append(size)
        f1_scores.append(f1)
        print(f"Training size: {size}, F1: {f1:.3f}")

        # Query
        if len(unlabeled_df) >= query_size:
            new_samples = query_uncertain(model, tokenizer, unlabeled_df, query_size, device)
            labeled_pool = pd.concat([labeled_pool, new_samples], ignore_index=True)
            unlabeled_df = unlabeled_df.drop(new_samples.index).reset_index(drop=True)

    return model, tokenizer, training_sizes, f1_scores


def plot_learning_curve(sizes, f1_scores):
    """
    Plot the learning curve showing F1 score vs. number of labeled samples.

    Visualizes the performance improvement of the model as more data is labeled
    through active learning. Saves the plot to 'learning_curve.png'.

    Args:
        sizes (list of int): Training set sizes at each iteration.
        f1_scores (list of float): Corresponding F1 scores.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, f1_scores, marker='o', label="Active Learning")
    plt.title("Learning Curve: F1 Score vs. Training Size")
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Macro F1 Score")
    plt.grid(True, alpha=0.3)
    plt.xticks(sizes)
    plt.tight_layout()
    plt.savefig("learning_curve.png")


def main():
    """
    Main execution function.

    Orchestrates the entire active learning pipeline:
    1. Loads and splits data.
    2. Initializes the model and tokenizer.
    3. Runs the active learning loop.
    4. Saves the final model.
    5. Plots the learning curve.

    Raises:
        ValueError: If DATASET_PATH is not set or file does not exist.
    """
    dataset_path = os.getenv("DATASET_PATH")
    if not dataset_path or not os.path.exists(dataset_path):
        raise ValueError("DATASET_PATH is not set or file not found.")

    print("Loading data...")
    seed, unlabeled, val = load_data(dataset_path)

    print("Setting up model...")
    label2id, id2label = get_label_mappings([seed, unlabeled, val])
    num_labels = len(label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels, id2label=id2label, label2id=label2id
    ).to(device)

    if device.type == "mps":
        model = model.to(torch.float32)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("Starting active learning...")
    model, tokenizer, sizes, f1s = active_learning(
        model, tokenizer, seed, unlabeled, val, device,
        iterations=20, query_size=20
    )

    # Save final model
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    print("Model saved to ./model")

    # Plot learning curve
    plot_learning_curve(sizes, f1s)
    print("Learning curve saved as 'learning_curve.png'")


if __name__ == "__main__":
    main()