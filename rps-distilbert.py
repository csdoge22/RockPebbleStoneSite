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
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_and_split_data(dataset_path, seed_size=50, val_ratio=0.2, stop_set_size=2000):
    """
    Load and split data into seed, unlabeled, validation, and stop sets.

    Args:
        dataset_path (str): Path to the CSV file.
        seed_size (int): Size of the initial labeled seed set.
        val_ratio (float): Proportion of data for validation.
        stop_set_size (int): Size of the unlabeled stop set for stopping criterion.

    Returns:
        dict: Dictionary with 'seed', 'unlabeled', 'val', 'stop_set' DataFrames.
    """
    df = pd.read_csv(dataset_path).set_index(keys= "id")
    print(df.shape[0])
    df.to_csv("./updated_dataset.csv")
    
    # First split: seed vs. rest
    seed, rest = train_test_split(df, train_size=seed_size, stratify=df["label"], random_state=42)
    
    # Second split: val vs. temp_pool
    val, temp_pool = train_test_split(rest, test_size=val_ratio, stratify=rest["label"], random_state=42)
    
    # Third split: stop_set vs. unlabeled
    # Always use train_test_split to get the stop_set and the remainder as unlabeled
    # If stop_set_size is larger than temp_pool, train_test_split will return the entire temp_pool for stop_set
    stop_set, unlabeled = train_test_split(
        temp_pool, 
        train_size=stop_set_size, 
        stratify=temp_pool["label"], 
        random_state=42
    )
    
    return {
        'seed': seed.reset_index(drop=True),
        'unlabeled': unlabeled.reset_index(drop=True),
        'val': val.reset_index(drop=True),
        'stop_set': stop_set.reset_index(drop=True)
    }

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
    print(f"Label mappings: {label2id}")
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    return label2id, id2label


def setup_model_and_tokenizer(data_dfs, device):
    """
    Initialize the tokenizer and model with label mappings.

    Args:
        data_dfs (list of pd.DataFrame): All data splits for label mapping.
        device (torch.device): Device to place the model on.

    Returns:
        tuple: (model, tokenizer, label2id, id2label)
    """
    label2id, id2label = get_label_mappings(data_dfs)
    num_labels = len(label2id)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels, id2label=id2label, label2id=label2id
    ).to(device)

    if device.type == "mps":
        model = model.to(torch.float32)

    return model, tokenizer, label2id, id2label


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

def get_predictions_on_stop_set(model, tokenizer, stop_texts, device):
    """Get model predictions on the stop set."""
    inputs = tokenizer(stop_texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    return preds

def run_active_learning(model, tokenizer, data, device, query_size=20,
                        kappa_threshold=0.99, window_size=3, max_iterations=20):
    """
    Run the active learning loop with Stabilizing Predictions stopping.

    Args:
        model, tokenizer, device: As before.
        data (dict): Output from load_and_split_data().
        query_size (int): Number of samples to query per iteration.
        kappa_threshold (float): Kappa agreement threshold for stopping.
        window_size (int): Number of recent model pairs to check.
        max_iterations (int): Hard cap on iterations.

    Returns:
        dict: Results including final model, sizes, f1s, and stop reason.
    """

    labeled_pool = data['seed'].copy()
    unlabeled_pool = data['unlabeled'].copy()
    stop_texts = data['stop_set']['text'].tolist()
    
    training_sizes = []
    f1_scores = []
    previous_preds = []  # Stores predictions from recent models
    
    args = TrainingArguments(
        output_dir="./tmp", eval_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=8,
        per_device_eval_batch_size=8, num_train_epochs=3,
        weight_decay=0.01, load_best_model_at_end=True,
        metric_for_best_model="f1", fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2 if device.type == "mps" else 1,
        report_to="none"
    )
    
    for i in range(max_iterations):
        print(f"\n--- AL Iteration {i+1} ---")
        if device.type == "mps":
            torch.mps.empty_cache()

        # Prepare Dataset
        dataset = DatasetDict({
            "train": Dataset.from_pandas(labeled_pool),
            "validation": Dataset.from_pandas(data['val'])
        })
        label2id, _ = get_label_mappings([labeled_pool, data['val']])
        encoded = dataset.map(lambda x: {"label": label2id[x["label"]]})
        tokenized = tokenize_and_encode(encoded, tokenizer)

        # Train
        trainer = Trainer(
            model=model, args=args, train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics, tokenizer=tokenizer
        )
        trainer.train()

        # Evaluate
        results = trainer.evaluate()
        f1 = results["eval_f1"]
        size = len(labeled_pool)
        training_sizes.append(size)
        f1_scores.append(f1)
        print(f"Training size: {size}, F1: {f1:.3f}")

        # --- Stabilizing Predictions Check ---
        current_preds = get_predictions_on_stop_set(model, tokenizer, stop_texts, device)
        
        if len(previous_preds) > 0:
            kappa = cohen_kappa_score(current_preds, previous_preds[-1])
            print(f"Kappa agreement with previous model: {kappa:.4f}")

            # Store current predictions
            previous_preds.append(current_preds)

            # Keep only last (window_size + 1) predictions
            if len(previous_preds) > window_size + 1:
                previous_preds.pop(0)

            # Only check for stopping if we have enough model comparisons
            if len(previous_preds) >= window_size + 1:
                # Compute Kappa for the last `window_size` consecutive pairs
                recent_kappas = [
                    cohen_kappa_score(previous_preds[-j], previous_preds[-j-1])
                    for j in range(1, window_size + 1)
                ]
                print(f"Recent Kappas: {recent_kappas}")

                if all(k >= kappa_threshold for k in recent_kappas):
                    print(f"Stopping at iteration {i+1}: predictions stabilized (Kappa ≥ {kappa_threshold} for {window_size} consecutive steps)")
                    break
        else:
            previous_preds.append(current_preds)

        # --- Query New Samples ---
        if len(unlabeled_pool) >= query_size:
            new_samples = query_uncertain(model, tokenizer, unlabeled_pool, query_size, device)
            labeled_pool = pd.concat([labeled_pool, new_samples], ignore_index=True)
            unlabeled_pool = unlabeled_pool.drop(new_samples.index).reset_index(drop=True)
        else:
            print("No more samples to query. Ending early.")
            break
    
    stop_reason = 'stabilized' if i + 1 < max_iterations else 'max_iterations'
    if stop_reason != 'stabilized' and len(unlabeled_pool) == 0:
        stop_reason = 'no_more_data'

    return {
        'model': model,
        'tokenizer': tokenizer,
        'training_sizes': training_sizes,
        'f1_scores': f1_scores,
        'final_iteration': i + 1,
        'stop_reason': stop_reason
    }


    
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

def save_and_report(results, data, output_dir="./model"):
    """
    Save the final model and generate the learning curve.

    Args:
        results (dict): Output from run_active_learning().
        data (dict): Original data splits.
        output_dir (str): Directory to save model.
    """
    # Move the model to CPU before saving
    model = results['model'].cpu()
    tokenizer = results['tokenizer']
    
    # Save model (DO THIS ONLY ONCE)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Plot learning curve
    plot_learning_curve(results['training_sizes'], results['f1_scores'])
    print("Learning curve saved as 'learning_curve.png'")

    # Print summary
    print(f"\n Active Learning Complete!")
    print(f"   Final Training Size: {results['training_sizes'][-1]}")
    print(f"   Final F1 Score: {results['f1_scores'][-1]:.3f}")
    print(f"   Stopping Reason: {results['stop_reason'].replace('_', ' ').title()}")

def main():
    dataset_path = os.getenv("DATASET_PATH")
    if not dataset_path or not os.path.exists(dataset_path):
        raise ValueError("DATASET_PATH is not set or file not found.")

    print("Loading and splitting data...")
    data = load_and_split_data(dataset_path, seed_size=50, stop_set_size=400)

    print("Setting up model...")
    model, tokenizer, _, _ = setup_model_and_tokenizer(
        [data['seed'], data['unlabeled'], data['val'], data['stop_set']], 
        device
    )

    print("Starting active learning with Stabilizing Predictions...")
    results = run_active_learning(
        model, tokenizer, data, device,
        query_size=20,
        kappa_threshold=0.99,
        window_size=3,
        max_iterations=20
    )

    print("Saving results...")
    save_and_report(results, data)

if __name__ == "__main__":
    main()