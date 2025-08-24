import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
import logging
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables with validation"""
    load_dotenv()
    dataset_path = os.getenv("DATASET_PATH")
    model_path = os.getenv("NAIVE_BAYES_PATH")
    
    if not dataset_path or not model_path:
        raise EnvironmentError("Missing required environment variables")
    
    return Path(dataset_path), Path(model_path)

def load_and_validate_data(dataset_path: Path) -> pd.DataFrame:
    """Load and validate dataset with enhanced checks"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    try:
        dataset = pd.read_csv(dataset_path).dropna()
    except Exception as e:
        raise IOError(f"Error reading dataset: {str(e)}")
    
    # Enhanced column validation
    required_columns = {'text', 'label'}
    if not required_columns.issubset(dataset.columns):
        missing = required_columns - set(dataset.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    # Data quality checks
    check_data_quality(dataset)
    return dataset

def check_data_quality(df: pd.DataFrame):
    """Perform comprehensive data quality checks"""
    logger.info("\n=== DATA QUALITY REPORT ===")
    
    # Check for empty text
    empty_text = df[df['text'].str.strip().str.len() == 0]
    if not empty_text.empty:
        logger.warning(f"Found {len(empty_text)} empty text samples")
    
    # Label distribution
    logger.info("\nLabel Distribution:")
    label_counts = df['label'].value_counts()
    logger.info(label_counts)
    
    # Text length analysis
    df['text_length'] = df['text'].str.len()
    logger.info("\nText Length Statistics:")
    logger.info(df['text_length'].describe().to_string())
    
    # Visualize label distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='label')
    plt.title("Class Distribution")
    plt.show()

def create_train_val_splits(dataset: pd.DataFrame, train_size: int = 600):
    """Create balanced splits with dynamic sizing"""
    df_train_list = []
    df_val_list = []
    
    for label in dataset['label'].unique():
        df_label = dataset[dataset['label'] == label]
        available_samples = len(df_label)
        
        # Dynamic sizing based on available data
        actual_train_size = min(train_size, available_samples - 100)
        if actual_train_size < 100:
            logger.warning(f"Insufficient samples for '{label}': {available_samples} total")
            continue
            
        # Create splits
        df_train_label = df_label.sample(n=actual_train_size, random_state=42)
        df_val_label = df_label.drop(df_train_label.index)
        
        df_train_list.append(df_train_label)
        df_val_list.append(df_val_label)
    
    if not df_train_list:
        raise ValueError("Insufficient data for all classes after splitting")
    
    return (
        pd.concat(df_train_list).sample(frac=1, random_state=42),
        pd.concat(df_val_list).sample(frac=1, random_state=42)
    )

def visualize_embeddings(X, y):
    """Visualize text feature space"""
    tfidf = TfidfVectorizer(min_df=3, max_df=0.9).fit(X)
    X_tfidf = tfidf.transform(X)
    
    svd = TruncatedSVD(n_components=2)
    X_2d = svd.fit_transform(X_tfidf)
    
    plt.figure(figsize=(10, 6))
    for label in np.unique(y):
        plt.scatter(
            X_2d[y == label, 0], 
            X_2d[y == label, 1],
            label=label,
            alpha=0.6
        )
    plt.legend()
    plt.title("Text Feature Space Visualization (SVD Reduced)")
    plt.show()

def cross_validate_model(X, y, n_splits=5):
    """Perform stratified cross-validation with proper array conversion"""
    skf = StratifiedKFold(n_splits=n_splits)
    f1_scores = []
    
    # Convert to numpy arrays to avoid pandas indexing issues
    X_np = X.values if isinstance(X, pd.Series) else np.array(X)
    y_np = y.values if isinstance(y, pd.Series) else np.array(y)
    
    for train_idx, val_idx in skf.split(X_np, y_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]
        
        model = make_pipeline(
            TfidfVectorizer(lowercase=True, stop_words="english"),
            MultinomialNB()
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
    
    return f1_scores

def plot_confusion_matrix(y_true, y_pred):
    """Plot annotated confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def load_existing_model(model_path: Path):
    """Safely load an existing trained model"""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # Basic validation that we loaded a proper model
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid scikit-learn model")
            
        logger.info(f"Successfully loaded model from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def train_model(X_train, y_train, X_val, y_val, model_path: Path):
    """Enhanced model training with diagnostics"""
    logger.info("\n=== MODEL TRAINING ===")
    
    # Improved vectorizer configuration
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )
    
    model = make_pipeline(
        vectorizer,
        MultinomialNB(alpha=0.1)
    )
    
    # Cross-validation (now with proper array conversion)
    logger.info("Running cross-validation...")
    cv_scores = cross_validate_model(X_train, y_train)
    logger.info(f"CV F1 Scores: {cv_scores}")
    logger.info(f"Mean CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Final training
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_val)
    logger.info(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    logger.info(f"Validation F1 (weighted): {f1_score(y_val, y_pred, average='weighted'):.4f}")
    
    # Confusion matrix
    plot_confusion_matrix(y_val, y_pred)
    
    # Feature visualization
    visualize_embeddings(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path.absolute()}")

def analyze_errors(model, X_val, y_val):
    """Display misclassified examples"""
    y_pred = model.predict(X_val)
    errors = X_val[y_pred != y_val]
    
    logger.info("\n=== MISCLASSIFIED EXAMPLES ===")
    for true, pred, text in zip(y_val[y_pred != y_val], y_pred[y_pred != y_val], errors):
        logger.info(f"TRUE: {true} | PRED: {pred}\n{text}\n{'-'*50}")

def main():
    try:
        # 1. Load configuration
        dataset_path, model_path = load_environment()
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Model path: {model_path}")
        
        # 2. Load and validate data
        dataset = load_and_validate_data(dataset_path)
        logger.info(f"\nLoaded dataset with {len(dataset)} samples")
        
        # 3. Create splits (dynamic sizing based on data)
        df_train, df_val = create_train_val_splits(dataset)
        X_train, y_train = df_train["text"], df_train["label"]
        X_val, y_val = df_val["text"], df_val["label"]
        
        # 4. Train or load model
        if not model_path.exists():
            train_model(X_train, y_train, X_val, y_val, model_path)
            model = make_pipeline(
                TfidfVectorizer(lowercase=True, stop_words="english"),
                MultinomialNB()
            ).fit(X_train, y_train)
            analyze_errors(model, X_val, y_val)
        else:
            model = load_existing_model(model_path)
            
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
