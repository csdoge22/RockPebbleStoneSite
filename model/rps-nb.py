import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

# read the dataset
dataset = pd.read_csv("model/rps_dataset.csv").dropna()

if not os.path.exists("model/rps-nb"):
    model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), max_features=1000), MultinomialNB())

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataset["text"], dataset["label"], random_state=42, test_size=0.3, stratify=dataset["label"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f1_score(y_test,model.predict(X_test), average='micro'))
    print(accuracy_score(y_test,model.predict(X_test)))
    
    with open("model/rps-nb", "wb") as file:
        pickle.dump(model, file)
else:
    with open("model/rps-nb", "rb") as file:
        model = pickle.load(file)
    

