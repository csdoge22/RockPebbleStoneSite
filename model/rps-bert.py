from transformers import pipeline
import pandas as pd

model1 = pipeline("text-classification", model="distilbert-base-uncased")

dataset = pd.read_csv("./rock_pebble_sand_dataset.csv")
print(model1("The "))
print(dataset)