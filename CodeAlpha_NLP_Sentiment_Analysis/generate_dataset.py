import pandas as pd
import random

positive = [
    "I love this product",
    "Amazing quality and service",
    "Very happy with purchase",
    "Excellent experience",
    "Totally satisfied",
    "Best thing I bought",
    "Highly recommended",
]

negative = [
    "Worst product ever",
    "Very disappointed",
    "Waste of money",
    "Bad quality",
    "I hate it",
    "Not worth buying",
    "Terrible experience",
]

neutral = [
    "It is okay",
    "Average product",
    "Nothing special",
    "It works fine",
    "Normal experience",
    "Not good not bad",
]

data = []

# Generate 70+70+70 = 210 rows
for _ in range(70):
    data.append([random.choice(positive), "positive"])
    data.append([random.choice(negative), "negative"])
    data.append([random.choice(neutral), "neutral"])

df = pd.DataFrame(data, columns=["text", "sentiment"])
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("data_large.csv", index=False)

print("Dataset created with", len(df), "rows")