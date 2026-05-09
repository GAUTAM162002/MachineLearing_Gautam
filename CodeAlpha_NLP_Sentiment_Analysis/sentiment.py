import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import joblib

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv(r"data_large.csv")

# NLP setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply cleaning
data["clean_text"] = data["text"].apply(clean_text)

# -----------------------------
# Sentiment Distribution Graph
# -----------------------------

data["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.show()

# -----------------------------
# WordCloud
# -----------------------------

positive_text = " ".join(
    data[data["sentiment"] == "positive"]["clean_text"]
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(10,5))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()

# -----------------------------
# Features & Labels
# -----------------------------

X = data["clean_text"]

y = data["sentiment"]

# TF-IDF
tfidf = TfidfVectorizer(max_features=300)

X_vectorized = tfidf.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model.pkl")

joblib.dump(tfidf, "vectorizer.pkl")

print("Model and vectorizer saved!")

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')

plt.xlabel('predicted')
plt.ylabel('Actual')

plt.show()