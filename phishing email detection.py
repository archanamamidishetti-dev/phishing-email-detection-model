import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("emails.csv")


suspicious_keywords = [
    "verify",
    "urgent",
    "click",
    "password",
    "bank",
    "account",
    "login",
    "update",
    "confirm",
    "security"
]

def extract_features(text):

    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))

    keyword_count = sum(
        word.lower() in text.lower()
        for word in suspicious_keywords
    )

    ip_present = int(
        bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))
    )

    return pd.Series([url_count, keyword_count, ip_present])

df[['url_count', 'keyword_count', 'ip_present']] = (
    df['text'].apply(extract_features)
)



X_train, X_test, y_train, y_test = train_test_split(
    df[['text', 'url_count', 'keyword_count', 'ip_present']],
    df['label'],
    test_size=0.2,
    random_state=42
)


preprocessor = ColumnTransformer(
    transformers=[
        ('text',
         TfidfVectorizer(stop_words='english'),
         'text'),

        ('numeric',
         StandardScaler(),
         ['url_count', 'keyword_count', 'ip_present'])
    ]
)


model = Pipeline([
    ('features', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])



model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Phishing','Safe'],
    yticklabels=['Phishing','Safe']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()