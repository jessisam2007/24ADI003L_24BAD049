import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("/Users/jessicasam/Downloads/spam.csv",encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df.dropna(inplace=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

df["clean_message"] = df["message"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["clean_message"])

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

model = MultinomialNB(alpha=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

misclassified = np.where(y_test != y_pred)[0]

print("\nMisclassified Examples:")
for i in misclassified[:5]:
    print("\nMessage:", df.loc[idx_test[i], "message"])
    print("Actual:", encoder.inverse_transform([y_test[i]])[0])
    print("Predicted:", encoder.inverse_transform([y_pred[i]])[0])

print("\nLaplace Smoothing Comparison:")
for a in [0.1, 1, 5]:
    temp_model = MultinomialNB(alpha=a)
    temp_model.fit(X_train, y_train)
    pred = temp_model.predict(X_test)
    print(f"Alpha={a} Accuracy:", accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Ham","Spam"],
            yticklabels=["Ham","Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_names = vectorizer.get_feature_names_out()
spam_prob = model.feature_log_prob_[1]
top_spam_words = np.argsort(spam_prob)[-15:]

plt.figure(figsize=(8,5))
plt.barh(feature_names[top_spam_words], spam_prob[top_spam_words])
plt.title("Top Words Indicating Spam")
plt.show()

spam_msgs = df[df["label"]=="spam"]["clean_message"]
ham_msgs = df[df["label"]=="ham"]["clean_message"]

spam_words = " ".join(spam_msgs).split()
ham_words = " ".join(ham_msgs).split()

spam_freq = pd.Series(spam_words).value_counts()[:10]
ham_freq = pd.Series(ham_words).value_counts()[:10]

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
spam_freq.plot(kind="bar")
plt.title("Top Spam Words")

plt.subplot(1,2,2)
ham_freq.plot(kind="bar")
plt.title("Top Ham Words")

plt.show()
