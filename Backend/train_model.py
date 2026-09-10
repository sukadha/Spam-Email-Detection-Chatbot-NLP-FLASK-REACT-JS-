import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

print("Original shape:", df.shape)


# ============================================================
# 2. KEEP ONLY REQUIRED COLUMNS
# ============================================================

df = df[["v1", "v2"]]

df.columns = [
    "label",
    "message"
]


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

df.dropna(
    subset=["label", "message"],
    inplace=True
)


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df.drop_duplicates(
    inplace=True
)

after_duplicates = len(df)

print(
    "Duplicates removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 5. CONVERT LABELS
# ============================================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ============================================================
# 6. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 7. CLEAN MESSAGES
# ============================================================

print("Cleaning messages...")

df["clean_text"] = df["message"].apply(
    clean_text
)


# ============================================================
# 8. INPUT AND OUTPUT
# ============================================================

X = df["clean_text"]

y = df["label"]


# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "Training messages:",
    len(X_train)
)

print(
    "Testing messages:",
    len(X_test)
)


# ============================================================
# 10. TF-IDF
# ============================================================

print("Creating TF-IDF...")

tfidf_vectorizer = TfidfVectorizer(

    ngram_range=(1, 2),

    min_df=1,

    max_df=0.95
)


X_train_tfidf = tfidf_vectorizer.fit_transform(
    X_train
)


X_test_tfidf = tfidf_vectorizer.transform(
    X_test
)


print(
    "TF-IDF shape:",
    X_train_tfidf.shape
)


# ============================================================
# 11. TRAIN SVM MODEL
# ============================================================

print("Training SVM model...")

model = LinearSVC()

model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 12. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print()
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Ham",
            "Spam"
        ]
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 16. SAVE ML MODEL
# ============================================================

joblib.dump(
    model,
    "spam_model.pkl"
)

print()
print("Created: spam_model.pkl")


# ============================================================
# 17. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    tfidf_vectorizer,
    "tfidf_vectorizer.pkl"
)

print(
    "Created: tfidf_vectorizer.pkl"
)


# ============================================================
# 18. SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    "processed_spam.csv",
    index=False
)

print(
    "Created: processed_spam.csv"
)


print()
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)