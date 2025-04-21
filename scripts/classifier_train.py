import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# 1. VERI
df = pd.read_pickle("data/sample_with_vectors_e5_full.pkl")
y = df["label"].astype(int).values

print("Sinif Dagilimi:")
print(df["label"].value_counts(), "\n")

# 2. FARKLI VEKTOR STRATEJILERI
def make_features(row, strategy="all"):
    s, g, d = row["soru_vec"], row["gpt4o_vec"], row["deepseek_vec"]
    if strategy == "all":
        return np.concatenate([s - g, s - d, g - d, np.abs(s - g), np.abs(s - g) - np.abs(s - d)])
    elif strategy == "abs_sg":
        return np.abs(s - g)
    elif strategy == "absdiff":
        return np.abs(s - g) - np.abs(s - d)
    elif strategy == "concat_all":
        return np.concatenate([s, g, d])
    elif strategy == "sg_sd":
        return np.concatenate([s - g, s - d])
    else:
        raise ValueError("Unknown strategy")

strategies = ["all", "abs_sg", "absdiff", "concat_all", "sg_sd"]

results = defaultdict(dict)

for strategy in strategies:
    print(f"\nVektor Stratejisi: {strategy}")

    X = np.stack(df.apply(lambda row: make_features(row, strategy), axis=1))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X,
        y_encoded, test_size=0.2, random_state=42)

    clf = XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=1,
        subsample=0.2,
        colsample_bytree=0.1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Classification Report:")
    report = classification_report(y_test, y_pred,
                                   output_dict=True,
                                   zero_division=0.0)
    print(report)

    results[strategy]["accuracy"] = report["accuracy"]
    results[strategy]["f1_macro"] = report["macro avg"]["f1-score"]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.title(f"Confusion Matrix - {strategy}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{strategy}.png")
    plt.close()

# OZET
print("\nStrateji Bazli Sonuclar:")
for strat in strategies:
    acc = results[strat]["accuracy"]
    f1 = results[strat]["f1_macro"]
    print(f"{strat:12}  | Accuracy: {acc:.3f} | F1-macro: {f1:.3f}")
