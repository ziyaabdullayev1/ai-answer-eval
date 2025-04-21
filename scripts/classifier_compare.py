import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Veri yükle
df = pd.read_pickle("data/sample_with_vectors_e5_full.pkl")
y = df["label"].astype(int).values

print("Sinif Dagilimi:\n", df["label"].value_counts(), "\n")

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
models = {
    "xgb": XGBClassifier(objective="multi:softmax", num_class=4, eval_metric="mlogloss", use_label_encoder=False),
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    "mlp": MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300)
}

all_results = []

for model_name, model in models.items():
    print(f"\n Model: {model_name}")
    results = defaultdict(dict)

    for strategy in strategies:
        print(f"   Vektör stratejisi: {strategy}")
        X = np.stack(df.apply(lambda row: make_features(row, strategy), axis=1))
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # SMOTE
        sm = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

        clf = model
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
        results[strategy]["accuracy"] = report["accuracy"]
        results[strategy]["f1_macro"] = report["macro avg"]["f1-score"]

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"{model_name.upper()} - {strategy}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"results/confusion_matrix_{model_name}_{strategy}.png")
        plt.close()

    # Sonuçları kaydet
    for strategy in strategies:
        all_results.append({
            "model": model_name,
            "strategy": strategy,
            "accuracy": results[strategy]["accuracy"],
            "f1_macro": results[strategy]["f1_macro"]
        })

# Bar chart ile karşılaştır
df_res = pd.DataFrame(all_results)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_res, x="strategy", y="f1_macro", hue="model")
plt.title("Model & Vektör Stratejisi Karşılaştırması (F1-macro)")
plt.ylabel("F1-macro Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/f1_macro_model_vs_strategy.png")
plt.show()
