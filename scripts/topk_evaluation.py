import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Dosya yolu
#e5 için
df = pd.read_pickle("data/sample_with_vectors.pkl")
# cosmos-e5 için
#df = pd.read_pickle("data/sample_with_vectors_cosmos.pkl")
# jina için
#df = pd.read_pickle("data/sample_with_vectors_jina.pkl")

# Vektör matrisleri
s_vectors = np.stack(df["soru_vec"].values)
g_vectors = np.stack(df["gpt4o_vec"].values)
d_vectors = np.stack(df["deepseek_vec"].values)
labels = df["label"].values

def evaluate_topk(s_vectors, answer_vectors, labels, model_name="GPT-4o"):
    top1_count = 0
    top5_count = 0
    for i in tqdm(range(len(s_vectors))):
        s_vec = s_vectors[i].reshape(1, -1)
        sims = cosine_similarity(s_vec, answer_vectors)[0]
        top5_idx = sims.argsort()[-5:][::-1]  # En benzer 5
        top5_labels = labels[top5_idx]

        # Gerçek etiketin en iyi cevapla eşleşip eşleşmediğini kontrol et
        real_label = labels[i]
        if top5_labels[0] == real_label:
            top1_count += 1
        if real_label in top5_labels:
            top5_count += 1

    total = len(s_vectors)
    top1_score = top1_count / total
    top5_score = top5_count / total
    print(f"{model_name} - Top-1 Accuracy: {top1_score:.4f}, Top-5 Accuracy: {top5_score:.4f}")
    return top1_score, top5_score

# GPT-4o için
evaluate_topk(s_vectors, g_vectors, labels, model_name="GPT-4o")

# Deepseek için
evaluate_topk(s_vectors, d_vectors, labels, model_name="Deepseek")
