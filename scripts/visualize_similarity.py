import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from tqdm import tqdm

# Dosyayı oku
df = pd.read_pickle("data/sample_with_vectors_e5_full.pkl")

# Vektörleri ayıkla
s_vectors = np.stack(df["soru_vec"].values)
g_vectors = np.stack(df["gpt4o_vec"].values)
d_vectors = np.stack(df["deepseek_vec"].values)
labels = df["label"].values

# Cosine similarity hesapla (her örnek için)
gpt4o_similarities = [cosine_similarity([s], [g])[0][0] for s, g in zip(s_vectors, g_vectors)]
deepseek_similarities = [cosine_similarity([s], [d])[0][0] for s, d in zip(s_vectors, d_vectors)]

# Spearman korelasyon
gpt4o_corr, _ = spearmanr(gpt4o_similarities, labels)
deepseek_corr, _ = spearmanr(deepseek_similarities, labels)

# Top-1 ve Top-5 hesaplama
def evaluate_topk(s_vectors, answer_vectors, labels):
    top1_count = 0
    top5_count = 0
    for i in range(len(s_vectors)):
        s_vec = s_vectors[i].reshape(1, -1)
        sims = cosine_similarity(s_vec, answer_vectors)[0]
        top5_idx = sims.argsort()[-5:][::-1]
        top5_labels = labels[top5_idx]

        real_label = labels[i]
        if top5_labels[0] == real_label:
            top1_count += 1
        if real_label in top5_labels:
            top5_count += 1

    total = len(s_vectors)
    return top1_count / total, top5_count / total

gpt4o_top1, gpt4o_top5 = evaluate_topk(s_vectors, g_vectors, labels)
deepseek_top1, deepseek_top5 = evaluate_topk(s_vectors, d_vectors, labels)

# Bilgi yazdır
print(f"[GPT-4o] Spearman corr = {gpt4o_corr:.3f}, Top-1 = {gpt4o_top1:.3f}, Top-5 = {gpt4o_top5:.3f}")

print(f"[Deepseek] Spearman corr = {deepseek_corr:.3f}, Top-1 = {deepseek_top1:.3f}, Top-5 = {deepseek_top5:.3f}")

# Grafik çizimi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(gpt4o_similarities, labels, alpha=0.5)
plt.title(f"GPT-4o\ncorr={gpt4o_corr:.2f} | Top-1={gpt4o_top1:.2f}, Top-5={gpt4o_top5:.2f}")
plt.xlabel("Cosine Similarity")
plt.ylabel("Label")

plt.subplot(1, 2, 2)
plt.scatter(deepseek_similarities, labels, alpha=0.5, color="orange")
plt.title(f"Deepseek\ncorr={deepseek_corr:.2f} | Top-1={deepseek_top1:.2f}, Top-5={deepseek_top5:.2f}")
plt.xlabel("Cosine Similarity")
plt.ylabel("Label")

plt.tight_layout()
output_file_path = "results/similarity_vs_label_with_topk.png"
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
with open(output_file_path, "w") as file:
    plt.savefig(output_file_path)
    plt.show()
