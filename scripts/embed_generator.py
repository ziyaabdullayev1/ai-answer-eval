import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# e5 için
#model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
# cosmos-e5 için
model = SentenceTransformer("ytu-ce-cosmos/turkish-e5-large")
# jina için
#model = SentenceTransformer("jinaai/jina-embeddings-v3")

df = pd.read_excel("data/ogrenci_sorular_2025.xlsx")
df.columns = ["soru", "gpt4o", "deepseek", "label"]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df_sample = df.dropna().sample(1000, random_state=42).reset_index(drop=True)

def encode_column(texts, prefix="query: "):
    return [model.encode(prefix + str(t), normalize_embeddings=True) for t in tqdm(texts)]

df_sample["soru_vec"] = encode_column(df_sample["soru"], prefix="query: ")
df_sample["gpt4o_vec"] = encode_column(df_sample["gpt4o"], prefix="passage: ")
df_sample["deepseek_vec"] = encode_column(df_sample["deepseek"], prefix="passage: ")

# Vektörleri kaydet
#e5 için
#df_sample.to_pickle("data/sample_with_vectors.pkl")
#cosmos-e5 için
df_sample.to_pickle("data/sample_with_vectors_cosmos.pkl")
#jina için
#df_sample.to_pickle("data/sample_with_vectors_jina.pkl")
print("Vektörlü veri kaydedildi.")
