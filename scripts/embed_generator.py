import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# e5 için
# model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
# print("Using e5 model to generate embeddings...")
#
# cosmos-e5 için
# model = SentenceTransformer("ytu-ce-cosmos/turkish-e5-large")
# print("Using cosmos-e5 model to generate embeddings...")
#
# jina için
model = SentenceTransformer("jinaai/jina-embeddings-v3")
print("Using jina model to generate embeddings...")

df = pd.read_excel("data/ogrenci_sorular_2025.xlsx")
df.columns = ["soru", "gpt4o", "deepseek", "label"]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df_copy = df.copy()
df = df.dropna().reset_index(drop=True)
#df_sample = df_copy.dropna().sample(1000, random_state=42).reset_index(drop=True)

def encode_column(texts, prefix="query: "):
    return [model.encode(prefix + str(t), normalize_embeddings=True) for t in tqdm(texts)]

#df_sample["soru_vec"] = encode_column(df_sample["soru"], prefix="query: ")
#df_sample["gpt4o_vec"] = encode_column(df_sample["gpt4o"], prefix="passage: ")
#df_sample["deepseek_vec"] = encode_column(df_sample["deepseek"], prefix="passage: ")

df["soru_vec"] = encode_column(df["soru"], prefix="query: ")
df["gpt4o_vec"] = encode_column(df["gpt4o"], prefix="passage: ")
df["deepseek_vec"] = encode_column(df["deepseek"], prefix="passage: ")

#e5 için
# df.to_pickle("data/sample_with_vectors_e5_full.pkl")

#cosmos-e5 için
# df.to_pickle("data/sample_with_vectors_cosmos_full.pkl")

#jina için
df.to_pickle("data/sample_with_vectors_jina_full.pkl")

print("Vektörlü veri kaydedildi.")
