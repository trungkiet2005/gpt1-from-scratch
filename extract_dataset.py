import pandas as pd

file_path = "train-00000-of-00001.parquet"

try:
    df = pd.read_parquet(file_path)
    
    texts = df["text"].tolist()
    list_texts = []
    for text in texts:
        list_texts.append(text)
    
    with open("dataset.txt", "w", encoding="utf-8") as f:
        for text in list_texts:
            f.write(text + "\n")

except Exception as e:
    print(f"Error loading dataset: {e}")


try:
    df_2 = pd.read_csv("poems_dataset.csv")
    texts_2 = df_2["content"].tolist()
    list_texts_2 = []
    for text in texts_2:
        list_texts_2.append(text)
    
    with open("dataset_2.txt", "w", encoding="utf-8") as f:
        for text in list_texts_2:
            f.write(text + "\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

