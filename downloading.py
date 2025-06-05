import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import ollama
import tempfile
import time


model_name = "minicpm-v:8b"
parquet_file = "train-00000-of-00330.parquet"


# Ścieżki wyjściowe
INSIDE_DIR = './dataset/inside'
OUTSIDE_DIR = './dataset/outside'
not_known_DIR = './dataset/not_known'
os.makedirs(INSIDE_DIR, exist_ok=True)
os.makedirs(OUTSIDE_DIR, exist_ok=True)



# Wczytaj Parquet do DataFrame
df = pd.read_parquet(parquet_file)

headers = {
    "User-Agent": "MyWITScript/1.0 (mieszkowskifff@gmail.com)"
}


def classify_and_save(image_url, caption, idx):
    try:
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"[{idx}] Błąd pobierania obrazu: {e}")
        return

    try:
        prompt = f"""You are an image scene classifier. Based on the image and the following caption, classify the scene strictly as "inside", "outside" or "not known".

Caption: "{caption}"

Respond only with: "inside", "outside" or "not known".
"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        result = ollama.chat(
            model = model_name,
            messages = [{
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }]
        )

        reply = result["message"]["content"].strip().lower()
        if reply == "inside":
            out_path = os.path.join(INSIDE_DIR, f"{idx}_{time.time()}.jpg")
        elif reply == "outside":
            out_path = os.path.join(OUTSIDE_DIR, f"{idx}_{time.time()}.jpg")
        elif reply == "not known":
            out_path = os.path.join(not_known_DIR, f"{idx}_{time.time()}.jpg")
        else:
            print(f"[{idx}] Nieoczekiwana odpowiedź modelu: {reply}")
            return

        image.save(out_path)
        print(f"[{idx}] Zapisano: {reply} → {out_path}")

    except Exception as e:
        print(f"[{idx}] Błąd modelu: {e}")

for idx, row in df.iterrows():
    classify_and_save(row['image_url'], row['caption_attribution_description'], idx)
