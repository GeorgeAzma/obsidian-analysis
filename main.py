import os
import pickle

import torch
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from config import (
    note_cache_file,
    image_cache_file,
    note_embedding_model,
    image_embedding_model,
    vault_path,
    image_path,
    exclude_notes,
    exclude_images,
)

notes = []
images = []


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def ensure_output_dir():
    os.makedirs("output", exist_ok=True)


def load_cache(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def process_notes(model_name, batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )

    all_files = []
    for root, _, files in os.walk(vault_path):
        for f in files:
            if f.endswith(".md"):
                title = os.path.basename(f)[:-3]
                if title in exclude_notes:
                    continue
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"No note files found in {vault_path}; skipping note embeddings.")
        return []

    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        batch_texts = []
        batch_titles = []

        for path in batch_files:
            with open(path, "r", encoding="utf-8") as file:
                title = os.path.basename(path)[:-3]
                text = file.read()
                batch_titles.append(title)
                batch_texts.append(f"# {title}\n\n{text}")
                print(f"{i + len(batch_titles)}/{len(all_files)} {title}")

        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=4096, padding='longest', truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        for title, text, path, emb in zip(batch_titles, batch_texts, batch_files, embeddings):
            notes.append({"title": title, "text": text, "path": path, "embedding": emb.tolist()})
        torch.cuda.empty_cache()

    with open(note_cache_file, "wb") as f:
        pickle.dump(notes, f)
    return notes


def process_images(model_name, batch_size=1):
    model = SentenceTransformer(model_name)

    all_files = []
    for root, _, files in os.walk(image_path):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                if os.path.basename(f) in exclude_images:
                    continue
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"No images found in {image_path}; skipping image embeddings.")
        return []

    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        batch_inputs = []

        for path in batch_files:
            batch_inputs.append({"image": path})
            print(f"{i + len(batch_inputs)}/{len(all_files)} {os.path.basename(path)}")

        embeddings = model.encode(batch_inputs, normalize_embeddings=True)

        for path, emb in zip(batch_files, embeddings):
            images.append({
                "path": path,
                "title": os.path.basename(path),
                "embedding": emb.tolist(),
            })

    with open(image_cache_file, "wb") as f:
        pickle.dump(images, f)
    
    return images


def main():
    ensure_output_dir()

    if os.path.exists(vault_path):
        if os.path.exists(note_cache_file):
            print(f"Loaded: {note_cache_file}")
            notes.extend(load_cache(note_cache_file))
        else:
            notes.extend(process_notes(note_embedding_model))
    else:
        print(f"No note files found in {vault_path}; skipping note embedding generation.")

    if os.path.exists(image_path):
        if os.path.exists(image_cache_file):
            print(f"Loaded: {image_cache_file}")
            images.extend(load_cache(image_cache_file))
        else:
            images.extend(process_images(image_embedding_model))
    else:
        print(f"No images found in {image_path}; skipping image embedding generation.")


if __name__ == "__main__":
    main()
