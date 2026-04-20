import os
import re
import pickle
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from config import (
    note_cache_file,
    image_cache_file,
    summary_cache_file,
    note_embedding_model,
    image_embedding_model,
    vault_path,
    image_path,
    exclude_notes,
    exclude_images,
)

notes = []
images = []

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when', 'where',
    'why', 'how',
}


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


def analyze_note_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    return {
        'words': len(words),
        'unique_words': len(set(filtered_words)),
        'chars': len(text),
    }


def format_file_size(size_in_bytes):
    if size_in_bytes < 1024:
        return f'{size_in_bytes} B'
    if size_in_bytes < 1024 * 1024:
        return f'{size_in_bytes / 1024:.1f} KB'
    if size_in_bytes < 1024 * 1024 * 1024:
        return f'{size_in_bytes / (1024 * 1024):.1f} MB'
    return f'{size_in_bytes / (1024 * 1024 * 1024):.1f} GB'


def get_image_metadata(path):
    with Image.open(path) as image_file:
        width, height = image_file.size
    file_size = os.path.getsize(path)
    created_at = datetime.fromtimestamp(os.path.getctime(path)).strftime('%Y-%m-%d')
    return {
        'resolution': f'{width}x{height}',
        'file_size': format_file_size(file_size),
        'created_at': created_at,
    }


def enrich_note_item(item):
    enriched_item = dict(item)
    if 'stats' not in enriched_item:
        enriched_item['stats'] = analyze_note_text(enriched_item.get('text', '') or '')
    return enriched_item


def enrich_image_item(item):
    enriched_item = dict(item)
    if 'metadata' not in enriched_item and enriched_item.get('path') and os.path.exists(enriched_item['path']):
        enriched_item['metadata'] = get_image_metadata(enriched_item['path'])
    return enriched_item


def build_global_summary(note_items, image_items):
    word_count = 0
    unique_words = set()
    chars = 0

    for item in note_items:
        stats = item.get('stats') or {}
        word_count += stats.get('words', 0)
        chars += stats.get('chars', 0)

        text = item.get('text', '') or ''
        text_words = re.findall(r'\b\w+\b', text.lower())
        unique_words.update(word for word in text_words if word not in STOP_WORDS and len(word) > 2)

    return {
        'notes': len(note_items),
        'images': len(image_items),
        'words': word_count,
        'unique_words': len(unique_words),
        'chars': chars,
    }


def process_notes(model_name, batch_size=1, max_length=4096):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

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

        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=max_length, padding='longest', truncation=True).to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        del inputs, outputs

        for title, text, path, emb in zip(batch_titles, batch_texts, batch_files, embeddings):
            notes.append({
                "title": title,
                "text": text,
                "path": path,
                "embedding": emb.tolist(),
                "stats": analyze_note_text(text),
            })
        torch.cuda.empty_cache()

    with open(note_cache_file, "wb") as f:
        pickle.dump(notes, f)
    return notes


def process_images(model_name, batch_size=1):
    model = SentenceTransformer(model_name)
    model.eval()

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

        with torch.inference_mode():
            embeddings = model.encode(batch_inputs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

        for path, emb in zip(batch_files, embeddings):
            images.append({
                "path": path,
                "title": os.path.basename(path),
                "embedding": emb.tolist(),
                "metadata": get_image_metadata(path),
            })

    with open(image_cache_file, "wb") as f:
        pickle.dump(images, f)
    
    return images


def main():
    ensure_output_dir()

    if os.path.exists(vault_path):
        if os.path.exists(note_cache_file):
            print(f"Loaded: {note_cache_file}")
            notes.extend(enrich_note_item(item) for item in load_cache(note_cache_file))
            with open(note_cache_file, "wb") as f:
                pickle.dump(notes, f)
        else:
            notes.extend(process_notes(note_embedding_model))
    else:
        print(f"No note files found in {vault_path}; skipping note embedding generation.")

    if os.path.exists(image_path):
        if os.path.exists(image_cache_file):
            print(f"Loaded: {image_cache_file}")
            images.extend(enrich_image_item(item) for item in load_cache(image_cache_file))
            with open(image_cache_file, "wb") as f:
                pickle.dump(images, f)
        else:
            images.extend(process_images(image_embedding_model))
    else:
        print(f"No images found in {image_path}; skipping image embedding generation.")

    with open(summary_cache_file, "wb") as f:
        pickle.dump(build_global_summary(notes, images), f)


if __name__ == "__main__":
    main()
