import os
import pickle
import re
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import (
    document_prompt,
    embedding_model,
    embedding_max_seq_length,
    retrieval_prompt,
)

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when', 'where',
    'why', 'how',
}


def load_cache(path, default=None):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    return default if default is not None else []

def load_embedding_model():
    device = 'cuda' if torch.cuda.is_available() else None
    model_kwargs = {'torch_dtype': torch.float16, 'attn_implementation': 'sdpa'} if torch.cuda.is_available() else {}
    model = SentenceTransformer(
        embedding_model,
        device=device,
        model_kwargs=model_kwargs,
        prompts={'document': document_prompt, 'query': retrieval_prompt},
        default_prompt_name='document',
    )
    model.max_seq_length = embedding_max_seq_length
    return model


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
    if 'stats' not in item:
        item['stats'] = analyze_note_text(item.get('text', '') or '')
    return item


def enrich_image_item(item):
    if 'metadata' not in item and item.get('path') and os.path.exists(item['path']):
        item['metadata'] = get_image_metadata(item['path'])
    return item


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

def embed_query(query, model):
    embedding = model.encode(query, prompt_name='query', normalize_embeddings=True)
    return np.asarray(embedding).flatten()


def find_similar_notes(query_embedding, notes, top_k=10, embeddings=None):
    if embeddings is None:
        embeddings = np.stack([np.asarray(note['embedding']) for note in notes])
    similarities = embeddings @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(notes[i]['title'], similarities[i]) for i in top_indices]