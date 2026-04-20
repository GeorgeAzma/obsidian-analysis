note_cache_file = "output/notes.pkl"
image_cache_file = "output/images.pkl"
summary_cache_file = "output/summary.pkl"

embedding_model = "Qwen/Qwen3-VL-Embedding-2B" # "Qwen/Qwen3-VL-Embedding-8B" for better quality but much slower
embedding_max_seq_length = 2048
retrieval_prompt = "Retrieve relevant documents for the query."
document_prompt = "Represent this note, image, or query for semantic retrieval."

vault_path = "obsidian"
image_path = "images"

import os

# parse .env file
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8') as env_file:
        for line in env_file:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

exclude_notes = os.environ.get("EXCLUDE_NOTES", "").split(",")
exclude_images = os.environ.get("EXCLUDE_IMAGES", "").split(",")