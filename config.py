# Path to the cached notes file
cache_file = "output/notes.pkl"

# Model name for embeddings
model_name = "Qwen/Qwen3-Embedding-4B"

# Generative model for topic naming
gen_model_name = "microsoft/Phi-3.5-mini-instruct"

# Vault path
vault_path = "obsidian"

import os

# parse .env file
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8') as env_file:
        for line in env_file:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

exclude_files = os.environ.get("EXCLUDE_FILES", "").split(",")