import os
import pickle

import torch
                
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from config import cache_file, model_name, vault_path, exclude_files
notes = []

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


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
                if title in exclude_files:
                    continue
                all_files.append(os.path.join(root, f))
    
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
            notes.append({"title": title, "text": text, "path": path, "embedding": emb.cpu().numpy().tolist()})
        torch.cuda.empty_cache()

    os.makedirs("output", exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(notes, f)
    
# if os.path.exists(cache_file):
#     with open(cache_file, "rb") as f:
#         notes = pickle.load(f)
# else:
process_notes(model_name)