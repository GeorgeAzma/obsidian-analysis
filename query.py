import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from config import note_cache_file, note_embedding_model

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def load_notes():
    with open(note_cache_file, "rb") as f:
        notes = pickle.load(f)
    return notes

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def embed_query(query, tokenizer, model):
    task = 'Given a knowledge base query, retrieve relevant notes that answer the query'
    instructed_query = get_detailed_instruct(task, query)
    inputs = tokenizer([instructed_query], return_tensors="pt", max_length=4096, padding='longest', truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy().flatten()

def find_similar_notes(query_embedding, notes, top_k=10):
    embeddings = np.array([note['embedding'] for note in notes])
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(notes[i]['title'], similarities[i]) for i in top_indices]

if __name__ == "__main__":
    notes = load_notes()
    tokenizer = AutoTokenizer.from_pretrained(note_embedding_model, padding_side='left')
    model = AutoModel.from_pretrained(
        note_embedding_model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )
    
    while True:
        query = input("Enter query: ")
        if query.lower() == 'quit':
            break
        query_emb = embed_query(query, tokenizer, model)
        results = find_similar_notes(query_emb, notes)
        print(f"Top {len(results)} matching notes for '{query}':")
        for title, sim in results:
            print(f"- {title} (similarity: {sim:.3f})")
        print()
