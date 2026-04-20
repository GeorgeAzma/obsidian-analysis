import os
import numpy as np
from config import note_cache_file
from src.util import embed_query, find_similar_notes, load_embedding_model, load_cache

if __name__ == "__main__":
    if not os.path.exists(note_cache_file):
        print(f"No note cache found at {note_cache_file}; run main.py first.")
        raise SystemExit(1)

    notes = load_cache(note_cache_file)
    if not notes:
        print("No notes found.")
        raise SystemExit(0)

    note_embeddings = np.stack([np.asarray(note['embedding']) for note in notes])

    model = load_embedding_model()
    
    while True:
        query = input("Enter query: ")
        if query.lower() == 'quit':
            break
        query_emb = embed_query(query, model)
        results = find_similar_notes(query_emb, notes, embeddings=note_embeddings)
        print(f"Top {len(results)} matching notes for '{query}':")
        for title, sim in results:
            print(f"- {title} (similarity: {sim:.3f})")
        print()
