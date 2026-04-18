import pickle
import numpy as np
import re
from config import note_cache_file

def load_notes():
    with open(note_cache_file, "rb") as f:
        notes = pickle.load(f)
    return notes

def extract_links(text):
    # Find all [[link]] patterns
    links = re.findall(r'\[\[([^\]]+)\]\]', text)
    return set(links)

def find_gaps():
    notes = load_notes()
    
    links_dict = {note['title']: extract_links(note['text']) for note in notes}
    
    # Embeddings
    embeddings = np.array([note['embedding'] for note in notes])
    
    potential_gaps = []
    seen_pairs = set()
    
    for i, note in enumerate(notes):
        if len(note['text']) < 100:  # Skip short notes
            continue
        emb = embeddings[i]
        similarities = np.dot(embeddings, emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb))
        # Exclude self
        similarities[i] = -1
        # Top 5 similar
        top_indices = np.argsort(similarities)[::-1][:5]
        
        for j in top_indices:
            sim = similarities[j]
            if sim > 0.8 and sim < 0.999:  # Threshold, exclude near-identical
                other_note = notes[j]
                if len(other_note['text']) < 100:  # Skip short
                    continue
                other_title = other_note['title']
                pair = frozenset({note['title'], other_title})
                if pair not in seen_pairs and other_title not in links_dict[note['title']]:
                    seen_pairs.add(pair)
                    potential_gaps.append((note['title'], other_title, sim))
    
    # Sort by similarity descending
    potential_gaps.sort(key=lambda x: x[2], reverse=True)
    
    print("Top potential gaps (high similarity but not linked):")
    for a, b, sim in potential_gaps[:50]:  # Top 50
        print(f"- {a} <--> {b} (similarity: {sim:.3f})")

if __name__ == "__main__":
    find_gaps()