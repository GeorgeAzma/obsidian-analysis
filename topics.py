import pickle
import numpy as np
import hdbscan
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import re
from config import cache_file, gen_model_name

def load_notes():
    with open(cache_file, "rb") as f:
        notes = pickle.load(f)
    return notes

def cluster_notes(notes):
    embeddings = np.array([note['embedding'] for note in notes])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean', cluster_selection_method='leaf')
    labels = clusterer.fit_predict(embeddings)
    # Compute centroids for each cluster
    centroids = {}
    for label in set(labels):
        if label != -1:  # Exclude noise
            cluster_embs = embeddings[labels == label]
            centroids[label] = np.mean(cluster_embs, axis=0)
    return labels, centroids

def get_cluster_notes(notes, labels, cluster_id, centroids):
    cluster_notes = [(note, centroids[cluster_id]) for note, label in zip(notes, labels) if label == cluster_id]
    # Compute distances
    distances = []
    for note, centroid in cluster_notes:
        emb = np.array(note['embedding'])
        dist = np.linalg.norm(emb - centroid)
        distances.append((note, dist))
    # Sort by distance ascending
    distances.sort(key=lambda x: x[1])
    sorted_notes = [note for note, _ in distances]
    return sorted_notes

def generate_topic_name(notes, tokenizer, model):
    prompt = f"Generate a short theme name (1-2 words) that describes these topics: {" | ".join(list(map(lambda note: note['title'], notes[:20])))}. Theme:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():        
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.3, top_p=0.9)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract after "Topic name:"
    if "Topic name:" in generated:
        name = generated.split("Topic name:")[-1].strip()
    else:
        name = generated[len(prompt):].strip()
    # Clean up: take first line, remove punctuation, limit to 3 words
    name = name.split('\n')[0]
    name = re.sub(r'[^\w\s]', '', name).strip()
    name = ' '.join(name.split()[:3])
    return name
 
def main():
    notes = load_notes()
    if not notes:
        print("No notes found.")
        return

    labels, centroids = cluster_notes(notes)

    # Load generative model
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )

    topics = []
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    print(f"Found {len(unique_labels)} clusters")
    
    for cluster_id in unique_labels:
        cluster_notes_list = get_cluster_notes(notes, labels, cluster_id, centroids)
        if not cluster_notes_list:
            continue
        titles = [note['title'] for note in cluster_notes_list]
        topic_name = generate_topic_name(cluster_notes_list, tokenizer, model)
        topics.append({
            "title": topic_name,
            "notes": titles
        })

    with open("output/topics.json", "w") as f:
        json.dump(topics, f, indent=2)

    print("Topics generated and saved to output/topics.json")
    print("\nAll generated topics:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic['title']} ({len(topic['notes'])} notes)")

if __name__ == "__main__":
    main()