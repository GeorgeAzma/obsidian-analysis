import re
from collections import Counter
import pickle

from config import note_cache_file

def load_notes():
    with open(note_cache_file, "rb") as f:
        notes = pickle.load(f)
    return notes

def analyze_words():
    notes = load_notes()
    all_text = ""
    for note in notes:
        all_text += note['text'] + " "

    # Tokenize: lowercase, remove punctuation, split
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Remove common stop words (basic list)
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how'])
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    word_counts = Counter(filtered_words)
    
    print(f"Total words: {len(words)}")
    print(f"Unique words: {len(word_counts)}")
    print("\nTop 20 most common words:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")

if __name__ == "__main__":
    analyze_words()