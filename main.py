import os
import pickle

from config import embedding_model, exclude_images, exclude_notes, image_cache_file, image_path, note_cache_file, summary_cache_file, vault_path
from src.util import (
    analyze_note_text,
    build_global_summary,
    enrich_image_item,
    enrich_note_item,
    get_image_metadata,
    load_cache,
    load_embedding_model,
)

notes = []
images = []


def process_notes(model, batch_size=1):
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
        batch_inputs = []
        batch_records = []

        for path in batch_files:
            with open(path, "r", encoding="utf-8") as file:
                title = os.path.basename(path)[:-3]
                text = file.read()
                note_text = f"# {title}\n\n{text}"
                batch_inputs.append({"text": note_text})
                batch_records.append((path, title, note_text))
                print(f"{i + len(batch_inputs)}/{len(all_files)} {title}")

        embeddings = model.encode(
            batch_inputs,
            batch_size=batch_size,
            prompt_name='document',
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if getattr(embeddings, "ndim", 2) == 1:
            embeddings = [embeddings]

        for (path, title, text), emb in zip(batch_records, embeddings):
            notes.append({
                "kind": "note",
                "title": title,
                "text": text,
                "path": path,
                "embedding": emb,
                "embedding_model": embedding_model,
                "stats": analyze_note_text(text),
            })

    with open(note_cache_file, "wb") as f:
        pickle.dump(notes, f)
    return notes


def process_images(model, batch_size=1):
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

        embeddings = model.encode(
            batch_inputs,
            batch_size=batch_size,
            prompt_name='document',
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if getattr(embeddings, "ndim", 2) == 1:
            embeddings = [embeddings]

        for path, emb in zip(batch_files, embeddings):
            images.append({
                "kind": "image",
                "path": path,
                "title": os.path.basename(path),
                "embedding": emb,
                "embedding_model": embedding_model,
                "metadata": get_image_metadata(path),
            })

    with open(image_cache_file, "wb") as f:
        pickle.dump(images, f)
    
    return images


def main():
    os.makedirs("output", exist_ok=True)
    model = load_embedding_model()

    if os.path.exists(vault_path):
        if os.path.exists(note_cache_file):
            print(f"Loaded: {note_cache_file}")
            notes.extend(enrich_note_item(item) for item in load_cache(note_cache_file))
            with open(note_cache_file, "wb") as f:
                pickle.dump(notes, f)
        else:
            notes.extend(process_notes(model))
    else:
        print(f"No note files found in {vault_path}; skipping note embedding generation.")

    if os.path.exists(image_path):
        if os.path.exists(image_cache_file):
            print(f"Loaded: {image_cache_file}")
            images.extend(enrich_image_item(item) for item in load_cache(image_cache_file))
            with open(image_cache_file, "wb") as f:
                pickle.dump(images, f)
        else:
            images.extend(process_images(model))
    else:
        print(f"No images found in {image_path}; skipping image embedding generation.")

    with open(summary_cache_file, "wb") as f:
        pickle.dump(build_global_summary(notes, images), f)


if __name__ == "__main__":
    main()
