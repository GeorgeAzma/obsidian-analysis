Analyze and visualize your obsidian vault and images based on semantic similarity

![notes-2d](notes-2d.png) ![notes-3d](notes-3d.png)
![images-2d](images-2d.png) ![images-3d](images-3d.png)

### How to use

- `pip install -r requirements.txt`
- drop your obsidian notes inside `obsidian/`
- drop your images inside `images/`
- run `python main.py`
- run `python plot.py`

### Files

- `main.py` 
    - generates `notes.pkl` which contains all notes from `obsidian/` and their embeddings using `Qwen3-Embedding-4B`
    - generates `images.pkl` which contains all images from `images/` and their embeddings using `Qwen3-VL-Embedding-2B` 
- `plot.py` creates interactive 2D/3D semantic similarity point map of your notes/images, also displays word count and other info
- `query.py` searches top most related notes based on your query
- `gap.py` detects similar notes that are not linked `[[My Note]] <--> [[Related Note]]`
- `config.py` what embedding model to use etc.