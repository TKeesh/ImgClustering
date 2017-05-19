### Requirements:

1. Installed python3.5 (recommended Anaconda3.5) with following packages: `tensorflow`, `opencv`, `sklearn`, `hdbscan`, `easydict`

**Note:** All required packages are listed in requirements.txt. You can try to execute:
  ```Shell
  pip install --upgrade -r requirements.txt
  ```

### Setting up data:
  Place images into ./data/ folder

### Generating embeddings for images:
  ```Shell
  python forward_resnet.py
  ```
  After this you will get ./output/ folder with embedding_data.npy and image_names_data.npy files

