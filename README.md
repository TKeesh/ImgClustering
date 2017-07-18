## Mozgalo2017 - Unsupervised content based image clustering

By Paolo Čerić and Tomislav Kiš

### Requirements

1. Installed Python3.5 (recommended Anaconda3.5) with following packages: `tensorflow`, `opencv`, `sklearn`, `hdbscan`, `easydict`

    **Note:** All required packages are listed in requirements.txt. You can try to execute:
    ```Shell
    pip install --upgrade -r requirements.txt
    ```

### Setting up data
You can use our `data_preprocessing.py` script to change dataset to `.jpg` or make some image transformations which are not crucial
```Shell
python data_preprocessing.py input_dir output_dir
```
`-m` - smart images preprocessing

`-c` - removes padding


1. Place `.jpg` images into `./data/` folder

2. Download pretrained model from `tensorflow-resnet-pretrained-20160509.tar.gz.torrent`
    
    If you want to retrain your own model please refer to: `https://github.com/ry/tensorflow-resnet`

3. Extract all the files into `./models/`

### Generating embeddings for images
1. Executing:
    ```Shell
    python forward_resnet.py
    ```
    With parameter ```-m (50, 101, 152)``` you can execute this script on ResNet-L50, ResNet-L101 and ResNet-L152 models.

    This will generate `./output/` folder with `embeddings_data.npy` and `image_names_data.npy`

    **Note:** If you want to see embeddings in tensorboard, execute with parameter ```--tb```    

  If you want to skip this step you can use our `*.npy` files from this commit for **Mozgalo2017_dataset**

2. Visualizing tensorboard if created:
    ```Shell
    tensorboard --logdir tensorboard/test_data
    ```

### Analyzing embeddings and creating clusters

1. Working on generated (`./output/embeddings_data.*`) embeddings:
    ```Shell
    python embeddings_processing.py
    ```
    Use the parameter `-n` to set the number of embeddings to work on

2. Create folders with clustered dataset:
    `--cf`

    Folders with cluster indices will be created in `./output/data_clusters`

3. Precission boost method:
    `-p`

    With TSNE boosts precission of noise images clustering, executes much slower
    
    Optional method which is significantly slower and only a bit more accurate. Can give better results on larger datasets.

4. Adjust algoritham to work on specific datasets:
    `--mcs`

    Set the minimum cluster size which algoritham finds

5. **Working on Mozgalo2017_dataset embeddings:**
    ```Shell
    python embeddings_processing.py -embspath './output/embeddings_data_0.npy'
    ```
    If using `--cf` all Mozgalo2017_dataset images must be in `./data/` or in custom folder specified with `-imgspath`. 
    Output: `./output/DATA_embeddings`, DATA calculated from imgspath folder

### ToDo

1. Test precission_boost on larger datasets, adjust TSNE

2. Detecting small clusters in the noise and detecting false noise clustering

3. Splitting clusters into the small clusters with visualization

4. Reduce min_cluster_size recursively while n_clusters < threshold - adjust model on the fly





    


