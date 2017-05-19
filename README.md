### Requirements:

1. Installed python3.5 (recommended Anaconda3.5) with following packages: `tensorflow`, `opencv`, `sklearn`, `hdbscan`, `easydict`

**Note:** All required packages are listed in requirements.txt. You can try to execute:
  ```Shell
  pip install --upgrade -r requirements.txt
  ```

### Setting up data:
1. Place images into `./data/ folder`
2. Download pretrained model from `ensorflow-resnet-pretrained-20160509.tar.gz.torrent`
3. Extract all the files into `./models/`

### Generating embeddings for images:
1. Executing:
```Shell
python forward_resnet.py
```
With parametar ```-m (50, 101, 152)``` you can execute this script on ResNet-L50, ResNet-L101 and ResNet-L152 models.

**Note:** If you want to see embeddings in tensorboard, execute with parametar ```--tb```

After this you will get `./output/` folder with `embedding_data.npy` and `image_names_data.npy`

2. Visualizing tensorboard if created:
```Shell
tensorboard --logdir tensorboard/test_data
```

 ### bla
   1. 


