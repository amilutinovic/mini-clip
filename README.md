# mini-clip

[CLIP (Contrastive Language–Image Pre-training)](https://openai.com/index/clip/) is OpenAI's model that connects images and their text descriptions. This project is a miniature version of the real CLIP.

It consists of:
- [`Image encoder`](02_image_encoder.ipynb): Converts input images into an embedding space
- [`Text encoder`](03_text_encoder.ipynb): Converts input text description into an embedding space

Model then trains, using these embeddings, using a method called `contrastive learning`.

## Dataset

We use the [Flickr8k dataset](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k), which contains 8,000 images, each paired with five English captions.


#### 1. Download the dataset from Kaggle:

1. Open the [Flickr8k dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2. Log in to your Kaggle account.
3. Click **Download**.
4. Extract the downloaded ZIP file into the `data/` directory.

## Pretrained Models
Trained model checkpoints (best models for all three image encoders) are available for download:
[Google Drive folder](https://drive.google.com/drive/folders/1k_PvqrXmhrLr_RMuwzBttYqbs3OQkYUL?usp=sharing)

After downloading, place the `.pt` files into the `models/` directory before running evaluation.

## Authors:
- Đurđa Milošević
- Staša Đorđević
- Anja Milutinović