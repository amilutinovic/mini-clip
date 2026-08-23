# mini-clip

[CLIP (Contrastive Language–Image Pre-training)](https://openai.com/index/clip/) is OpenAI's model that connects images and their text descriptions. This project is a miniature version of the real CLIP.

It consists of:
- [`Image encoder`](image_encoder.ipynb): Converts input images into an embedding space
- [`Text encoder`](text_encoder.ipynb): Converts input text description into an embedding space

Model then trains, using these embeddings, using a method called `contrastive learning`.

## Dataset

We use the [Flickr8k dataset](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k), which contains 8,000 images, each paired with five English captions.


#### 1. Download the text annotations with:

```bash
mkdir -p data
cd data

wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

unzip Flickr8k_Dataset.zip
unzip Flickr8k_text.zip

rm Flickr8k_Dataset.zip Flickr8k_text.zip
rm Flickr8k_Dataset.zip
```

## Authors:
- Đurđa Milošević
- Staša Đorđević
- Anja Milutinović