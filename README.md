# siamese-nn-oneshot-reproduction
Reproduction of "Siamese Neural Networks for One-shot Image Recognition," by Gregory Koch, Richard Zemel, Ruslan Salakhutdinov (ICML 2015)

Download data from the Omniglot repository : https://github.com/brendenlake/omniglot

Koch et. al. Keras implementation for reference: https://github.com/sorenbouma/keras-oneshot

## Setting up repository for development
1. download `images_background` and `images_evaluation` from (here)[https://github.com/brendenlake/omniglot] and unzip into `data/raw`.
2. create a virtual environment and install from `requirements.txt`
3. run `python create_data.py [30|60|90]` to create data for training. e.g., `python create_data.py 30` creates training data of size 30,000 in accordance with the paper and stores it in `data/processed/trainX_30k.npy` and `data/processed/trainY_30k.npy`. Code snippet to read the `.npy` file is included in the last part of `create_data.py`.

## Data Shape

`create_data.py` converts the image from background and evaluation into numpy arrays of shape:

trainX_30k: (30000, 2, 105, 105), trainY_30k: (30000, 1)

trainX_90k: (90000, 2, 105, 105), trainY_90k: (90000, 1)

trainX_150k: (150000, 2, 105, 105), trainY_150k: (150000, 1)

Where index 0 is number of samples, index 1 is the pairs of images (1 pair, 2 images), index 2 and 3 are the height and width of the image
