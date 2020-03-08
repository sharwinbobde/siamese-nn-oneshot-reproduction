# siamese-nn-oneshot-reproduction
Reproduction of "Siamese Neural Networks for One-shot Image Recognition," by Gregory Koch, Richard Zemel, Ruslan Salakhutdinov (ICML 2015)

Download data from the Omniglot repository : https://github.com/brendenlake/omniglot

Koch et. al. Keras implementation for reference: https://github.com/sorenbouma/keras-oneshot

## Setting up repository for development
1. download `images_background` and `images_evaluation` from (here)[https://github.com/brendenlake/omniglot] and unzip into `data/raw`.
2. create a virtual environment and install from `requirements.txt`
3. run `create_data.py` to create data for training and evaluation.

## Data Shape

`create_data.py` converts the image from background and evaluation into numpy arrays of shape (number of letters, number of drawings of letters(20), height, width) and pickles them