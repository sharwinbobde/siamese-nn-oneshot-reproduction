import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

# Take the evaluation dataset
data_path = "data/processed/"
file = "validation_alphabet.pkl"

# create the alphabet list
with open(os.path.join(data_path, file), 'rb') as f:
    alphabet_list = pickle.load(f)


def create_oneshot_example(N):
    """ Creates a batch with N examples, one of the images
    is the reference images and the others are the candidates.
    Only one of the candidates represents the same character, but
    drawn by a different drawer"""
    # get a random character
    random_alph = np.random.randint(len(alphabet_list))
    _n_characters, _n_drawers = alphabet_list[random_alph].shape[:2]
    true_character = np.random.randint(_n_characters)
    ex1, ex2 = np.random.choice(_n_drawers, replace=False, size=(2,))

    # Create a stacked version of the original character
    X1 = np.asarray([alphabet_list[random_alph][true_character, ex1, :, :]] * N).reshape(N, 1, 105, 105)
    x2 = alphabet_list[random_alph][true_character, ex2, :, :]

    # Now generate the other noise examples
    support_alphabets = np.random.choice(len(alphabet_list), size=(N - 1,))
    # Create the support set and set the first image as the true same character

    support_set = x2
    support_set = np.expand_dims(support_set, 0)

    for a in support_alphabets:
        if a != random_alph:
            # Get a letter whatsoever
            _n_characters, _n_drawers = alphabet_list[a].shape[:2]

            character = np.random.randint(_n_characters)
            drawer = np.random.randint(_n_drawers)
            image = alphabet_list[a][character, drawer, :, :].reshape(1, 105, 105)
            support_set = np.concatenate((support_set, image), axis=0)

        else:
            _n_characters, _n_drawers = alphabet_list[a].shape[:2]
            possible_characters = np.arange(_n_characters)
            possible_characters = np.setxor1d(possible_characters, [true_character])
            character = np.random.choice(possible_characters)
            drawer = np.random.randint(_n_drawers)
            image = alphabet_list[a][character, drawer, :, :].reshape(1, 105, 105)
            support_set = np.concatenate((support_set, image), axis=0)

    labels = np.zeros((N,))
    np.random.shuffle(support_set)
    for j in range(support_set.shape[0]):
        if np.array_equal(support_set[j], x2):
            labels[j] = 1

    support_set = np.expand_dims(support_set, axis=1)

    return X1, support_set, labels


if __name__ == '__main__':
    # Get 400 random examples of same and different characters
    np.random.seed(42)
    num_samples = 1500
    N = 20
    # Each pair is gonna be a batch of N images
    X_oneshot = np.zeros((num_samples, N, 2, 105, 105), dtype='uint8')
    Y_oneshot = np.zeros((num_samples, N), dtype='uint8')

    if os.path.exists(os.path.join(data_path,"X_oneshot.npy")):
        print(f"Removing previous files...")
        os.remove(os.path.join(data_path,"X_oneshot.npy"))
        os.remove(os.path.join(data_path, "Y_oneshot.npy"))

    for i in range(num_samples):
        X, support_set, labels = create_oneshot_example(N)
        # expand dims
        pair = np.concatenate((X, support_set), axis=1)
        X_oneshot[i] = pair
        Y_oneshot[i] = labels
        if i % 10 == 0:
            print(i)

    with open(os.path.join(data_path, 'X_oneshot.npy'), 'wb') as f:
        np.save(f, X_oneshot)
    with open(os.path.join(data_path, 'Y_oneshot.npy'), 'wb') as f:
        np.save(f, Y_oneshot)
