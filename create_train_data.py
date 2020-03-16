import glob
import os
import sys
import imageio
import numpy as np
import pickle
import itertools
import gc


def extract_data(path):
    """
    Extracts the omniglot data from the path provided,
    organizes it by alphabet and returns it
    Parameters:
    -----------
    path: str
        Path at which the dataset is available

    Returns:
    --------
    alphabet_list: numpy.array shape(num_alphabets, num_letters, height, width)
        The images of the letters from the path, organized by alphabet
    """

    # List to store images,
    # one index for each alphabet
    alphabet_list = []

    # Iterate through alphabet folders
    alphabet_paths = glob.glob(train_path+"/*")
    for alphabet_path in alphabet_paths:
        # Get name of alphabet
        alphabet_name = alphabet_path.split("/")[-1]

        # Create new entry in list of alphabets
        alphabet_list.append([])

        # Get list of letter folders under alphabet folder
        letter_paths = glob.glob(alphabet_path+"/*")
        print(f"reading {alphabet_name}...")

        # Iterate through letter folders
        for letter_path in letter_paths:
            # Initialize list of images for all letters
            # belonging to an alphabet
            letter_images = []
            # get list of images inside the current letter folder
            image_paths = glob.glob(letter_path+"/*")

            # iterate through images
            for image_path in image_paths:
                # read image and add to the list of images
                image = imageio.imread(image_path)
                letter_images.append(image)

            # Append all images of letter to
            # current index of alphabet list
            alphabet_list[-1].append(np.stack(letter_images))

        # create numpy array for images of all alphabets
        # shape of np array each index: (20*num_letters, 105, 105)
        alphabet_list[-1] = np.stack(alphabet_list[-1])

    return alphabet_list


if __name__ == "__main__":
    """
    Run with command `python create_train_data.py [30|60|90]`
    """
    train_path = "data/raw/images_background"

    alphabet_list = None
    # train alphabet exists, load from file
    if os.path.exists("data/processed/train_alphabet.pkl"):
        with open("data/processed/train_alphabet.pkl", "rb") as f:
            alphabet_list = pickle.load(f)
        print(alphabet_list[0].shape)
    # else, create it
    else:
        print("creating data for train...")
        alphabet_list = extract_data(train_path)

        # Save alphabet_list to pickle,
        # since it takes a while to compute
        with open("data/processed/train_alphabet.pkl", "wb") as f:
            pickle.dump(alphabet_list, f)

    # read for argument from command line to determine
    # the size of dataset required to be generated
    size = None
    if len(sys.argv) < 2:
        print("must provide an argument for size of dataset (30, 90, or 150)")
        sys.exit()
    if sys.argv[1] in ("30", "90", "150"):
        size = float(sys.argv[1])
    else:
        print("argument must be one of 30, 90, or 150")
        sys.exit()

    num_alphabets = len(alphabet_list)
    alphabet_indexes = [i for i in range(num_alphabets)]
    # compute the number of samples per alphabet
    samples_per_alphabet = int((size*1000)/num_alphabets)

    # iterate through the images, alphabet wise
    for i, alphabet in enumerate(alphabet_list):

        # sample 1000 images at a time to prevent OOM
        n_samples = 1000
        for _ in range(int(samples_per_alphabet/n_samples)):
            gc.collect()
            # Initialize lists for alphabet pairs
            # and corresponding labels
            alphabets1 = []
            alphabets2 = []
            labels = []

            num_letters, num_drawings, _, _ = alphabet.shape
            alphabet_index1 = i
            # randomly draw `n_samples` samples
            letter_indexes1 = np.random.choice(num_letters, size=n_samples)
            drawing_indexes1 = np.random.choice(num_drawings, size=n_samples)

            # fetch samples and append to `alphabets1`
            alphabets1.append(alphabet[letter_indexes1, drawing_indexes1, :, :])

            # sample with equal probability same and different alphabets to alphabet1

            # To do this, we draw `n_samples` from a list with
            # `num_alphabets-1` instances of current alphabet and
            # 1 instance each of the other `num_alphabet-1` alphabets
            alphabet_indexes2 = np.random.choice([i]*(num_alphabets-1) + alphabet_indexes[:i] + alphabet_indexes[i+1:], size=n_samples)

            letter_indexes2 = []
            alphabet2 = []

            drawing_indexes2 = np.random.choice(num_drawings, size=n_samples)

            # for each alphabet corresponding to an index in
            # `alphabet_indexes2`, we sample a random image and
            # append to `alphabet2`
            for alphabet_index in alphabet_indexes2:
                letter_indexes2.append(np.random.choice(alphabet_list[alphabet_index].shape[0]))
                alphabet2.append(alphabet_list[alphabet_index])

            letter_indexes2 = np.array(letter_indexes2)
            alphabet2 = np.concatenate(alphabet2)
            alphabets2.append(alphabet2[letter_indexes2, drawing_indexes2, :, :])

            # create labels
            # 1 for `alphabet_indexes2` == i
            # 0 otherwise
            labels.append((np.repeat(i, n_samples) == alphabet_indexes2).astype(int))

            alphabets1 = np.expand_dims(np.concatenate(alphabets1), axis=1)
            alphabets2 = np.expand_dims(np.concatenate(alphabets2), axis=1)
            # create `alphabets`, shape: (size*1000, 2, 105, 105)
            alphabets = np.hstack((alphabets1, alphabets2))

            # shape: (size*1000, 1)
            labels = np.concatenate(labels).reshape(-1, 1)

            with open(f"data/processed/trainX_{int(size)}k.npy", "ab") as f:
                np.save(f, alphabets)
            del alphabets
            with open(f"data/processed/trainY_{int(size)}k.npy", "ab") as f:
                np.save(f, labels)
            del labels
        print(f"Alphabet {i}")

    # example code to read npy files
    # with open(f"data/processed/trainX_{int(size)}k.npy", "rb") as f:
    #     fsz = os.fstat(f.fileno()).st_size
    #     train_x = np.load(f)
    #     while f.tell() < fsz:
    #         train_x = np.vstack((train_x, np.load(f)))
    #     print(train_x.shape)

    # with open(f"data/processed/trainY_{int(size)}k.npy", "rb") as f:
    #     fsz = os.fstat(f.fileno()).st_size
    #     train_y = np.load(f)
    #     while f.tell() < fsz:
    #         train_y = np.vstack((train_y, np.load(f)))
    #     print(train_y.shape)
