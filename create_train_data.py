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
    alphabet_paths = glob.glob(train_path + "/*")

    # List of alphabets
    alphs = [a for a in alphabet_paths]
    # Choose 20 of the 40 alphabets to train on
    train_alphabets = np.random.choice(alphs, size=20, replace=False)
    # We should take these into account so we do not include them in validation or testing
    # Keep just the other 20 alphabets
    alphs = [a for a in alphs if a not in train_alphabets]
    print("Length of remaining alphabets ", len(alphs))

    # Get 10 for validation and the remaining for testing
    validation_alphabets = np.random.choice(alphs, size=10, replace=False)
    test_alphabets = [a for a in alphs if a not in validation_alphabets]

    print(f"Alphabets to train on ({len(train_alphabets)}): ", train_alphabets)
    print(f"Alphabets to validate on ({len(validation_alphabets)}): ", validation_alphabets)
    print(f"Alphabets to test on ({len(test_alphabets)}): ", test_alphabets)

    alphabet_groups = [train_alphabets, validation_alphabets, test_alphabets]
    # array where the final results will be kept (train, val, test)
    final_results = []

    # Get the appropriate images for each alphabet
    for i, group in enumerate(alphabet_groups):
        print(f'{i}: {len(group)} --> {group}')
        final_results.append([])
        alphabet_list = []
        for alphabet_path in group:
            # Get name of alphabet
            alphabet_name = alphabet_path.split("/")[-1]
            # Create new entry in list of alphabets

            alphabet_list.append([])

            # Get list of letter folders under alphabet folder
            letter_paths = glob.glob(alphabet_path + "/*")
            print(f"reading {alphabet_name}...")

            # Iterate through letter folders
            for letter_path in letter_paths:
                # Initialize list of images for all letters
                # belonging to an alphabet
                letter_images = []
                # get list of images inside the current letter folder
                image_paths = glob.glob(letter_path + "/*")

                # iterate through images for the first 12 drawers (training)
                # Get the appropriate slice for each group
                if i == 0:
                    # Training -> 12 drawers
                    slice = image_paths[:12]
                elif i == 1:
                    # Validation -> 4 drawers
                    slice = image_paths[12:16]
                else:
                    # Testing -> 4 drawers
                    slice = image_paths[16:]
                for image_path in slice:
                    # read image and add to the list of images
                    image = imageio.imread(image_path)
                    letter_images.append(image)

                # Append all images of letter to
                # current index of alphabet list
                alphabet_list[-1].append(np.stack(letter_images))

            # create numpy array for images of all alphabets
            # shape of np array each index: (20*num_letters, 105, 105)
            alphabet_list[-1] = np.stack(alphabet_list[-1])

        final_results[-1].append(alphabet_list)

    return final_results


if __name__ == "__main__":
    """
    Run with command `python create_train_data.py [30|60|90]`
    """
    print("Fixing the random seed to 42")
    np.random.seed(42)

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
        # alphabet_list = extract_data(train_path)
        tr, val, test = extract_data(train_path)

        # Save alphabet_list to pickle,
        # since it takes a while to compute
        with open("data/processed/train_alphabet.pkl", "wb") as f:
            pickle.dump(tr[0], f)
        with open("data/processed/validation_alphabet.pkl", "wb") as f:
            pickle.dump(val[0], f)
        with open("data/processed/test_alphabet.pkl", "wb") as f:
            pickle.dump(test[0], f)

    # read for argument from command line to determine
    # the size of dataset required to be generated
    size = None
    if len(sys.argv) < 2:
        print("must provide an argument for size of dataset (30, 90, or 150)")
        sys.exit()
    if sys.argv[1] in ("30", "90", "150"):
        size = float(sys.argv[1])
        print(f"size = {size}")
    else:
        print("argument must be one of 30, 90, or 150")
        sys.exit()

    # Check to delete the file before starting
    if os.path.exists(f'data/processed/trainX_{int(size)}k.npy'):
        print("Removing old files")
        os.remove(f'data/processed/trainX_{int(size)}k.npy')
        os.remove(f"data/processed/trainY_{int(size)}k.npy")

    num_alphabets = len(alphabet_list)
    alphabet_indexes = [i for i in range(num_alphabets)]
    # compute the number of samples per alphabet
    samples_per_alphabet = int((size * 1000) / num_alphabets)
    print(samples_per_alphabet)

    # Where we'll keep all the data
    dataset = None
    # iterate through the images, alphabet wise
    for i, alphabet in enumerate(alphabet_list):
        # sample 1000 images at a time to prevent OOM
        n_samples = 1500
        final_res = None
        labels = None
        for _ in range(int(samples_per_alphabet / n_samples)):
            # gc.collect()

            num_letters, num_drawings, _, _ = alphabet.shape


            x1_alphabet = i
            # randomly draw `n_samples` samples
            x1_letters = np.random.choice(num_letters, size=n_samples)
            x1_drawers = np.random.choice(num_drawings, size=n_samples)
            x1_images = []
            # fetch samples and append to `alphabets1`
            for l, d in zip(x1_letters, x1_drawers):
                x1_images.append(alphabet[l, d])

            # choose randomly 1000 alphabet indexes
            x2_alphabets = np.random.choice([i] * (num_alphabets - 1) +
                                            alphabet_indexes[:i] + alphabet_indexes[i + 1:], size=n_samples)
            x2_letters = []
            x2_drawers = []
            x2_images = []
            # For each of those alphabets, choose a specific letter
            # we need to be careful cause not all alphabets have the same number of letters
            for index in x2_alphabets:
                # Number of letters
                n_letters = alphabet_list[index].shape[0]
                x2_letters.append(np.random.choice(range(n_letters)))
                x2_drawers.append(np.random.choice(range(num_drawings)))
                x2_images.append(alphabet_list[index][x2_letters[-1]][x2_drawers[-1]])

            # Create the labels by comparing indexes
            _labels = []
            # Run through all the images and return same or different
            for alph2, letter1, letter2 in zip(x2_alphabets, x1_letters, x2_letters):
                # We know it's the same letter only if the alphabet indexes are the same
                # and the letter indexes are also the same
                if alph2 == i and letter1 == letter2:
                    _labels.append(1)
                else:
                    _labels.append(0)

            # Now we have the indexes of alphabets, letters and drawers for X2, so we just retrieve the images
            # We have the X1's in x1_images
            # We have the X2's in x2_images
            # We have the labels
            # Turn the arrays into (N,1,105,105)
            x1_images = np.expand_dims(x1_images, 1)
            x2_images = np.expand_dims(x2_images, 1)
            # Concatenate the arrays into (N,2,105,105)
            pairs = np.concatenate((x1_images, x2_images), axis=1)

            if final_res is None:
                final_res = pairs
                labels = _labels
            else:
                # Concatenate to add to the first axis
                final_res = np.concatenate((final_res, pairs), axis=0)
                labels = np.hstack((labels, _labels))

        if dataset is None:
            dataset = final_res
        else:
            dataset = np.concatenate((dataset, final_res), axis=0)
        print(dataset.shape)
        print(f"Alphabet {i}")

    # Save everything
    with open(f'data/processed/trainX_{int(size)}k.npy', 'wb') as f:
        np.save(f, dataset)
    with open(f'data/processed/trainY_{int(size)}k.npy', 'wb') as f:
        np.save(f, labels)

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
