"""Creates a set of 10k validation or test samples that will have 50%
same and different image pairs"""

import numpy as np
import sys
import os
import pickle

if __name__ == '__main__':
    # Get from teh args if this is training or test
    data_path = "data/processed/"
    file = None
    save_file = None
    np.random.seed(42)

    if len(sys.argv) < 2:
        print("Must provide one of the following: validate | test")
        sys.exit()
    if sys.argv[1] in ("validate", "test"):
        if sys.argv[1] =="validate":
            file = "validation_alphabet.pkl"
            save_file = os.path.join(data_path, "validation")
        else:
            file = "test_alphabet.pkl"
            save_file = os.path.join(data_path, "test")
    else:
        print("Argument must be either validate or test")
        sys.exit()

    # Once we have the file, read the appropriate data and generate the samples
    if os.path.exists(f"{save_file}X.npy"):
        print(f"Removing previous file {save_file}...")
        os.remove(f"{save_file}X.npy")
        os.remove(f"{save_file}Y.npy")

    with open(os.path.join(data_path,file), "rb") as f:
        alphabet_list = pickle.load(f)


    # Variables used for looping
    num_alphabets = len(alphabet_list)
    dataset = None
    labels = None
    # Now that we have the alphabet list we generate the 10k samples
    # of which half will be same and half will be equal
    # We do it in batches of 1000
    # We'll take the same number of samples per alphabet,
    # which is 10k /10 = 1000
    for i, alphabet in enumerate(alphabet_list):
        # sample 1000 images at a time to prevent OOM
        n_samples = 1000
        final_res = None
        alph_lab = None
        for _ in range(int(1000 / n_samples)):
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

            x2_alphabets = []
            x2_letters = []
            x2_drawers = []
            x2_images = []

            # make sure half samples have same letter
            # i.e. same alphabet, may or may not have same drawing.
            # for other half, make sure sample has different letter.
            # we need to be careful cause not all alphabets have the same number of letters

            # First half of the examples will be the same
            for index in range(n_samples // 2):
                _alphabet = x1_alphabet
                _letter = x1_letters[index]
                _drawing = np.random.randint(num_drawings)
                x2_alphabets.append(_alphabet)
                x2_letters.append(_letter)
                x2_drawers.append(_drawing)
                x2_images.append(alphabet_list[_alphabet][x2_letters[-1]][x2_drawers[-1]])
            print("Finished adding same letters")

            # Second half of the examples will be different
            for index in range(n_samples // 2, n_samples):
                _alphabet = np.random.randint(num_alphabets)
                x2_alphabets.append(_alphabet)
                # If it's the same alphabet choose a different letter
                n_letters = alphabet_list[_alphabet].shape[0]
                _letter = None
                _drawing = None
                if _alphabet == x1_alphabet:
                    _possible_letters = set(range(n_letters))
                    _possible_letters.remove(x1_letters[index])
                    _possible_letters = list(_possible_letters)
                    _letter = np.random.choice(_possible_letters)
                    _drawing = np.random.randint(num_drawings)
                    x2_letters.append(_letter)
                    x2_drawers.append(_drawing)
                else:
                    _letter = np.random.randint(n_letters)
                    _drawing = np.random.randint(num_drawings)
                    x2_letters.append(_letter)
                    x2_drawers.append(_drawing)
                x2_images.append(alphabet_list[_alphabet][x2_letters[-1]][x2_drawers[-1]])

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
                alph_lab = _labels
            else:
                # Concatenate to add to the first axis
                final_res = np.concatenate((final_res, pairs), axis=0)
                alph_lab = np.hstack((alph_lab, _labels))

        if dataset is None:
            dataset = final_res
            labels = alph_lab
        else:
            dataset = np.concatenate((dataset, final_res), axis=0)
            labels = np.hstack((labels, alph_lab))
        print(dataset.shape)
        print(f"Alphabet {i}")

    # Save everything
    with open(f"{save_file}X.npy", "wb") as f:
        np.save(f, dataset)
    with open(f"{save_file}Y.npy", "wb") as f:
        np.save(f, labels)



