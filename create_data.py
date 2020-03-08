import glob
import os
import imageio
import numpy as np
import pickle


def create_data(path):
  """
  Extracts the omniglot data from the path provided

  Parameters:
  -----------
  path: str
    Path at which the dataset is available

  Returns:
  --------
  X: numpy.array (num_letters, num_drawings, height, width)
    The images of the letters from the path

  alphabet_dict: dict
    Key is name of alphabets and value is a
    list of start and end index for the alphabet in axis 0 of X
  """
  X = []
  alphabet_dict = {}
  
  # counter to keep track of alphabet indices
  i = 0

  # iterate through alphabet folders
  alphabet_paths = glob.glob(train_path+"/*")
  for alphabet_path in alphabet_paths:
    # get name of alphabet
    alphabet_name = alphabet_path.split("/")[-1]
    # initialize dictionary value
    alphabet_dict[alphabet_name] = [i, None]
    # get list of letter folders under alphabet folder
    letter_paths = glob.glob(alphabet_path+"/*")
    print(f"reading {alphabet_name}...")
    # iterate through letter folders
    for letter_path in letter_paths:
      # initialize list of images for all letters
      letter_images = []
      # get list of images inside the current letter folder
      image_paths = glob.glob(letter_path+"/*")

      # iterate through images
      for image_path in image_paths:
        # read image and add to the list of images
        image = imageio.imread(image_path)
        letter_images.append(image)

      # append all images of a letter to X
      X.append(np.stack(letter_images))

      # update alphabet_dict
      i += 1
      alphabet_dict[alphabet_name][1] = i - 1

  # create numpy array from list X
  X = np.stack(X)

  return X, alphabet_dict

if __name__ == "__main__":
  train_path = "data/raw/images_background"
  test_path = "data/raw/images_evaluation"
  save_path = "data/processed"

  print("creating data for train...")
  train_X, train_alphabets = create_data(train_path)
  print("creating data for evaluation...")
  test_X, test_alphabets = create_data(test_path)

  with open(os.path.join(save_path,"train.pkl"), "wb") as f:
    pickle.dump((train_X, train_alphabets), f)

  with open(os.path.join(save_path,"eval.pkl"), "wb") as f:
    pickle.dump((test_X, test_alphabets), f)