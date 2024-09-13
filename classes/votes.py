import numpy as np
from classes.learnablenb import LearnableNB
from utils import processor_functions as pf

class Votes(LearnableNB):

  class_names: list[str] = ['republican', 'democrat']

  num_classes: int = len(class_names) # number of classes
  num_attributes: int = 16 # number of attributes (excluding the class feature)
  domain_values: tuple[int, int] = (0, 2)  # range of possible discrete values
  num_values: int = domain_values[-1] - domain_values[0] + 1 # number of possible discrete values

  class_prior: np.array = np.zeros(num_classes)
  prob_tensor: np.array = np.ones((num_classes, num_attributes, num_values)) #initialized to all 1's for smoothing

  def __init__(self, attributes: np.array(int), classify: bool = False):
    super().__init__(attributes, classify)

  @staticmethod
  def process_data(lines: list[str]):
    """
    Process raw_data lines by converting to digits, reordering class names, and shuffling examples.
    Parameters: lines (list of str): Raw raw_data lines from the input file.
    Returns: processed_lines (list of str): a list of strings with processed raw_data.
    """
    strings_digits: list[str] = ['?', 'y', 'n']  # ? is not a missing value
    processed_lines = []
    for i in range(len(lines)):
      line = lines[i].strip().split(',')
      line.append(line.pop(0))  # moves the class name from front to back
      class_names = Votes.class_names
      line[-1] = class_names.index(line[-1]) # sets class name to digit
      print("#################", len(line))
      for j in range(len(line)):
        line[j] = strings_digits.index(line[j]) # ensure all entries except class name/id are digits
      processed_lines.append(line)
    examples = np.array(processed_lines, dtype=int)  # ensure class uses a digit id, get a matrix of floats
    # NO NEED TO BIN OR DOCUMENT since entries are already discrete
    np.random.shuffle(examples) # ensure raw_data is in random order to eliminate bias
    noisy_examples = pf.add_noise(examples, 0.10)  # add noise to class, get a matrix of floats

    clean_lines = pf.array_to_lines(examples)    # get list of strings in proper format
    noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format

    return clean_lines, noisy_lines, None # get list of strings in proper format