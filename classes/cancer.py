from random import randint

import numpy as np
from classes.learnablenb import LearnableNB
from utils import processor_functions as pf

class Cancer(LearnableNB):

  class_names: list[str] = ['Benign', 'Malignant']

  num_classes: int = len(class_names) # number of classes
  num_attributes: int = 9 # number of attributes (excluding the class feature)
  domain_values: tuple[int, int] = (0, 9)  # range of possible discrete values
  num_values: int = domain_values[-1] - domain_values[0] + 1 # number of possible discrete values

  class_prior: np.array = np.zeros(num_classes, dtype=int)
  prob_tensor: np.array = np.ones((num_classes, num_attributes, num_values), dtype=float) #initialized to all 1's for smoothing

  def __init__(self, attributes: np.array(int), classify: bool = False):
    super().__init__(attributes, classify)

  @staticmethod
  def process_data(lines):
    """
    Process raw_data lines by filling missing values and shuffling examples.
    Parameters: lines (list of str): Raw raw_data lines from the input file.
    Returns:
    """
   # Ensure any missing values are filled in
    processed_lines = []
    for i in range(len(lines)):
      line = lines[i].split(',')
      line.pop(0)  # remove sample id from front of line
      line[-1] = int(line[-1])//2
      for j in range(len(line)):
        if line[j] == '?':
          line[j] = randint(*Cancer.domain_values)
        else:
          line[j] = int(line[j]) - 1
      processed_lines.append(line)
    examples = np.array(processed_lines, dtype=int)
    print(examples)
    # NO NEED TO BIN OR DOCUMENT since entries are already discrete
    #np.random.shuffle(examples) # ensure raw_data is in random order to eliminate bias
    noisy_examples = pf.add_noise(examples, 0.10)  # add noise to class, get a matrix of floats

    clean_lines = pf.array_to_lines(examples)    # get list of strings in proper format
    noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format

    return clean_lines, noisy_lines, None