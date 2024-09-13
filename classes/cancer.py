import numpy as np

from classes.learnablenb import LearnableNB
from utils import processor_functions as pf

class Cancer(LearnableNB):

  class_names: list[str] = ['Benign', 'Malignant']

  num_classes: int = len(class_names) # number of classes
  num_attributes: int = 10 # number of attributes (excluding the class feature)
  domain_values: tuple[int, int] = (1, 10)  # range of possible discrete values
  num_values: int = domain_values[-1] - domain_values[0] + 1 # number of possible discrete values

  class_prior: np.array = np.zeros(num_classes)
  prob_tensor: np.array = np.ones((num_classes, num_attributes, num_values)) #initialized to all 1's for smoothing

  def process_data(self, lines: list[str]):
    """
    Process raw_data lines by filling missing values and shuffling examples.
    Parameters: lines (list of str): Raw raw_data lines from the input file.
    Returns: processed_lines (list of str): a list of strings with processed raw_data.
    """
    better_lines = pf.fix_missing_value(lines, *Cancer.domain_values)   # Ensure any missing values are filled in
    best_lines = pf.remove_sample_id(better_lines)
    examples = pf.lines_to_numeric_array(best_lines, Cancer.class_names)  # Ensure class uses a digit id, get a matrix of floats
    # NO NEED TO BIN OR DOCUMENT since entries are already discrete
    np.random.shuffle(examples) # ensure raw_data is in random order to eliminate bias
    noisy_examples = pf.add_noise(examples, 0.10)  # add noise to class, get a matrix of floats
    clean_lines = pf.array_to_lines(examples)    # get list of strings in proper format
    noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format

    return clean_lines, noisy_lines, None

