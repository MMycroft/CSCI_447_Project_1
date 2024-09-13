import numpy as np
from classes.learnablenb import LearnableNB
from utils import processor_functions as pf

class Iris(LearnableNB):

  class_names: list[str] = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

  num_classes: int = len(class_names) # number of classes
  num_attributes: int = 4 # number of attributes (excluding the class feature)
  num_bins: int = 15 # arbitrary number of bins used for discretizing continuous values

  class_prior: np.array = np.zeros(num_classes)
  prob_tensor: np.array = np.ones((num_classes, num_attributes, num_bins)) #initialized to all 1's for smoothing

  def __init__(self, attributes: np.array(int), classify: bool = False):
    super().__init__(attributes, classify)

def process_data(lines: list[str]):
  """
  Process raw_data lines by binning attributes and shuffling examples.
  Parameters: lines (list of str): Raw raw_data lines from the input file.
  Returns: tuple: (processed_lines, documentation) where:
          - processed_lines is a list of strings with processed raw_data.
          - documentation is a string describing the binning information.
  """
  examples = pf.lines_to_numeric_array(lines, Iris.class_names)  # ensure class uses a digit id, get a matrix of floats
  bins = pf.get_attribute_bins(examples, Iris.num_bins)  # get list of attribute bin edges
  binned_examples = pf.bin_attributes(examples, bins)  # bin the examples
  np.random.shuffle(binned_examples)  # ensure raw_data is in random order to eliminate bias
  noisy_examples = pf.add_noise(binned_examples, 0.10)  # add noise to class, get a matrix of floats
  clean_lines = pf.array_to_lines(binned_examples)    # get list of strings in proper format
  noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format
  documentation = pf.get_bin_string(bins) # get string for documenting binning information

  return clean_lines, noisy_lines, documentation