import numpy as np
from utils import processor_functions as pf
from abc import ABC, abstractmethod

class LearnableNB(ABC):

  class_names: list[str] = []
  entries_digits: list[str] = []

  num_classes: int = len(class_names)                             # number of classes
  num_attributes: int = 0                                         # number of attributes (excluding the class attribute)
  domain_values: tuple[int, int] = (0,0)                          # range of possible discrete values
  num_values: int = domain_values[-1] - domain_values[0] + 1      # number of possible discrete values
  num_bins: int = 0                                               # number of possible discrete bins

  class_prior: np.array   #initialized to all 0's
  prob_tensor: np.array   #initialized to all 1's for smoothing

  @classmethod
  def naive_bayes_trainer(cls, training_examples: list['LearnableNB']):
    n = len(training_examples)
    #1. For each class in the training set, calculate the class prior probability Q(C=c_i)
    for e in training_examples:
      cls.class_prior[e.class_id] += 1    # counts occurrences of class
    cls.class_prior /= n    # changes counts to probabilities
    #2. Data is seperated into class subsets by using three indices: class (i), attribute (j), value (k)
    #   The attribute probabilities are stored in a 3D array (tensor!!)

    #3. For each attribute in the class-specific training set,
    #   calculate the attribute-value prior probability F(Aj = a_k, C = c_i)
    for e in training_examples:
      for j in range(cls.num_attributes - 1): # ignore last attribute (attribute id)
        attribute = j
        value = e.attributes[j]
        cls.prob_tensor[e.class_id][attribute][value] += 1    # counts occurrences of value in given attribute and class

    for i in range(cls.num_classes):
      for j in range(cls.num_attributes - 1): # ignore last attribute (attribute id)
        value_counts = cls.prob_tensor[i][j]    # index of the array is the attribute value, count of the value is contained in the array
        total_count = np.sum(value_counts)
        if total_count> 0:  # check that nothing went wrong
          n_ci = total_count
          d = LearnableNB.num_values
          # plus 1 in the numerator is handled with the initialization of prob_tensor to all 1's
          cls.prob_tensor[i][j] /= (n_ci + d)    # changes counts to probabilities

  @classmethod
  def naive_bayes_classifier(cls, test_examples: list['LearnableNB']):
    probabilities = []
    for e in test_examples:
      class_prob = []
      for i in range(cls.num_classes):
        p = cls.class_prior[i]
        for j in range(cls.num_attributes - 1): # ignore last attribute (attribute id)
          attribute = j
          value = e.attributes[j]
          p *= cls.prob_tensor[i][attribute][value]
        class_prob.append(p)
        probabilities.append(class_prob)
      class_id = class_prob.index(max(class_prob))
      e.set_class_id(class_id)    # sets class id and class name

  @staticmethod
  def zero_one_loss(classified_examples: list['LearnableNB']):
    results = []
    for e in classified_examples:
      results.append(e.attributes[-1] == e.class_id)
    return sum(results) / len(results)

  @staticmethod
  def f1_score_loss(classified_examples: list['LearnableNB']):
    x = {'TP': 0, 'FP': 0, 'FN': 0}
    for c in range(LearnableNB.num_classes):
      for e in classified_examples:
        x['TP'] += int(e.class_id == c and e.attributes[-1] == e.class_id)
        x['FP'] += int(e.class_id == c and e.attributes[-1] != e.class_id)
        x['FN'] += int(e.attributes[-1] == c and e.attributes[-1] != e.class_id)
    precision = x['TP'] / (x['TP'] + x['FP'])
    recall = x['TP'] / (x['TP'] + x['FN'])
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

  # CONSTRUCTOR
  def __init__(self, attributes: np.array(int), classify: bool=False):

    self.attributes: np.array(int) = attributes
    self.class_id: int = self.attributes[-1] if classify else -1
    self.class_name: str = LearnableNB.class_names[self.class_id // 2 - 1] if classify else ''

  # instancemethod
  def classify(self):
    self.class_id = self.attributes[-1]
    self.class_name = LearnableNB.class_names[self.class_id // 2 - 1]

  # instancemethod
  def set_class_id(self, new_id):
    if 0 <= new_id <= LearnableNB.num_classes * 2:
      self.class_id = new_id
      self.class_name = LearnableNB.class_names[self.class_id // 2 - 1]
    else:
      self.class_id = -1
      self.class_name = ''

  # instancemethod
  def set_class_name(self, new_name):
    if new_name in LearnableNB.class_names:
      self.class_name = new_name
      self.class_id = LearnableNB.class_names.index(new_name)
    else:
      self.class_name = ''
      self.class_id = -1

  @abstractmethod
  def process_data(self, lines: list[str]):
    """
    This is a generic example and is not used. All other process_data
    functions in this package are formatted after this function according
    to the needs of the .raw_data file.

    Process raw_data lines by converting to digits, reordering class names,
    filling missing values, binning attributes, and shuffling experiments.
    Parameters: lines (list of str): Raw raw_data lines from the input file.

    Returns: tuple: (processed_lines, documentation) where:
            - processed_lines is a list of strings with processed raw_data.
            - documentation is a string describing the binning information.
    """
    # digit_lines = pf.strings_to_digits(lines, LearnableNB.entries_digits)  # ensure all entries except class name/id are digits
    # ordered_lines = pf.front_to_back(digit_lines)  # ensure class name/id is at the back of the attribute vector
    # better_lines = pf.fix_missing_value(ordered_lines, 1, 10)  # ensure any missing values are filled in
    # best_lines = pf.remove_sample_id(better_lines)
    # experiments = pf.lines_to_numeric_array(best_lines, LearnableNB.class_names)  # ensure class uses a digit id, get a matrix of floats
    # bins = pf.get_attribute_bins(experiments, 15)  # get list of attribute bin edges
    # binned_experiments = pf.bin_attributes(experiments, bins)  # bin the experiments
    # np.random.shuffle(binned_experiments)  # ensure raw_data is in random order to eliminate bias
    # noisy_experiments = pf.add_noise(binned_experiments, 0.10)  # add noise to class, get a matrix of floats
    # clean_lines = pf.array_to_lines(binned_experiments)  # get list of strings in proper format
    # noisy_lines = pf.array_to_lines(noisy_experiments)  # get list of strings in proper format
    # documentation = pf.get_bin_string(bins)  # get string for documenting binning information
    #
    # return clean_lines, noisy_lines, documentation
    pass