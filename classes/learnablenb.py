import numpy as np
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

  ##################################################################################

  # CONSTRUCTOR
  def __init__(self, attributes, classify: bool=False):

    self.attributes = attributes
    self.class_id = None
    self.class_name = None
    if classify:
      self.class_id = int(self.attributes[-1])
      self.class_name = self.class_names[self.class_id]

  # instancemethod
  def classify(self):
    self.class_id = self.attributes[-1]
    self.class_name = self.class_names[self.class_id]

  # instancemethod
  def set_class_id(self, new_id):
    if 0 <= new_id <= LearnableNB.num_classes * 2:
      self.class_id = new_id
      self.class_name = LearnableNB.class_names[self.class_id]
    else:
      self.class_id = None
      self.class_name = None

  # instancemethod
  def set_class_name(self, new_name):
    if new_name in LearnableNB.class_names:
      self.class_name = new_name
      self.class_id = LearnableNB.class_names.index(new_name)
    else:
      self.class_name = None
      self.class_id = None

  @classmethod
  def naive_bayes_trainer(cls, training_examples: list['LearnableNB']):
    n = len(training_examples)
    #1. For each class in the training set, calculate the class prior probability Q(C=c_i)
    for e in training_examples:
      cls.class_prior[e.class_id] += 1    # counts occurrences of class
    cls.class_prior = cls.class_prior/n    # changes counts to probabilities
    #2. Data is seperated into class subsets by using three indices: class (i), attribute (j), value (k)
    #   The attribute probabilities are stored in a 3D array (tensor!!)

    #3. For each attribute in the class-specific training set,
    #   calculate the attribute-value prior probability F(Aj = a_k, C = c_i)
    for e in training_examples:
      for j in range(cls.num_attributes):
        attribute = j
        value = e.attributes[j]
        cls.prob_tensor[int(e.class_id)][int(attribute)][int(value)] += 1    # counts occurrences of value in given attribute and class

    for i in range(cls.num_classes):
      for j in range(cls.num_attributes):
        value_counts = cls.prob_tensor[i][j]    # index of the array is the attribute value, count of the value is contained in the array
        total_count = np.sum(value_counts)
        if total_count> 0:  # check that nothing went wrong
          n_ci = total_count
          d = LearnableNB.num_values
          # plus 1 in the numerator is handled with the initialization of prob_tensor to all 1's
          cls.prob_tensor[i][j] /= (n_ci + d)    # changes counts to probabilities

  @classmethod
  def naive_bayes_classifier(cls, test_examples: list['LearnableNB']):
    print("TEST", test_examples)
    classified_examples = []
    probabilities = []
    for e in test_examples:
      class_prob = []
      for i in range(cls.num_classes):
        p = cls.class_prior[i]
        for j in range(cls.num_attributes):
          attribute = j
          value = e.attributes[j]
          value_prob = cls.prob_tensor[i][attribute][value]
          p *= value_prob
        class_prob.append(p)
      probabilities.append(class_prob)
      class_id = class_prob.index(max(class_prob))
      e.set_class_id(class_id)    # sets class id and class name
      classified_examples.append(e)
    return classified_examples

###################################################################################

  @staticmethod
  def zero_one_loss(classified_examples: list['LearnableNB']):
    results = []
    for e in classified_examples:
      results.append(e.attributes[-1] == e.class_id)
    return sum(results) / len(results)

  @staticmethod
  def f1_score_loss(classified_examples: list['LearnableNB']):
    score = 0
    counts = {}
    scores = {}
    for c in range(LearnableNB.num_classes):
      counts[c] = {'TP': 0, 'FP': 0, 'FN': 0}
      scores[c] = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
      for e in classified_examples:
        counts[c]['TP'] += int(e.class_id == c and e.attributes[-1] == e.class_id)
        counts[c]['FP'] += int(e.class_id == c and e.attributes[-1] != e.class_id)
        counts[c]['FN'] += int(e.attributes[-1] == c and e.attributes[-1] != e.class_id)
      scores[c]['precision'] = counts[c]['TP'] / (counts[c]['TP'] + counts[c]['FP'])
      scores[c]['recall'] = counts[c]['TP'] / (counts[c]['TP'] + counts[c]['FN'])
      scores[c]['f1_score'] = (2 * scores[c]['precision'] * scores[c]['recall']) / (scores[c]['precision'] + scores[c]['recall'])
    score += scores[c]['f1_score']
    return score / LearnableNB.num_classes

#########################################################################

