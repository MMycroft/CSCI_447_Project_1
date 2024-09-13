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
  def set_class(self, new_id):
    if 0 <= new_id < self.num_classes:
      self.class_id = new_id
      self.class_name = self.class_names[self.class_id]
    else:
      self.class_id = None
      self.class_name = None

  # instancemethod
  def set_class_name(self, new_name):
    if new_name in self.class_names:
      self.class_name = new_name
      self.class_id = self.class_names.index(new_name)
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
    #   calculate the attribute likelihoods F(Aj = a_k, C = c_i)
    for e in training_examples:
      for attr_id in range(cls.num_attributes):
        value = e.attributes[attr_id]
        cls.prob_tensor[int(e.class_id)][int(attr_id)][int(value)] += 1    # counts occurrences of value in given attribute and class

    for c_id in range(cls.num_classes):
      for attr_id in range(cls.num_attributes):
        value_counts = cls.prob_tensor[c_id][attr_id]    # index of the array is the attribute value, count of the value is contained in the array
        total_count = np.sum(value_counts)
        if total_count> 0:  # check that nothing went wrong
          n_ci = total_count
          d = LearnableNB.num_values
          # plus 1 in the numerator is handled with the initialization of prob_tensor to all 1's
          cls.prob_tensor[c_id][attr_id] /= (n_ci + d)    # changes counts to probabilities

  @classmethod
  def naive_bayes_classifier(cls, test_examples: list['LearnableNB']):
    classified_examples = []
    C = [0] * cls.num_classes
    for e in test_examples:
      for c_id in range(cls.num_classes):
        Q = cls.class_prior[c_id]
        F = []
        for attr_id in range(cls.num_attributes):
          val = e.attributes[attr_id]
          f = cls.prob_tensor[c_id][attr_id][val]
          F.append(f)
        C[c_id] = Q * np.prod(F)
      class_id = C.index(max(C))
      e.set_class(class_id)
      classified_examples.append(e)
    return classified_examples

###################################################################################

  @staticmethod
  def zero_one_loss(classified_examples: list['LearnableNB']):
    total = len(classified_examples)
    correct = sum([e.attributes[-1] == e.class_id for e in classified_examples])
    loss = 1 - correct / total

    return loss

  @staticmethod
  def f1_score_loss(classified_examples: list['LearnableNB']):
    num_classes = classified_examples[0].num_classes
    score = 0
    counts = {}
    scores = {}
    for c in range(num_classes):
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

    return 1 - score / num_classes
#########################################################################

