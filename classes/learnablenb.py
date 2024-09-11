import numpy as np
from utils import processor_functions as pf
from abc import ABC, abstractmethod

class LearnableNB(ABC):

    class_names: list[str] = []
    entries_digits: list[str] = []

    num_classes: int = len(class_names)                             # number of classes
    num_features: int = 0                                           # number of features (excluding the class feature)
    domain_values: tuple[int, int] = (0,0)                          # range of possible discrete values
    num_values: int = domain_values[-1] - domain_values[0] + 1      # number of possible discrete values
    num_bins: int = 0

    class_prior: np.array
    prob_tensor: np.array                                           # initiated with ones to avoid anihilation by 0

    @classmethod
    def naive_bayes_trainer(cls, train_examples: list['LearnableNB']):
        num_examples = len(train_examples)

        for e in train_examples:
            cls.class_prior[e.class_id] += 1
        cls.class_prior /= num_examples

        for e in train_examples:
            for feature, value in enumerate(e.features):
                cls.prob_tensor[e.class_id][feature][value] += 1
        for i in range(cls.num_classes):
            for j in range(cls.num_features):
                num_values = np.sum(cls.prob_tensor[i][j])
                if num_values > 0:
                    cls.prob_tensor[i][j] = (cls.prob_tensor[i][j] + 1)/num_values

    @classmethod
    def naive_bayes_classifier(cls, test_examples: list['LearnableNB']):
        for e in test_examples:
            class_prob = []
            for i, p in enumerate(cls.class_prior):
                x = p
                for feature, value in enumerate(e.features):
                    x *= cls.prob_tensor[i][feature][value]
                class_prob.append(x)
            class_id = class_prob.index(max(class_prob))
            e.set_class_id(class_id)

    @staticmethod
    def naive_bayes_tester(classified_examples: list['LearnableNB']):
        results = []
        for i, e in enumerate(classified_examples):
            results.append(e.features[-1] == e.class_id)

    # CONSTRUCTOR
    def __init__(self, features: list[int], classify: bool=False):

        self.features: list[int] = features
        self.class_id: int = self.features[-1] if classify else -1
        self.class_name: str = LearnableNB.class_names[self.class_id // 2 - 1] if classify else ''

    # instancemethod
    def classify(self):
        self.class_id = self.features[-1]
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
    @staticmethod
    def process_data(lines: list[str]):
        """
        This is a generic example and is not used. All other process_data
        functions in this package are formatted after this function according
        to the needs of the .raw_data file.

        Process raw_data lines by converting to digits, reordering class names,
        filling missing values, binning attributes, and shuffling examples.
        Parameters: lines (list of str): Raw raw_data lines from the input file.

        Returns: tuple: (processed_lines, documentation) where:
                - processed_lines is a list of strings with processed raw_data.
                - documentation is a string describing the binning information.
        """

        digit_lines = pf.strings_to_digits(lines, LearnableNB.entries_digits)  # ensure all entries except class name/id are digits
        ordered_lines = pf.front_to_back(digit_lines)  # ensure class name/id is at the back of the attribute vector
        better_lines = pf.fix_missing_value(ordered_lines, 1, 10)  # ensure any missing values are filled in
        best_lines = pf.remove_sample_id(better_lines)
        examples = pf.lines_to_numeric_array(best_lines, LearnableNB.class_names)  # ensure class uses a digit id, get a matrix of floats
        bins = pf.get_attribute_bins(examples, 15)  # get list of attribute bin edges
        binned_examples = pf.bin_attributes(examples, bins)  # bin the examples
        np.random.shuffle(binned_examples)  # ensure raw_data is in random order to eliminate bias
        noisy_examples = pf.add_noise(binned_examples, 0.10)  # add noise to class, get a matrix of floats
        clean_lines = pf.array_to_lines(binned_examples)  # get list of strings in proper format
        noisy_lines = pf.array_to_lines(noisy_examples)  # get list of strings in proper format
        documentation = pf.get_bin_string(bins)  # get string for documenting binning information

        return clean_lines, noisy_lines, documentation


