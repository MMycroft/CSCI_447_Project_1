import numpy as np

from classes.learnablenb import LearnableNB
from utils import processor_functions as pf

class Soybean(LearnableNB):

    class_names: list[str] = ['D1', 'D2', 'D3', 'D4']

    num_classes: int = len(class_names)                             # number of classes
    num_features: int = 35                                          # number of features (excluding the class feature)
    domain_values: tuple[int, int] = (0, 6)                         # range of possible discrete values
    num_values: int = domain_values[-1] - domain_values[0] + 1      # number of possible discrete values

    class_prior: np.array = np.zeros(num_classes)
    prob_tensor: np.array = np.ones((num_classes, num_features, num_values)) # initiated with ones to avoid anihilation by 0

    @staticmethod
    def process_data(lines: list[str]):
        """
        Process raw_data lines by converting to digits and shuffling examples.
        Parameters: lines (list of str): Raw raw_data lines from the input file.
        Returns: processed_lines (list of str): a list of strings with processed raw_data.
        """
        examples = pf.lines_to_numeric_array(lines, Soybean.class_names)  # ensure class uses a digit id, get a matrix of floats
        # NO NEED TO BIN OR DOCUMENT since entries are already discrete
        np.random.shuffle(examples)  # ensure raw_data is in random order to eliminate bias
        noisy_examples = pf.add_noise(examples, 0.10)   # add noise to class, get a matrix of floats
        clean_lines = pf.array_to_lines(examples)    # get list of strings in proper format
        noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format

        return clean_lines, noisy_lines, None