import numpy as np
from utils import processor_functions as pf

class_name_id = {
    'republican': 1,
    'democrat': 4
}

votes_to_digits = {
    '?': 0, # not a missing value
    'y': 1,
    'n': 2
}


def process_data(lines):
    """
    Process raw_data lines by converting to digits, reordering class names, and shuffling examples.
    Parameters: lines (list of str): Raw raw_data lines from the input file.
    Returns: processed_lines (list of str): a list of strings with processed raw_data.
    """
    digit_lines = pf.strings_to_digits(lines, votes_to_digits)  # ensure all entries except class name/id are digits
    ordered_lines = pf.front_to_back(digit_lines)  # ensure class name/id is at the back of the attribute vector
    examples = pf.lines_to_array(ordered_lines, class_name_id)  # ensure class uses a digit id, get a matrix of floats
    # NO NEED TO BIN OR DOCUMENT since entries are already discrete
    np.random.shuffle(examples) # ensure raw_data is in random order to eliminate bias
    noisy_examples = pf.add_noise(examples, 0.10)   # add noise to class, get a matrix of floats
    clean_lines = pf.array_to_lines(examples)    # get list of strings in proper format
    noisy_lines = pf.array_to_lines(examples) # get list of strings in proper format

    return clean_lines, noisy_lines # get list of strings in proper format