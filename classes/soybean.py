import numpy as np
from utils import processor_functions as pf

class_name_id = {
    'D1': 1,
    'D2': 2,
    'D3': 3,
    'D4': 4
}

def process_data(lines):
    """
    Process raw_data lines by converting to digits and shuffling examples.
    Parameters: lines (list of str): Raw raw_data lines from the input file.
    Returns: processed_lines (list of str): a list of strings with processed raw_data.
    """
    examples = pf.lines_to_array(lines, class_name_id)   # ensure class uses a digit id, get a matrix of floats
    # NO NEED TO BIN OR DOCUMENT since entries are already discrete
    np.random.shuffle(examples)  # ensure raw_data is in random order to eliminate bias
    noisy_examples = pf.add_noise(examples, 0.10)   # add noise to class, get a matrix of floats
    clean_lines = pf.array_to_lines(examples)    # get list of strings in proper format
    noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format

    return clean_lines, noisy_lines