import numpy as np
from utils import processor_functions as pf

class_name_id = {
    'Benign': 2,
    'Malignant': 4
}

def process_data(lines):
    """
    Process data lines by filling missing values and shuffling examples.
    Parameters: lines (list of str): Raw data lines from the input file.
    Returns: processed_lines (list of str): a list of strings with processed data.
    """
    better_lines = pf.fix_missing_value(lines, 1, 10)   # Ensure any missing values are filled in
    examples = pf.lines_to_array(better_lines, class_name_id, True) # Ensure class uses a digit id, get a matrix of floats
    # NO NEED TO BIN OR DOCUMENT since entries are already discrete
    np.random.shuffle(examples) # ensure data is in random order to eliminate bias
    processed_lines = pf.array_to_lines(examples)   # get list of strings in proper format

    return processed_lines