import numpy as np
from utils import processor_functions as pf

class_name_id = {
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica': 3
}

def process_data(lines):
    """
    Process data lines by binning attributes and shuffling examples.
    Parameters: lines (list of str): Raw data lines from the input file.
    Returns: tuple: (processed_lines, documentation) where:
            - processed_lines is a list of strings with processed data.
            - documentation is a string describing the binning information.
    """
    examples = pf.lines_to_array(lines, class_name_id)  # ensure class uses a digit id, get a matrix of floats
    bins = pf.get_attribute_bins(examples, 15)    # get list of attribute bin edges
    binned_examples = pf.bin_attributes(examples, bins) # bin the examples
    np.random.shuffle(binned_examples)  # ensure data is in random order to eliminate bias
    processed_lines = pf.array_to_lines(binned_examples)    # get list of strings in proper format
    documentation = pf.get_bin_string(bins) # get string for documenting binning information

    return processed_lines, documentation