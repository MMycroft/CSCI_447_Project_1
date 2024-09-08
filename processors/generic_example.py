import numpy as np
from utils import processor_functions as pf

class_name_id = {
    'name_1': 1,
    'name_2': 2,
    'name_3': 3,
    'name_4': 4
}

entries_to_digits = {
    'entry_1': 0,
    'entry_2': 1,
    'entry_3': 2
}

def process_data(lines):
    """
    This file is a generic example and is not used. All other process_data
    functions in this package are formatted after this function according
    to the needs of the .raw_data file.

    Process raw_data lines by converting to digits, reordering class names,
    filling missing values, binning attributes, and shuffling examples.
    Parameters: lines (list of str): Raw raw_data lines from the input file.

    Returns: tuple: (processed_lines, documentation) where:
            - processed_lines is a list of strings with processed raw_data.
            - documentation is a string describing the binning information.
    """

    digit_lines = pf.strings_to_digits(lines, entries_to_digits)    # ensure all entries except class name/id are digits
    ordered_lines = pf.front_to_back(digit_lines)   # ensure class name/id is at the back of the attribute vector
    better_lines = pf.fix_missing_value(ordered_lines, 1, 10)   # ensure any missing values are filled in
    examples = pf.lines_to_array(better_lines, class_name_id, False)    # ensure class uses a digit id, get a matrix of floats
    bins = pf.get_attribute_bins(examples, 15)  # get list of attribute bin edges
    binned_examples = pf.bin_attributes(examples, bins) # bin the examples
    np.random.shuffle(binned_examples)  # ensure raw_data is in random order to eliminate bias
    noisy_examples = pf.add_noise(binned_examples, 0.10)   # add noise to class, get a matrix of floats
    clean_lines = pf.array_to_lines(binned_examples)    # get list of strings in proper format
    noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format
    documentation = pf.get_bin_string(bins) # get string for documenting binning information

    return clean_lines, noisy_lines, documentation