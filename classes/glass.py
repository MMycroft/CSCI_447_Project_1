import numpy as np
from utils import processor_functions as pf

class_name_id = {   # class_name_id is unneeded and is included for completeness
    "building_windows_float_processed": 1,
    "building_windows_non_float_processed": 2,
    "vehicle_windows_float_processed": 3,
    "vehicle_windows_non_float_processed": 4,
    "containers": 5,
    "tableware": 6,
    "headlamps": 7
}

def process_data(lines):
    """
    Process raw_data lines by binning attributes and shuffling examples.
    Parameters: lines (list of str): Raw raw_data lines from the input file.
    Returns: tuple: (processed_lines, documentation) where:
            - processed_lines is a list of strings with processed raw_data.
            - documentation is a string describing the binning information.
    """
    examples = pf.lines_to_array(lines, class_name_id, True) # get a matrix of floats
    bins = pf.get_attribute_bins(examples, 15)    # get list of attribute bin edges
    binned_examples = pf.bin_attributes(examples, bins) # bin the example values
    np.random.shuffle(binned_examples)  # ensure raw_data is in random order to eliminate bias
    noisy_examples = pf.add_noise(binned_examples, 0.10)   # add noise to class, get a matrix of floats
    clean_lines = pf.array_to_lines(binned_examples)    # get list of strings in proper format
    noisy_lines = pf.array_to_lines(noisy_examples) # get list of strings in proper format
    documentation = pf.get_bin_string(bins) # get string for documenting binning information

    return clean_lines, noisy_lines, documentation