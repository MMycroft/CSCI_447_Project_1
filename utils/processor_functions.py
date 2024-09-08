import numpy as np
from math import ceil
from random import randint, sample

def strings_to_digits(lines, strings_digits):
    """
    Converts string entries in each line to their corresponding digit values based on a mapping.

    Parameters:
        lines (list of str): List of lines, where each line is a comma-separated string.
        strings_digits (dict): Dictionary mapping string entries to digit values.

    Returns:
        list of str: List of lines with string entries replaced by their corresponding digit values.
    """
    for i in range(len(lines)):
        line = lines[i].split(',')
        for j in range(len(line) - 1):  #ignore class names
            line[j] = str(strings_digits.get(line[j], -1))
        lines[i] = ','.join(line)

    return lines

def front_to_back(lines):
    """
    Moves the class name from the front of each line to the back.

    Parameters:
        lines (list of str): List of lines, where each line is a comma-separated string with class name at the front.

    Returns:
        list of str: List of lines with the class name moved to the end of each line.
    """
    for i in range(len(lines)):
        line = lines[i].split(',')
        line.append(line.pop(0))  # moves the class name from front to back
        lines[i] = ','.join(line)

    return lines

def fix_missing_value(lines, min_val=1, max_val=10):
    """
    Replaces missing values (denoted by '?') in each line with a random integer in the specified range.

    Parameters:
        lines (list of str): The input lines, where each line is a comma-separated string.
        min_val (int): The minimum value to use for replacing missing values.
        max_val (int): The maximum value to use for replacing missing values.

    Returns:
        list of str: The lines with missing values replaced by random integers.
    """
    for i in range(len(lines)):
        line = lines[i].split(',')
        for j in range(len(line)):
            if line[j] == '?':
                line[j] = str(randint(min_val, max_val))
        lines[i] = ','.join(line)
    return lines

def lines_to_array(lines, class_id, example_id=False):
    """
    Converts a list of strings into a NumPy array of floats.

    Parameters:
        lines (list of str): The input lines, where each line is a comma-separated string.
        class_id (dict): A dictionary mapping class names to numeric IDs.
        example_id (bool): If True, the first element in each line is treated as an example ID and is excluded.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents an example and each column represents an attribute.
    """
    result = []

    for line in lines:
        attributes = line.split(',')
        attributes.pop(0) if example_id else None   # remove example id if applicable

        if str.isdigit(attributes[-1]):
            class_name = attributes[-1].strip()
            attributes[-1] = str(class_id.get(class_name, -1))
        result.append([float(val) for val in attributes])

    return np.array(result)

def array_to_lines(array):
    """
    Converts a NumPy array of floats to a list of strings, where each row of the array becomes a comma-separated line.

    Parameters:
        array (np.ndarray): The input array to convert, where each row represents an example and each column represents an attribute.

    Returns:
        list of str: A list of strings, where each string represents a row of the array in CSV format.
    """
    lines = []

    for row in array:
        line = [str(int(col)) for col in row]
        lines.append(','.join(line) + '\n')

    return lines

def get_attribute_bins(examples, num_bins=15):
    """
    Separates data into a specified number of bins for each attribute.
    Returns a list of dictionaries, where each dictionary maps an interval to an integer.
    
    Parameters:
        examples (numpy.ndarray): 2D array where each row represents an example and each column represents an attribute.
        num_bins (int): Number of bins to divide each attribute into.
    
    Returns:
        list of dict: A list where each dictionary contains bin intervals as keys and bin numbers as values.
    """
    # examples row: example value, column: attribute number
    attributes = examples.T  # attributes row: attribute number, column: example value

    attribute_bins = [] # list of dictionaries that map bin intervals to bin numbers

    num_values = len(attributes[0]) # number of example values in the matrix

    if num_bins > num_values:
            num_bins = num_values

    values_per_bin = num_values // num_bins
    values_left_over = num_values % num_bins

    for i in range(len(attributes) - 1):    # ignores class attribute
        attribute = np.sort(attributes[i])

        bin_edges = [attribute[0]]
        attribute_bin = {}  # dictionary mapping bin intervals to bin numbers for this particular attribute
        
        index = 0
        for j in range(num_bins):
            index += values_per_bin + (1 if j < values_left_over else 0)    # evenly distributes remainder so that bins are similarly sized
            bin_edges.append(attribute[index] if index < num_values else attribute[-1])

            min_val = bin_edges[j]
            max_val = bin_edges[j + 1]
            attribute_bin[(min_val, max_val)] = j + 1

        attribute_bins.append(attribute_bin)

    return attribute_bins

def get_bin_string(attribute_bins):
    """
    Converts the bin mappings into a human-readable string format.
    
    Parameters:
        attribute_bins (list of dict): List where each dictionary contains bin intervals and their corresponding bin numbers.
    
    Returns:
        str: Human-readable string representing the bin mappings for each attribute.
    """
    bin_string = ''
    for i, attribute_bin in enumerate(attribute_bins):
        bin_string += f"Feature {i + 1} bins:\n"
        for (min_val, max_val), bin_num in attribute_bin.items():
            bin_string += f"  {bin_num}: ({min_val:.5f},{max_val:.5f})\n"
        bin_string += "\n"
    return bin_string

def bin_attributes(examples, attribute_bins):
    """
    Discretizes the attributes based on the provided bin mappings.
    
    Parameters:
        examples (numpy.ndarray): 2D array where each row represents an example and each column represents an attribute.
        attribute_bins (list of dict): List where each dictionary contains bin intervals and their corresponding bin numbers.
    
    Returns:
        numpy.ndarray: 2D array with discretized attribute values.
    """
    # examples row: example value, column: attribute number
    attributes = examples.T  # attributes row: attribute number, column: example value

    for i in range(len(attributes) - 1):    # ignore class attribute
        attribute = attributes[i]
        attribute_bin = attribute_bins[i]

        for j in range(len(attribute)):
            for (min_val, max_val) in attribute_bin:
                if min_val < attribute[j] <= max_val:
                    attributes[i][j] = int(attribute_bin[(min_val, max_val)])
                    break

    return attributes.T

def add_noise(examples, noise_level=0.10):
    """
    Adds noise to a subset of attributes in the dataset by shuffling values within an attribute.

    Parameters:
        examples (numpy.ndarray): 2D array where each row represents an example and each column represents an attribute.
        noise_level (float): Fraction of attributes to add noise to. Default is 0.10 (10%).

    Returns:
        numpy.ndarray: 2D array with noise added to a fraction of the attributes.
    """
    noise_level = noise_level if (0 < noise_level < 1) else 0.10 # Ensure noise_level is between 0 and 1
    attributes = examples.T  # attributes - row: attribute number, column: example value
    num_attributes = len(attributes)-1   # ignore class attribute
    num_noisy = min(ceil(num_attributes * noise_level), num_attributes)    # number of attributes to add noise to

    for i in sample(range(num_attributes), num_noisy):
        np.random.shuffle(attributes[i])

    return attributes.T