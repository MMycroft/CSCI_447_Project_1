"""
data_processor\n
Read data from files, discretize it, generate noise,
then write the clean and noisy data to new files.\n
Authors: Matthew Mycroft, Ryan Dupuis
"""

import sys, os
from processors import iris, glass, cancer, votes, soybean


def main(argv):
    """
    Read data from files, discretize it, then write the clean and noisy data to new files.\n
    Parameters: argv (list of str): command line arguments\n
    Usage: python processor_functions.py <input_file> <output_file> <doc_file>\n
    Returns: None\n
    Exceptions: FileNotFoundError, IOError, Exception
    """

    # Print usage if argv is of incorrect length
    if len(argv) < 3:
        print("Usage: python processor_functions.py <input_file> <output_file> <doc_file>")
        return

    # Define files
    in_file = os.path.join("data", argv[1])
    out_file = os.path.join("clean_data", argv[2])
    doc_file = os.path.join("docs", argv[3])

    in_file_name = os.path.basename(in_file)

    # Define dictionary of data to be processed
    process = {
        "breast-cancer-wisconsin.data": cancer.process_data,
        "glass.data": glass.process_data,
        "house-votes-84": votes.process_data,
        "iris.data": iris.process_data,
        "soybean-small": soybean.process_data
    }

    process_function = process.get(in_file_name)

    # Make sure there is a process function
    if process_function is None:
        print(f"Error: No processing function for '{in_file_name}'.")
        return

    try:
        # Open files
        with open(in_file, 'r') as in_f:
            in_file_lines: list[str] = in_f.readlines()
        # Obtain clean_lines: list[str] and documentation: str
        clean_lines, documentation = process_function(in_file_lines)

        # Write clean lines
        with open(out_file, 'w') as out_f:
            out_f.writelines(clean_lines)

        # Write documentation if provided
        if documentation:
            with open(doc_file, 'w') as doc_f:
                doc_f.write(documentation)

    # Exceptions
    except FileNotFoundError:
        print(f"Error: File '{in_file_name}' not found.")
    except IOError as e:
        print(f"Error reading or writing file '{in_file_name}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main(sys.argv)