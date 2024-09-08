import sys, os
from processors import iris, glass, cancer, votes, soybean

def main():
    if len(sys.argv) < 3:
        print("Usage: python processor_functions.py <input_file_name> <output_file_name> <doc_file_name>")
        return

    in_file_name = sys.argv[1]
    out_file_name = sys.argv[2]
    doc_file_name = sys.argv[3]

    in_file = os.path.join("data", in_file_name)
    clean_file = os.path.join("clean_data", out_file_name)
    noise_file = os.path.join("noisy_data", "noisy_" + out_file_name)
    doc_file = os.path.join("docs", doc_file_name)

    process = {
        "breast-cancer-wisconsin.data": cancer.process_data,
        "glass.data": glass.process_data,
        "house-votes-84": votes.process_data,
        "iris.data": iris.process_data,
        "soybean-small": soybean.process_data
    }

    process_function = process.get(in_file_name)

    if process_function is None:
        print(f"Error: No processing function for '{in_file_name}'.")
        return

    try:
        # read input data file
        with open(in_file, 'r') as in_f:
            in_file_lines: list[str] = in_f.readlines()
        # clean_lines: list[str] noisy_lines: list[str] documentation: str
        clean_lines, noisy_lines, documentation = process_function(in_file_lines)
        # write data files
        with open(clean_file, 'w') as clean_f:
            clean_f.writelines(clean_lines)
        with open(noise_file, 'w') as noise_f:
            noise_f.writelines(clean_lines)
        # write documentation if provided
        if documentation:
            with open(doc_file, 'w') as doc_f:
                doc_f.write(documentation)

    except FileNotFoundError:
        print(f"Error: File '{in_file_name}' not found.")
    except IOError as e:
        print(f"Error reading or writing file '{in_file_name}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()