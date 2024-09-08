import sys, os
from processors import iris, glass, cancer, votes, soybean

def main():
    if len(sys.argv) < 3:
        print("Usage: python processor_functions.py <input_file> <output_file> <doc_file>")
        return

    in_file = os.path.join("data", sys.argv[1])
    out_file = os.path.join("clean_data", sys.argv[2])
    doc_file = os.path.join("docs", sys.argv[3])

    in_file_name = os.path.basename(in_file)

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
        with open(in_file, 'r') as in_f:
            in_file_lines: list[str] = in_f.readlines()
        # clean_lines: list[str] documentation: str
        clean_lines, documentation = process_function(in_file_lines)

        with open(out_file, 'w') as out_f:
            out_f.writelines(clean_lines)
        if documentation:   # write documentation if provided
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