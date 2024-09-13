import sys, os
from classes import iris, glass, cancer, votes, soybean


def main():
  if len(sys.argv) < 3:
    print("Usage: python data_processor.py <input_file> <output_file> <doc_file>")
    return

  in_file_name = sys.argv[1]
  out_file_name = sys.argv[2]
  doc_file_name  = sys.argv[3]

  in_file = os.path.join("raw_data", in_file_name)
  clean_file = os.path.join("clean_data", "clean_" + out_file_name)
  noisy_file = os.path.join("noisy_data", "noisy_" + out_file_name)
  doc_file = os.path.join("docs", doc_file_name)

  process = {
    "breast-cancer-wisconsin.data": cancer.Cancer.process_data,
    "glass.data": glass.Glass.process_data,
    "house-votes-84.data": votes.Votes.process_data,
    "iris.data": iris.Iris.process_data,
    "soybean-small.data": soybean.Soybean.process_data
  }

  process_function = process.get(in_file_name)

  if process_function is None:
    print(f"Error: No processing function for '{in_file_name}'.")
    return

  try:
    with open(in_file, 'r') as in_f:
      in_file_lines: list[str] = in_f.readlines()
    # clean_lines: list[str] noisy_lines: list[str] documentation: str
    clean_lines, noisy_lines, documentation = process_function(in_file_lines)
    # write output files
    with open(clean_file, 'w') as clean_f:
      clean_f.writelines(clean_lines)
    with open(noisy_file, 'w') as noisy_f:
      noisy_f.writelines(clean_lines)
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