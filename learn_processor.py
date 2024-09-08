import sys, os, numpy as np
from processors import cancer, glass, votes, iris, soybean
from utils import naive_bayes as nb


def main():
    if len(sys.argv) < 2:
        print("Usage: python learn_processor.py <input_file> <output_file>")
        return

    in_file_name = sys.argv[0]
    out_file_name = sys.argv[1]

    prefix = in_file_name.split('_')[0]
    suffix = in_file_name.split('_')[1]

    in_file_folder = prefix + '_data'

    in_file = os.path.join(in_file_folder, in_file_name)
    out_file = os.path.join("learned_data", out_file_name)

    class_names = {
        "cancer": cancer.class_name_id,
        "glass": glass.class_name_id,
        "votes": votes.class_name_id,
        "iris": iris.class_name_id,
        "soybean": soybean.class_name_id,
    }

    class_name_id = class_names.get(suffix)

    if class_name_id is None:
        print(f"Error: No class name for '{suffix}'.")
        return

    examples = np.loadtxt(in_file, delimiter=",")

    nb.naive_bayes_classifier(examples)





    if __name__ == "__main__":
        main()