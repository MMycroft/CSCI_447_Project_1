import sys, os, numpy as np
from classes.learnablenb import LearnableNB
from classes import cancer, glass, votes, iris, soybean

def main():
  if len(sys.argv) < 3:
    print("Usage: python learn_processor.py <input_file> <output_file>")
    return

  in_file_name = sys.argv[1]
  out_file_name = sys.argv[2]

  prefix = in_file_name.split('_')[0]
  suffix = in_file_name.split('_')[1]

  in_file_folder = prefix + '_data'

  in_file = os.path.join(in_file_folder, in_file_name)
  out_file = os.path.join("loss", out_file_name)

  learnable_classes = {
    "cancer": cancer.Cancer,
    "glass": glass.Glass,
    "votes": votes.Votes,
    "iris": iris.Iris,
    "soybean": soybean.Soybean
  }

  Learnable = learnable_classes.get(suffix)

  if Learnable is None:
    print(f"Error: No class for '{suffix}'.")
    return

  try:
    data: np.array(int) = np.loadtxt(in_file, delimiter=",")
  except Exception as e:
    print(f"Error loading data from '{in_file}': {e}")
    return

  folds: list[np.array(int)] = np.array_split(data, 10)

  losses = []
  for i in range(len(folds)):
    test_data: np.array(int) = folds[i]
    train_data: np.array(int) = np.concatenate(folds[0:i] + folds[i+1:])

    test_experiments: list['LearnableNB'] = [Learnable(data, False) for data in test_data]
    train_experiments: list['LearnableNB'] = [Learnable(data, True) for data in train_data]

    Learnable.naive_bayes_trainer(train_experiments)
    Learnable.naive_bayes_classifier(test_experiments)

    losses.append([Learnable.zero_one_loss(test_experiments), Learnable.f1_score_loss(test_experiments)])

  try:
    with open(out_file, 'w') as out_f:
      for loss in losses:
        out_f.write(f"{loss[0]},{loss[1]}\n")

  except FileNotFoundError:
    print(f"Error: File '{out_file}' not found.")
  except IOError as e:
    print(f"Error writing file '{out_file}': {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")


  if __name__ == "__main__":
    main()