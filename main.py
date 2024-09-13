import sys, os
import numpy as np
from classes.learnablenb import LearnableNB
from classes import iris, glass, cancer, votes, soybean


def main():
  # retrieve program arguments
  if len(sys.argv) < 2:
    print("Usage: python main.py <input_file>")
    return

  names = {
    "breast-cancer-wisconsin.data": 'cancer',
    "glass.data": 'glass',
    "house-votes-84.data": 'votes',
    "iris.data": 'iris',
    "soybean-small.data": 'soybean'
  }

  input_file = sys.argv[1]
  name = names.get(input_file)
  if name is None:
    print(f"Error: No class for '{input_file}'.")
    return
# determine appropriate file pathes
  in_file = os.path.join("raw_data", input_file)
  clean_file = os.path.join("processed_data", name + "_clean.data")
  noisy_file = os.path.join("processed_data", name + "_noisy.data")
  bin_file = os.path.join("bin_docs", name + "_bins.txt")
  clean_loss_file = os.path.join("loss", name + "_loss_clean.txt")
  noisy_loss_file = os.path.join("loss", name + "_loss_noisy.txt")

  learnable_classes = {
    "cancer": cancer.Cancer,
    "glass": glass.Glass,
    "votes": votes.Votes,
    "iris": iris.Iris,
    "soybean": soybean.Soybean
  }
  # acquire class based on system argument
  learnable_class = learnable_classes.get(name)
  if learnable_class is None:
    print(f"Error: No class for '{name}'.")
    return

  try:
    with open(in_file, 'r') as in_f:
      in_file_lines: list[str] = in_f.readlines()
    # clean_lines: list[str] noisy_lines: list[str] documentation: str
      clean_lines, noisy_lines, documentation = learnable_class.process_data(in_file_lines)
    # write output files
    with open(clean_file, 'w') as clean_f:
      clean_f.writelines(clean_lines)
    with open(noisy_file, 'w') as noisy_f:
      noisy_f.writelines(noisy_lines)
    # write documentation if provided
    if documentation:
      with open(bin_file, 'w') as doc_f:
        doc_f.write(documentation)

  except FileNotFoundError:
    print(f"Error: File for '{name}' not found.")
  except IOError as e:
    print(f"Error reading or writing file for '{name}': {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")

  for i in range(len(clean_lines)):
    clean_lines[i] = clean_lines[i].split(',')
    noisy_lines[i] = noisy_lines[i].split(',')
  clean_data = np.array(clean_lines, dtype=int)
  noisy_data = np.array(noisy_lines, dtype=int)
  print(clean_data)
  print(noisy_data)

  def n_fold_cross_validation(data: np.array(int)):
    folds: list[np.array(int)] = np.array_split(data, 10)

    losses = []
    for i in range(len(folds)):
      test_data: np.array(int) = np.array(folds[i], dtype=int)
      train_data: np.array(int) = np.array(np.concatenate(folds[0:i] + folds[i + 1:]), dtype=int)
      test_experiments: list['LearnableNB'] = []
      for data in test_data:
        learnable_class(data, False)

      train_experiments: list['LearnableNB'] = [learnable_class(data, True) for data in train_data]

      learnable_class.naive_bayes_trainer(train_experiments)
      learnable_class.naive_bayes_classifier(test_experiments)
      print(test_experiments)

      losses.append([learnable_class.zero_one_loss(test_experiments), learnable_class.f1_score_loss(test_experiments)])

    loss_string = ''
    for i in range(len(losses)):
      loss_string += f"{losses[i][0]},{losses[i][1]}\n"

    return loss_string

  clean_loss = n_fold_cross_validation(clean_data)
  noisy_loss = n_fold_cross_validation(noisy_data)

  try:
    with open(clean_loss_file, 'w') as c_loss_f, open(noisy_loss_file, 'w') as n_loss_f:
      c_loss_f.write(clean_loss)
      n_loss_f.write(noisy_loss)

  except FileNotFoundError:
    print(f"Error: File for loss '{name}' not found.")
  except IOError as e:
    print(f"Error writing file for loss '{name}': {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
  main()