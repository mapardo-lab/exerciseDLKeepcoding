import matplotlib.pyplot as plt
import numpy  as np  
import pandas as pd
import seaborn as sns
import optuna
import os
from optuna.importance import get_param_importances
import pickle

def optuna_results(study):
  """
  Prints the best trial results from an Optuna study, 
  including hyperparameter values and their relative 
  importance scores.
  """
  # print best results
  print("Best trial:")
  trial = study.best_trial
  importances = get_param_importances(study)

  print("  Value: ", trial.value)
  print("  Params: ")
  print("\t\t\tValue\t\tImportance ")
  for key, value in trial.params.items():
    print(f"    {key}:\t{value:.5f}\t\t{importances[key]:.2f}")

def optuna_init(sampler, outputdir, study_id):
  """
  Initializes an Optuna study for hyperparameter optimization, ensuring a clean start by removing
  any existing study with the same name. The study is persisted in an SQLite database for
  potential resumption of optimization.
  """
  try:
    # remove study if exists
    optuna.delete_study(study_name = study_id + "_optimization", 
                        storage=os.path.join("sqlite:///", 
                        outputdir, study_id + "_study.sqlite3"))
  except:
    pass

  # build optuna study
  study = optuna.create_study(study_name = study_id + "_optimization", direction="maximize",
                            storage=os.path.join("sqlite:///", outputdir, study_id + "_study.sqlite3"),
                            sampler=sampler)
  return study

def save_metrics_optuna(trial, results, outputdir):
  """
  Saves evaluation metrics from an Optuna trial to a pickle file and stores the file path
  as a user attribute in the trial object for later reference.
  """
  output_metrics_file = os.path.join(outputdir,f"metrics_{trial.number}.pkl")
  with open(output_metrics_file, "wb") as f:
      pickle.dump(results, f)

  # save path for output as user parameter
  trial.set_user_attr("metrics_path", output_metrics_file)