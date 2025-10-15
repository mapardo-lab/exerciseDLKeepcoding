import numpy  as np  
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import os
import pickle
import statistics
from sklearn.model_selection import cross_validate
from torch.utils.data import DataLoader
from utilsTrain import plot_training_curves
from utilsTrain import ResultTrain

class ObjectiveFunction():
    """
    Base class for defining objective functions with configurable hyperparameter search spaces.
    """
    def __init__(self, fixed_params, search_space, score, run_name):
        self.fixed_params = fixed_params
        self.search_space = search_space
        self.score = score
        self.run_name = run_name

    def get_optimizable_params(self, trial): 
        # Define optimizable parameters based on search_space config
        optimizable_params = {}
        for module_name, module_config in self.search_space.items():
            optimizable_params[module_name] = {}
            for param_name, param_config in module_config.items():
                if param_config['type'] == 'float':
                    if param_config.get('log', False):
                        optimizable_params[module_name][param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'], 
                            log=True
                        )
                    else:
                        optimizable_params[module_name][param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                elif param_config['type'] == 'int':
                    if param_config.get('exp2', False):
                        optimizable_params[module_name][param_name] = 2**trial.suggest_int(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                    else:
                        optimizable_params[module_name][param_name] = trial.suggest_int(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                elif param_config['type'] == 'categorical':
                    optimizable_params[module_name][param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config['choices']
                    )
        return optimizable_params

class ObjectiveFunctionML(ObjectiveFunction):
    """
    Objective function for machine learning models using cross-validation for hyperparameter optimization.
    """
    def __init__(self, X, y, model, fixed_params, search_space, scoring, score, cv, run_name):
        super().__init__(fixed_params, search_space, score, run_name)
        self.model = model
        self.scoring = scoring
        self.X = X
        self.y = y 
        self.cv = cv
    
    def __call__(self, trial):
        params =  self.get_optimizable_params(trial)
        model_params = {**self.fixed_params['model'], **params['model']}

        trial.set_user_attr('run', self.run_name)
        
        model = self.model(**model_params)

        cv_results = cross_validate(model, self.X, self.y, cv=self.cv, scoring=self.scoring, return_train_score=True, n_jobs=1)
        score = cv_results['test_' + self.score].mean()

        results_as_lists = {key: value.tolist() for key, value in cv_results.items()}
        trial.set_user_attr('model', f'{type(model)}') 
        trial.set_user_attr('k-fold', self.cv)
        trial.set_user_attr('score', self.score)
        trial.set_user_attr('train_results', results_as_lists) 

        return score

class ObjectiveFunctionDL(ObjectiveFunction):
    """
    Objective function for deep learning models with training loop and pruning capabilities.
    """
    def __init__(self, train_dataset, val_dataset, train, train_config, 
                 num_epochs, fixed_params, search_space, score, run_name):
        super().__init__(fixed_params, search_space, score, run_name)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train = train
        self.train_config = train_config
        self.num_epochs = num_epochs
      
    def __call__(self, trial):
        params = self.get_optimizable_params(trial)
        trial.set_user_attr('run', self.run_name)

        training = self.train(self.train_config, params, self.fixed_params)
        train_loader = DataLoader(dataset = self.train_dataset, shuffle=True, **params['data_loader'], **self.fixed_params['data_loader']) 
        val_loader = DataLoader(dataset = self.val_dataset, shuffle=False, **params['data_loader'], **self.fixed_params['data_loader']) 

        trial.set_user_attr('num_epochs', self.num_epochs)
        trial.set_user_attr('fixed_params', self.fixed_params)

        for epoch in range(self.num_epochs):
            training.train_epoch(train_loader)
            training.eval_model(val_loader)

            # Report to pruner
            result = training.results_val.results[self.score][-1]
            trial.report(result, step = epoch)

            # Check for pruning
            if trial.should_prune():
                # save metrics
                trial.set_user_attr('train_results', training.results_train.results)
                trial.set_user_attr('val_results', training.results_val.results)
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()
        
        # save metrics
        trial.set_user_attr('train_results', training.results_train.results)
        trial.set_user_attr('val_results', training.results_val.results)
    
        # return score
        return training.results_val.results[self.score][-1]

def create_study(study_params, user_attr):
    """
    Creates and configures an Optuna study with specified parameters and user attributes.
    """
    study = optuna.create_study(**study_params)
    for attr in user_attr:
        study.set_user_attr(attr[0],attr[1])
    return study

def info_studies_from_storage(db_url):
    """
    Get study names and best scores from Optuna database storage.
    """
    
    storage = optuna.storages.RDBStorage(url=db_url)
    list_studies = storage.get_all_studies()
    
    studies_data = []
    
    for study_summary in list_studies:
        study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        study_info = {
            'study_name': study.study_name,
            'best_score': study.best_value,
            'description': study.user_attrs['comments']
        }
        
        studies_data.append(study_info)
    
    return pd.DataFrame(studies_data)
    
def info_trials_from_study(trials, extended='False'):
    """
    Get a dataframe with info from all trials of a study
    """
    data = []
    for trial in trials:
        # Get run name, default to 'unknown' if not present
        run_name = trial.user_attrs.get('run', None)
        
        # Create row with run name and all parameters
        row = {'run': run_name,
               'number': trial.number,
               'state': trial.state,
               'values': trial.values,
               'datetime_start': trial.datetime_start,
               'datetime_complete': trial.datetime_complete
              }
        # Add all parameters from this trial
        for param_name, param_value in trial.params.items():
            row[param_name] = param_value
        # Add all distributions from this trial
        for param_name, distribution in trial.distributions.items():
            row[f"dist_{param_name}"] = distribution
        data.append(row)
    return pd.DataFrame(data)

def user_attr_study(study):
    """
    Display specific user attributes from an Optuna study in a readable format.
    """
    # TODO Show in order
    # TODO show features with transformations
    for key, value in study.user_attrs.items():
        if key in ['dataset', 'description', 'script', 'target']:
            print(f"{key} -- {value}")
        
def remove_study_from_storage(storage_url, study_name):
    """
    Delete an Optuna study from the specified storage.
    """
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"Study '{study_name}' deleted successfully")
    except KeyError:
        print(f"Study '{study_name}' not found")  
        

def get_datetime_runs(df):
    """
    Generate a summary of trial runs with datetime information.
    """
    summary_datetime = df.groupby('run').agg(
        datetime_start=('datetime_start', 'min'),
        datetime_complete=('datetime_complete', 'max')
    ).reset_index()
    summary_datetime['datetime_start'] = summary_datetime['datetime_start'].dt.floor('s')
    summary_datetime['datetime_complete'] = summary_datetime['datetime_complete'].dt.floor('s')
    return summary_datetime

def get_score_runs(df):
    """
    Generate a summary of trial runs with score statistics for completed trials.
    """
    summary_runs = df.groupby('run').agg(
        num_trials=('run', 'count')).reset_index()
    df_complete = df[df['state'] == 1].copy()
    df_complete['score'] = df_complete['values'].apply(lambda x: x[0] if len(x) > 0 else None)
    summary_score = df_complete.groupby('run').agg(
        completed_trials=('run', 'count'),  # or use 'size'
        best_score=('score', 'max'),
    ).reset_index()
    df_output = pd.merge(summary_runs, summary_score, on='run')
    return df_output

def get_range_params_runs(df, params):
    """
    Return parameter value ranges (min, max) for each run's completed trials.
    """
    summary_params = []
    df_complete = df[df['state'] == 1]
    for run_name, group in df_complete.groupby('run'):
        run_summary = {'run': run_name}
        
        # For each parameter column, calculate min and max
        param_columns = [col for col in group.columns if col in params]
        
        for param in param_columns:
            run_summary[param] = (round(group[param].min(),4), round(group[param].max(),4))
        
        summary_params.append(run_summary)
    return  pd.DataFrame(summary_params)
    
def get_dist_params_runs(df, params):
    """
    Return parameter value ranges (min, max) for each run's completed trials.
    """
    df_runs = df.groupby('run').first().reset_index()
    dist_params = list(map(lambda x: 'dist_' + x, params))
    columns = ['run'] + dist_params
    return df_runs[columns]
    
def get_params_trials(trials):
    """
    Extract unique parameter names from a list of Optuna trials.
    """
    params = set()
    for trial in trials:
        params.update(trial.params.keys())
    return params
    
def info_runs_from_study(trials, extended = False):
    """
    Aggregate trial information into a comprehensive run summary DataFrame.
    """
    df = info_trials_from_study(trials)
    df_score = get_score_runs(df)
    if extended:
        df_datetime = get_datetime_runs(df)
        params = get_params_trials(trials)
        df_range_params = get_range_params_runs(df, params)
        df_dist_params = get_dist_params_runs(df, params)
        df_merged = pd.merge(df_score, df_datetime, on = 'run')
        df_merged = pd.merge(df_merged, df_dist_params, on = 'run')
        df_merged = df_merged.sort_values('datetime_start')
        df_output = df_merged
    else:
        df_output = df_score
    return df_output
    
def plot_slice(study, params=[]):
    """
    Generate slice plots of objective values versus specified hyperparameters for a study's completed trials.
    """
    # TODO Carefull not more than six hyperparameters
    trials = study.trials
    if len(params) == 0:
        params = get_params_trials(trials)
    df = info_trials_from_study(trials)
    df_complete = df[df['state'] == 1].copy()
    df_complete['score'] = df_complete['values'].apply(lambda x: x[0] if len(x) > 0 else None)

    # Columns to plot
    n_plots = len(params)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(2*n_plots, 4), sharey=True)
    axes = axes.flatten()

    best_score =max(df_complete['score'])
    
    for i, col in enumerate(params):
        axes[i].axhline(y=best_score, color='red', linestyle='--', linewidth=2, alpha=0.3)
        axes[i].scatter(df_complete[col], df_complete['score'], alpha=0.9, color='lightblue', s=30, edgecolor='grey')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Objetive value')
        axes[i].grid(True, alpha=0.3)     
        # Hide y-axis label for all except first plot
        if i > 0:
            axes[i].set_ylabel('')

    plt.tight_layout()
    plt.show()
    
def optuna_results(study):
  """
  Prints the best trial results from an Optuna study, 
  including hyperparameter values
  """
  # print best results
  print("Best trial:")
  trial = study.best_trial
  #importances = get_param_importances(study)

  print("  Value: ", trial.value)
  print("  Params: ")
  print("\t\t\tValue")
  for key, value in trial.params.items():
    print(f"    {key}:\t{value:.5f}")
  #print("\t\t\tValue\t\tImportance ")
  #for key, value in trial.params.items():
  #  print(f"    {key}:\t{value:.5f}\t\t{importances[key]:.2f}")

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
    
def plot_train_nn(trial):
    """
    Plot training and validation curves from neural network training results stored in an Optuna trial.
    """
    df = pd.DataFrame(trial.user_attrs['train_results'])
    train_losses = df['train_losses']
    val_losses = df['val_losses']
    train_accs = df['train_accs']
    val_accs = df['val_accs']
    num_epochs = df.shape[0]
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc=None)

def best_trial_scores_ML(study, list_scores): 
    """
    Retrieves and computes the performance metrics from the best trial in an Optuna study (Machine Learning), 
    returning the model name along with the mean training and validation scores rounded 
    to three decimal places for concise evaluation. 
    """
    results = []

    model = study.best_trial.user_attrs['model'].split('.')[-1].replace("'>","")
    results.append(model)
    for score in list_scores:
        results.append(round(statistics.mean(study.best_trial.user_attrs['train_results'][score]), 3))
    return results

def best_trial_scores_DL(study, list_scores): 
    """
    Retrieves and computes the performance metrics from the best trial in an Optuna study (Deep Learning), 
    returning the model name along with the mean training and validation scores rounded 
    to three decimal places for concise evaluation. 
    """
    results = []

    model = study.best_trial.user_attrs['model'].split('.')[-1].replace("'>","")
    results.append(model)
    for score in list_scores:
        results.append(round(study.best_trial.user_attrs['train_results'][score][-1], 3))
    return results