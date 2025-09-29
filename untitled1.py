pruner = MedianPruner(
    n_startup_trials=5,       # No pruning until 5 trials complete
    n_warmup_steps=2,         # Wait 2 steps before considering pruning
)



# Report to pruner
trial.report(current_score, step=n_rounds)
        
# Check for pruning
if trial.should_prune():
    print(f"Trial {trial.number} pruned at {n_rounds} rounds")
    raise optuna.TrialPruned()
         
# Update best score
best_score = max(best_score, current_score)