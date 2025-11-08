import optuna
from optuna.trial import TrialState
import logging
import sys

study_name = 'dp-comp-sigma2'
storage_name = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(directions=["minimize", "maximize"], study_name=study_name, storage=storage_name, load_if_exists=True)
# df = study.trials_dataframe(attrs=("com_p", "sigma"))
for index, trial in enumerate(study.best_trials):
    print("Trials: ", index)
    print("Trial Performance ", trial.params)
    print("Trial values ", trial.values)

fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1], target_name="maximize")
fig.show()
