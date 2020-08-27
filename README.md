# minituna

Simplified Optuna implementation for new contributors.

## minituna_v1

```python
import minituna_v1 as minituna

def objective(trial: minituna.Trial) -> float:
    x = trial.suggest_uniform("x", 0, 10)
    y = trial.suggest_uniform("y", 0, 10)
    return (x - 3) ** 2 + (y - 5) ** 2

if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 10)
    print("Best trial:", study.best_trial)
```

```console
$ python example.py
trial_id=0 is completed with value=0.11539445997147783
trial_id=1 is completed with value=11.554241070552933
trial_id=2 is completed with value=6.956107970175709
trial_id=3 is completed with value=8.452114918501348
trial_id=4 is completed with value=1.2053458904783958
trial_id=5 is completed with value=36.097821542061666
trial_id=6 is completed with value=61.63078249453704
trial_id=7 is completed with value=3.256384442457427
trial_id=8 is completed with value=16.07359934215089
trial_id=9 is completed with value=10.693346790638826
Best trial: FrozenTrial(trial_id=0, state='completed', value=0.11539445997147783, params={'x': 2.9999186187424467, 'y': 5.33969759102556}, distributions={'x': Distribution(low=0, high=10), 'y': Distribution(low=0, high=10)})
```

## minituna_v2

```python
import minituna_v2 as minituna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(
        classifier_obj, x, y, n_jobs=-1, cv=3
    )
    accuracy = score.mean()
    return 1 - accuracy


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 10)

    best_trial = study.best_trial
    print(
        f"Best trial: id={best_trial.trial_id} value={best_trial.value} params={best_trial.params}"
    )
```

```
$ python example_sklearn.py
trial_id=0 is completed with value=0.040000000000000036
trial_id=1 is completed with value=0.6799999999999999
trial_id=2 is completed with value=0.033333333333333326
trial_id=3 is completed with value=0.040000000000000036
trial_id=4 is completed with value=0.046666666666666745
trial_id=5 is completed with value=0.6799999999999999
trial_id=6 is completed with value=0.053333333333333344
trial_id=7 is completed with value=0.6799999999999999
trial_id=8 is completed with value=0.040000000000000036
trial_id=9 is completed with value=0.6799999999999999
Best trial: id=2 value=0.033333333333333326 params={'classifier': 'RandomForest', 'rf_max_depth': 4}
```
