# minituna

Simplified Optuna implementation for new contributors.

## minituna_v1

https://github.com/optuna/optuna/blob/master/examples/quadratic_simple.py

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
$ python example_quadratic.py
trial_id=0 is completed with value=13.275505983863615
trial_id=1 is completed with value=34.1227147864478
trial_id=2 is completed with value=11.199369841219616
trial_id=3 is completed with value=15.051955617824198
trial_id=4 is completed with value=26.7725919634248
trial_id=5 is completed with value=42.093131408456784
trial_id=6 is completed with value=17.01377949289734
trial_id=7 is completed with value=8.868050512421352
trial_id=8 is completed with value=11.002184635683296
trial_id=9 is completed with value=7.905097506668502
Best trial: FrozenTrial(trial_id=9, state='completed', value=7.905097506668502, params={'x': 4.654302572011295, 'y': 2.726592753837246})
```

## minituna_v2

https://github.com/optuna/optuna/blob/master/examples/sklearn_simple.py

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

```console
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

## minituna_v3

https://github.com/optuna/optuna/blob/master/examples/pruning/simple.py

```python
import minituna_v3 as minituna

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise minituna.TrialPruned()

    accuracy = clf.score(valid_x, valid_y)
    return 1 - accuracy


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 30)

    best_trial = study.best_trial
    print(
        f"Best trial: id={best_trial.trial_id} value={best_trial.value} params={best_trial.params}"
    )
```
