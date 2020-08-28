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

## minituna_v2 : More distributions support

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

## minituna_v3 : Pruning

https://github.com/optuna/optuna/blob/master/examples/visualization/plot_study.ipynb

```python
import minituna_v3 as minituna

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


mnist = fetch_openml(name="Fashion-MNIST", version=1)
classes = list(set(mnist.target))

# For demonstrational purpose, only use a subset of the dataset.
n_samples = 4000
data = mnist.data[:n_samples]
target = mnist.target[:n_samples]

x_train, x_valid, y_train, y_valid = train_test_split(data, target)


def objective(trial):
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(
            [trial.suggest_int("n_units_l{}".format(i), 32, 64) for i in range(3)]
        ),
        learning_rate_init=trial.suggest_float("lr_init", 1e-5, 1e-1, log=True),
    )

    for step in range(100):
        clf.partial_fit(x_train, y_train, classes=classes)
        accuracy = clf.score(x_valid, y_valid)
        error = 1 - accuracy

        # Report intermediate objective value.
        trial.report(error, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise minituna.TrialPruned()
    return error


if __name__ == "__main__":
    study = minituna.create_study(pruner=minituna.Pruner())
    study.optimize(objective, 30)

    best_trial = study.best_trial
    print(
        f"Best trial: id={best_trial.trial_id} value={best_trial.value} params={best_trial.params}"
    )
```

```console
$ python example_pruning.py
trial_id=0 is completed with value=0.645
trial_id=1 is completed with value=0.30200000000000005
trial_id=2 is completed with value=0.885
trial_id=3 is completed with value=0.891
trial_id=4 is completed with value=0.241
trial_id=5 is completed with value=0.36
trial_id=6 is completed with value=0.30600000000000005
trial_id=7 is pruned at step=0 value=0.868
trial_id=8 is completed with value=0.20199999999999996
trial_id=9 is pruned at step=0 value=0.874
trial_id=10 is completed with value=0.31699999999999995
trial_id=11 is pruned at step=0 value=0.9
trial_id=12 is pruned at step=1 value=0.835
trial_id=13 is completed with value=0.238
trial_id=14 is completed with value=0.19799999999999995
trial_id=15 is pruned at step=0 value=0.9299999999999999
trial_id=16 is completed with value=0.22799999999999998
trial_id=17 is pruned at step=0 value=0.882
trial_id=18 is completed with value=0.256
trial_id=19 is pruned at step=0 value=0.87
trial_id=20 is pruned at step=0 value=0.864
trial_id=21 is pruned at step=11 value=0.377
trial_id=22 is completed with value=0.22799999999999998
trial_id=23 is completed with value=0.236
trial_id=24 is completed with value=0.20299999999999996
trial_id=25 is pruned at step=0 value=0.895
trial_id=26 is pruned at step=0 value=0.899
trial_id=27 is pruned at step=0 value=0.858
trial_id=28 is completed with value=0.21899999999999997
trial_id=29 is pruned at step=45 value=0.267
Best trial: id=14 value=0.19799999999999995 params={'n_units_l0': 52, 'n_units_l1': 51, 'n_units_l2': 61, 'lr_init': 0.005854153852825279}
```
